"""
Cross-validated training of the attention head on cached CLIP embeddings.

Why CV instead of a single 70/10/20 split: with ~1,000 videos, a single test
fold has ~200 samples, so a point-estimate AUC like 0.9535 carries a wide error
bar. K-fold gives out-of-fold (OOF) predictions for every video plus a mean±std,
which is the honest number to report and to compare against the baseline.

Leakage note: the default split is stratified by label only. If the same
identity/source appears in both train and test (common in DFD-style data), AUC is
optimistic. Pass `groups` (e.g. per-identity or per-source id) to switch to
StratifiedGroupKFold and get a generalization-honest estimate.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     GroupShuffleSplit, StratifiedShuffleSplit)

from src.attn_head import DeepfakeHead


def _iter_batches(n, bs, shuffle, device):
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for i in range(0, n, bs):
        yield order[i:i + bs]


@torch.no_grad()
def _predict(model, bank, idx, device, tta=True):
    """Return P(fake) for the given indices, optionally flip-TTA averaged."""
    model.eval()
    probs = []
    for b in _iter_batches(len(idx), 128, shuffle=False, device=device):
        sel = idx[b]
        variants = [False, True] if tta else [False]
        p = 0.0
        for fl in variants:
            batch = bank.batch(sel, train=False, flip=fl)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(batch["face"], batch["full"], batch["face_mask"],
                               batch["full_mask"], batch["face_valid"])
            p = p + F.softmax(logits.float(), dim=-1)[:, 1]
        probs.append((p / len(variants)).cpu())
    return torch.cat(probs).numpy()


def train_one_fold(bank, train_idx, val_idx, device, cfg):
    """Train a head on train_idx, early-stopping on val AUC. Returns (model, best_auc)."""
    torch.manual_seed(cfg["seed"])
    dim = getattr(bank, "feat_dim", 768)  # 768, or 768+forensic when the forensic stream is on
    model = DeepfakeHead(dim=dim, hidden=cfg["hidden"], heads=cfg["heads"], dropout=cfg["dropout"]).to(device)

    labels = bank.label[train_idx]
    n_real = int((labels == 0).sum()); n_fake = int((labels == 1).sum())
    class_w = torch.tensor([max(n_fake / max(n_real, 1), 1.0), 1.0], device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=1e-5)

    train_idx = train_idx.to(device)
    best_auc, best_state, no_improve = 0.0, None, 0

    for epoch in range(cfg["epochs"]):
        model.train()
        for b in _iter_batches(len(train_idx), cfg["batch_size"], shuffle=True, device=device):
            sel = train_idx[b]
            batch = bank.batch(sel, train=True, noise_frac=cfg["noise_frac"], frame_drop=cfg["frame_drop"])
            opt.zero_grad()
            logits = model(batch["face"], batch["full"], batch["face_mask"],
                           batch["full_mask"], batch["face_valid"])
            loss = F.cross_entropy(logits, batch["label"], weight=class_w,
                                   label_smoothing=cfg["label_smoothing"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        val_probs = _predict(model, bank, val_idx.to(device), device, tta=cfg["tta"])
        val_labels = bank.label[val_idx].cpu().numpy()
        auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.0
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


DEFAULT_CFG = dict(
    epochs=60, patience=12, batch_size=32, lr=3e-4, weight_decay=1e-2,
    dropout=0.3, hidden=256, heads=4, noise_frac=0.1, frame_drop=0.1,
    label_smoothing=0.05, tta=True, seed=42, n_splits=5, val_frac=0.15,
)


def run_cv(bank, device="cuda", cfg=None, groups=None, save_dir=None):
    """Run stratified (or grouped) k-fold CV. Returns a results dict with OOF AUC."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    y = bank.label.cpu().numpy()
    n = len(y)
    idx_all = np.arange(n)

    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
        split = splitter.split(idx_all, y, groups=np.asarray(groups))
        split_kind = "StratifiedGroupKFold (leakage-safe)"
    else:
        splitter = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
        split = splitter.split(idx_all, y)
        split_kind = "StratifiedKFold (label-only; may leak identity/source)"

    print(f"Cross-validation: {split_kind}\n")
    oof_probs = np.zeros(n, dtype=np.float64)
    fold_aucs, train_aucs, models = [], [], []
    groups_arr = np.asarray(groups) if groups is not None else None

    for fold, (tr, va) in enumerate(split, 1):
        # Nested split: carve the early-stopping val set out of TRAIN so the test
        # fold (va) is never used for model selection (no optimistic bias).
        if groups_arr is not None:
            inner = GroupShuffleSplit(n_splits=1, test_size=cfg["val_frac"], random_state=cfg["seed"])
            i_tr, i_va = next(inner.split(tr, groups=groups_arr[tr]))
        else:
            inner = StratifiedShuffleSplit(n_splits=1, test_size=cfg["val_frac"], random_state=cfg["seed"])
            i_tr, i_va = next(inner.split(tr, y[tr]))
        tr_inner, va_inner = tr[i_tr], tr[i_va]

        model, _ = train_one_fold(bank, torch.tensor(tr_inner), torch.tensor(va_inner), device, cfg)
        oof_probs[va] = _predict(model, bank, torch.tensor(va).to(device), device, tta=cfg["tta"])
        fold_auc = roc_auc_score(y[va], oof_probs[va])
        tr_auc = roc_auc_score(y[tr_inner], _predict(model, bank, torch.tensor(tr_inner).to(device), device, tta=False))
        fold_aucs.append(fold_auc); train_aucs.append(tr_auc); models.append(model)
        print(f"  Fold {fold}/{cfg['n_splits']}: test AUC = {fold_auc:.4f}   "
              f"(train {tr_auc:.4f}, gap {tr_auc-fold_auc:+.4f})")

    oof_auc = roc_auc_score(y, oof_probs)
    oof_acc = accuracy_score(y, (oof_probs > 0.5).astype(int))
    mean, std = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    train_mean = float(np.mean(train_aucs))

    print(f"\n{'='*52}")
    print(f"  OOF AUC:        {oof_auc:.4f}   (baseline 0.9535, delta {oof_auc-0.9535:+.4f})")
    print(f"  Fold AUC:       {mean:.4f} +/- {std:.4f}")
    print(f"  Train AUC:      {train_mean:.4f}   (train-OOF gap {train_mean-oof_auc:+.4f} <- overfitting check)")
    print(f"  OOF Accuracy:   {oof_acc:.4f}")
    print(f"{'='*52}")

    # Per-category OOF AUC (real pooled with each fake category)
    cats = np.array(bank.categories)
    per_cat = {}
    real_mask = (y == 0)
    for cat in sorted(set(cats[y == 1])):
        m = real_mask | (cats == cat)
        if len(set(y[m])) == 2:
            per_cat[cat] = round(float(roc_auc_score(y[m], oof_probs[m])), 4)
    if per_cat:
        print("\n  Per-category OOF AUC:")
        for cat, a in per_cat.items():
            print(f"    {cat:<28} {a:.4f}")

    results = {
        "oof_auc": round(oof_auc, 4), "oof_accuracy": round(oof_acc, 4),
        "fold_auc_mean": round(mean, 4), "fold_auc_std": round(std, 4),
        "fold_aucs": [round(a, 4) for a in fold_aucs],
        "train_auc_mean": round(train_mean, 4), "train_oof_gap": round(train_mean - oof_auc, 4),
        "per_category_auc": per_cat, "baseline_auc": 0.9535,
        "delta": round(oof_auc - 0.9535, 4), "split_kind": split_kind, "config": cfg,
    }

    if save_dir:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(models):
            torch.save(m.state_dict(), save_dir / f"head_fold{i+1}.pt")
        np.save(save_dir / "oof_probs.npy", oof_probs)
        with open(save_dir / "cv_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(models)}-model ensemble + results to {save_dir}")

    results["models"] = models
    results["oof_probs"] = oof_probs
    return results
