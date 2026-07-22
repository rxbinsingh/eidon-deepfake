"""
Regularization sweep on the cached embeddings.

The honest grouped CV showed a train-OOF gap of ~0.11 (moderate overfitting), so
this sweeps regularization strength and ranks configs by honest OOF AUC and by the
train-OOF gap. Because epochs are ~1 s on cached embeddings, a full grouped CV per
config takes ~1-2 min, so the whole sweep runs in minutes.

Each config runs the SAME leakage-safe, nested-validation CV as `cv_train.run_cv`
(early stopping never touches the scored fold), just quietly.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

from src.cv_train import DEFAULT_CFG, train_one_fold, _predict

# Curated grid (not full cartesian) spanning "current" -> "very strong" regularization.
DEFAULT_GRID = [
    {"name": "current",         "dropout": 0.3, "weight_decay": 1e-2, "frame_drop": 0.1,  "noise_frac": 0.1,  "hidden": 256, "heads": 4},
    {"name": "dropout0.5",      "dropout": 0.5, "weight_decay": 1e-2, "frame_drop": 0.1,  "noise_frac": 0.1,  "hidden": 256, "heads": 4},
    {"name": "wd5e-2",          "dropout": 0.3, "weight_decay": 5e-2, "frame_drop": 0.1,  "noise_frac": 0.1,  "hidden": 256, "heads": 4},
    {"name": "drop0.5+wd5e-2",  "dropout": 0.5, "weight_decay": 5e-2, "frame_drop": 0.1,  "noise_frac": 0.1,  "hidden": 256, "heads": 4},
    {"name": "small_head",      "dropout": 0.3, "weight_decay": 1e-2, "frame_drop": 0.1,  "noise_frac": 0.1,  "hidden": 128, "heads": 2},
    {"name": "more_aug",        "dropout": 0.3, "weight_decay": 1e-2, "frame_drop": 0.25, "noise_frac": 0.2,  "hidden": 256, "heads": 4},
    {"name": "moderate_all",    "dropout": 0.4, "weight_decay": 3e-2, "frame_drop": 0.2,  "noise_frac": 0.15, "hidden": 256, "heads": 4},
    {"name": "strong_all",      "dropout": 0.5, "weight_decay": 5e-2, "frame_drop": 0.25, "noise_frac": 0.2,  "hidden": 128, "heads": 2},
    {"name": "very_strong",     "dropout": 0.5, "weight_decay": 1e-1, "frame_drop": 0.3,  "noise_frac": 0.2,  "hidden": 128, "heads": 2},
]


def _quiet_grouped_cv(bank, groups, device, cfg):
    """Leakage-safe nested-val grouped CV; returns (oof_auc, fold_std, train_auc)."""
    y = bank.label.cpu().numpy()
    groups = np.asarray(groups)
    idx_all = np.arange(len(y))
    oof = np.zeros(len(y))
    fold_aucs, train_aucs = [], []

    skf = StratifiedGroupKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    for tr, va in skf.split(idx_all, y, groups=groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg["val_frac"], random_state=cfg["seed"])
        i_tr, i_va = next(gss.split(tr, groups=groups[tr]))
        tr_in, va_in = tr[i_tr], tr[i_va]
        model, _ = train_one_fold(bank, torch.tensor(tr_in), torch.tensor(va_in), device, cfg)
        oof[va] = _predict(model, bank, torch.tensor(va).to(device), device, tta=cfg["tta"])
        fold_aucs.append(roc_auc_score(y[va], oof[va]))
        train_aucs.append(roc_auc_score(y[tr_in],
                          _predict(model, bank, torch.tensor(tr_in).to(device), device, tta=False)))
    return roc_auc_score(y, oof), float(np.std(fold_aucs)), float(np.mean(train_aucs))


def run_sweep(bank, groups, device="cuda", grid=None, base_cfg=None):
    """Run each grid config through grouped CV; print a ranked table; return results."""
    grid = grid if grid is not None else DEFAULT_GRID
    # slightly shorter schedule for the sweep; re-validate the winner at full length
    base = {**DEFAULT_CFG, "epochs": 45, "patience": 8, **(base_cfg or {})}

    print(f"Regularization sweep: {len(grid)} configs, {base['n_splits']}-fold grouped CV each\n")
    results = []
    for g in grid:
        cfg = {**base, **{k: v for k, v in g.items() if k != "name"}}
        oof, std, train = _quiet_grouped_cv(bank, groups, device, cfg)
        gap = train - oof
        results.append({"name": g["name"], "oof_auc": oof, "fold_std": std,
                        "train_auc": train, "gap": gap, "config": g})
        print(f"  {g['name']:<16} OOF={oof:.4f}  +/-{std:.3f}  train={train:.4f}  gap={gap:+.4f}")

    results.sort(key=lambda r: (-r["oof_auc"], r["gap"]))
    print(f"\n{'='*60}\n  RANKED by honest OOF AUC\n{'='*60}")
    print(f"  {'config':<16} {'OOF':>7} {'std':>7} {'gap':>8}")
    for r in results:
        print(f"  {r['name']:<16} {r['oof_auc']:>7.4f} {r['fold_std']:>7.3f} {r['gap']:>+8.4f}")
    best = results[0]
    print(f"\n  Best: {best['name']}  (OOF {best['oof_auc']:.4f}, gap {best['gap']:+.4f})")
    print(f"  Config: {best['config']}")
    print(f"  -> re-run Cell 5 with cfg=dict(**this, n_splits=5) to lock it in and save the ensemble.")
    return results
