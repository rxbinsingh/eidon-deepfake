"""
Leave-one-generator-out (LOGO) evaluation.

In-distribution CV (even leakage-safe) measures accuracy on manipulation types the
model trained on. Production reality is different: new deepfake generators appear
constantly, and the real question is "does it catch a method it has never seen?"
LOGO answers that — for each generator family it trains on reals + all OTHER
families and tests on the held-out family (plus a disjoint held-out real set).

No group is shared between train and test (identity/source grouping from
`grouping.py`), so the only thing new at test time is the generator.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.cv_train import DEFAULT_CFG, train_one_fold, _predict

# Recommended grouping of raw categories into generator "families". Holding out a
# single category (e.g. inswapper) while keeping uniface in train isn't truly
# "unseen" — both are face-swaps. Families make the held-out method genuinely novel.
DEFAULT_FAMILIES = {
    "face_swap": "faceswap",
    "face_swap_uniface": "faceswap",
    "expression_swap": "expression",
    "t2v_sora_1080p": "t2v_generative",
    "t2v_veo2_720p": "t2v_generative",
    "fullbody_gan": "t2v_generative",
}


def run_logo(bank, groups, device="cuda", cfg=None, families=None,
             real_test_frac=0.3, save_dir=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    families = families if families is not None else DEFAULT_FAMILIES

    y = bank.label.cpu().numpy()
    cats = np.array(bank.categories)
    groups = np.asarray(groups)
    idx = np.arange(len(y))

    fam_of = np.array([families.get(c, c) for c in cats])
    fake_families = sorted(set(fam_of[y == 1]))

    # Fixed grouped split of reals into train/test (shared across all held-out folds)
    real_idx = idx[y == 0]
    gss = GroupShuffleSplit(n_splits=1, test_size=real_test_frac, random_state=cfg["seed"])
    rtr, rte = next(gss.split(real_idx, groups=groups[real_idx]))
    train_real, test_real = real_idx[rtr], real_idx[rte]

    print(f"Leave-one-generator-out ({len(fake_families)} families). "
          f"Reals: {len(train_real)} train / {len(test_real)} test.\n")

    per_gen, details = {}, {}
    for G in fake_families:
        test_fake = idx[(y == 1) & (fam_of == G)]
        test_idx = np.concatenate([test_real, test_fake])
        test_groups = set(groups[test_idx].tolist())

        # Train on train-reals + all OTHER families' fakes, minus any group that
        # appears in the test set (guarantees no identity/source crosses over).
        pool = np.concatenate([train_real, idx[(y == 1) & (fam_of != G)]])
        pool = pool[np.array([g not in test_groups for g in groups[pool]], dtype=bool)]

        # Small grouped val slice for early stopping (disjoint from test)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=cfg["seed"])
        tr2, va2 = next(gss2.split(pool, groups=groups[pool]))
        val_ok = len(set(y[pool[va2]])) == 2
        tr_idx = pool[tr2] if val_ok else pool
        va_idx = pool[va2] if val_ok else pool[tr2][:max(2, len(pool) // 10)]

        model, _ = train_one_fold(bank, torch.tensor(tr_idx), torch.tensor(va_idx), device, cfg)
        probs = _predict(model, bank, torch.tensor(test_idx).to(device), device, tta=cfg["tta"])
        auc = float(roc_auc_score(y[test_idx], probs))
        per_gen[G] = round(auc, 4)
        details[G] = {"train_n": int(len(pool)), "test_real": int(len(test_real)),
                      "test_fake": int(len(test_fake))}
        print(f"  held-out [{G:<16}] AUC = {auc:.4f}   "
              f"(train {len(pool)} | test {len(test_real)} real + {len(test_fake)} fake)")

    mean = float(np.mean(list(per_gen.values())))
    print(f"\n{'='*52}")
    print(f"  Mean unseen-generator AUC: {mean:.4f}")
    print(f"  (in-distribution grouped OOF was ~0.97 — expect this to be lower)")
    print(f"{'='*52}")

    results = {"per_generator_auc": per_gen, "mean_auc": round(mean, 4),
               "details": details, "families": families, "config": cfg}
    if save_dir:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "logo_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved LOGO results to {save_dir}")
    return results
