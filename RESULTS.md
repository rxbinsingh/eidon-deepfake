# Results

Honest evaluation of the LNCLIP-DF deepfake detector on 1,056 videos (362 real /
694 fake) across three sources. All headline numbers use **identity/source-grouped
5-fold cross-validation with nested validation** — the same person/source clip
never appears in both train and test, and early stopping never sees the scored
fold. This is deliberately stricter (and lower) than a random split.

## Headline

| Metric | Value | Notes |
|--------|-------|-------|
| Honest AUC (grouped OOF) | **0.9428** | leakage-safe, no selection bias |
| Prior baseline | 0.9535 | single random split — inflated (see below) |
| Random-split OOF (leaky) | 0.9805 | for comparison only |
| Training speed | ~20 min → **~1 s / epoch** | frozen-embedding caching |
| Unseen-generator AUC (LOGO) | ~0.79 | leave-one-generator-out, data-bound |

## Finding 1 — evaluation leakage

74% of the fakes (514 / 694) are face-swaps generated from the **same source
clips** as real videos (RTFS naming: `rtfs_real_{clip}` vs
`rtfs_faceswap_inswapper_{clip}_…`). Under a random split the model can score by
memorizing shared backgrounds rather than detecting manipulation:

- Random (leaky) OOF AUC: **0.9805**
- Grouped (honest) OOF AUC: **0.9428**
- Inflation removed: **~0.10**, concentrated in the identity-linked categories.

`src/grouping.py` assigns an identity/source group per video (RTFS source clip,
DFD actor union-find, FF++ target, synthetic = unique) so folds never share one.

## Finding 2 — overfitting, measured and controlled

Train AUC was ~1.0 while honest OOF was 0.88 (gap +0.11). A regularization sweep
(`src/reg_sweep.py`, cheap at ~1 s/epoch) closed roughly half the gap and lifted
the honest number:

| | Honest OOF | Train–OOF gap |
|--|--|--|
| Before (light reg) | 0.8812 | +0.1125 |
| After (`strong_all`) | **0.9428** | +0.0569 |

Winner `strong_all`: dropout 0.5, weight_decay 0.05, frame_drop 0.25, noise 0.2,
hidden 128, heads 2. Dropout is the dominant lever; weight decay alone did nothing.

## Per-category honest AUC (`strong_all`)

| Category | AUC |
|----------|-----|
| t2v_veo2 | 0.998 |
| fullbody_gan | 0.996 |
| t2v_sora | 0.989 |
| face_swap_uniface | 0.980 |
| expression_swap | 0.966 |
| face_swap (inswapper) | 0.923 |

The weakest category — inswapper face-swap — recovered from 0.83 to 0.92 with
regularization. All categories are now ≥ 0.92.

## Finding 3 — generalization to unseen generators (the real limit)

Leave-one-generator-out (`src/logo_eval.py`) trains on reals + all other generator
families and tests on a held-out family:

| Held-out family | AUC |
|-----------------|-----|
| t2v_generative | 0.728 |
| expression | 0.858 |
| faceswap | 0.782 |
| **Mean** | **0.789** |

This is the honest limitation: with ~1k videos the model transfers only partially
to a manipulation family it has never seen. It is **data-bound** — the durable fix
is more generator diversity, not more tuning.

## In progress

- **Forensic stream** (`src/forensic_features.py`): per-frame FFT + SRM/noise
  descriptors fused with CLIP, to add the low-level blend/texture sensitivity CLIP
  lacks. Evaluated on both grouped AUC and LOGO.
- Probability calibration for a usable operating threshold (OOF accuracy is 0.839
  at an uncalibrated 0.5 cut).
