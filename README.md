# Eidon AI — Deepfake Detection

CLIP ViT-L/14 for video deepfake detection with a lightweight attention-pooling
head trained on cached frozen embeddings, evaluated with leakage-safe grouped
cross-validation. See [RESULTS.md](RESULTS.md) for the honest numbers and findings.

## Architecture (current)

- **Backbone:** CLIP ViT-L/14 (OpenAI pretrained), **fully frozen** — used as a
  feature extractor. Freezing preserves generalization on a small (~1k video) set.
- **Dual-stream input:** 16 frames per video through the same CLIP as (a) the
  cropped face and (b) the full frame → a 768-d vector per frame per stream.
- **Head:** attention pooling over frames per stream (learns which frames carry the
  artifact) + a learned "no-face" vector for detection failures + a small MLP.
- **Optional forensic stream:** per-frame FFT + SRM/noise descriptors fused with the
  CLIP streams to add low-level blend/texture sensitivity CLIP lacks.
- **Fast training:** frozen CLIP features are cached once, so the head trains at
  ~1 s/epoch (vs ~20 min/epoch recomputing CLIP every epoch).

The earlier LayerNorm-tuned + angular-margin variant (`src/lnclip_model.py`,
`train.py`) is kept for reference; the frozen-embedding pipeline above is the
current best and, under honest evaluation, matches it while training ~1000× faster.

## Evaluation

- **Grouped cross-validation** (`src/grouping.py`, `src/cv_train.py`): identity/
  source groups so a clip's real and its face-swaps never split across folds.
- **Leave-one-generator-out** (`src/logo_eval.py`): generalization to unseen
  manipulation families.
- Honest grouped OOF AUC **0.9428**; see [RESULTS.md](RESULTS.md).

## Dataset

| Source | Real | Fake | Total |
|--------|------|------|-------|
| RTFS-10K + FakeParts | 250 | 330 | 580 |
| Week 2 (Sora/Veo2/UniFace) | 50 | 120 | 170 |
| Google DFD | 62 | 244 | 306 |
| **Total** | **362** | **694** | **1,056** |

## Software Requirements

- Python 3.11
- PyTorch 2.4.1 + CUDA 12.1
- open_clip_torch 2.26+
- InsightFace 0.7+, OpenCV 4.10+, dlib 19.24+

## Hardware Requirements

- GPU: NVIDIA T4 (16GB) minimum / L4 (24GB) recommended
- RAM: 16GB minimum
- Storage: 15GB

## Usage

### Colab (recommended)

Open `notebooks/Train_Fast.ipynb` in Google Colab with a T4 GPU and run the cells
in order:

1. Install + mount Drive + locate videos
2. Build the CLIP embedding cache (once; reuses the pixel cache)
3. Leakage-safe grouped cross-validation
4. Leave-one-generator-out generalization eval
5. Regularization sweep
6. Forensic-stream comparison (CLIP vs CLIP+forensic)

### Legacy CLI (LayerNorm-tuned variant)

```bash
python train.py \
    --data_dirs dataset_production dataset_week2 dfd_dataset \
    --epochs 20 \
    --batch_size 8
```

## References

- Deepfake Detection that Generalizes Across Benchmarks (2025, arxiv 2508.06248)
- Unlocking Hidden Potential of CLIP (2025, arxiv 2503.19683)
- FMSD: Forgery-aware Layer Masking (2026, arxiv 2601.01041)

## Team

Eidon AI Research Internship, 2025
