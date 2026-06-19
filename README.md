# Eidon AI — Deepfake Detection

LNCLIP-DF: LayerNorm-tuned CLIP ViT-L/14 for video deepfake detection with angular margin classification.

## Architecture

- **Backbone:** CLIP ViT-L/14 (OpenAI pretrained)
- **Tuning:** Only LayerNorm parameters in last 6 transformer layers (0.03% of model)
- **Loss:** Angular Margin (ArcFace-style) on unit hypersphere
- **Augmentation:** Compression (JPEG/blur/resize) + spherical noise
- **Temporal:** Mean pool across 16 face crops per video
- **Normalization:** L2 → all embeddings on unit sphere

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

Open `notebooks/Train_Colab.ipynb` in Google Colab with T4 GPU.

### CLI

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
