"""
Hand-crafted forensic features — the low-level cues frozen CLIP is blind to.

CLIP is trained for semantics, so it under-weights exactly the signals that give
away high-quality deepfakes: frequency fingerprints from upsampling/GANs and
noise/texture inconsistencies at blend boundaries. This module extracts a small,
fixed-length forensic descriptor per frame that we cache once (like the CLIP
embeddings) and concatenate onto each stream — so the attention head sees both
semantic and forensic evidence, with no cost to training speed.

Descriptor (per 224x224 RGB frame): 73-d
  - 64  radial FFT log-power spectrum   (upsampling / GAN frequency fingerprint)
  -  6  SRM high-pass residual stats     (blend-boundary / texture inconsistency)
  -  3  per-channel noise-residual std   (color-plane noise mismatch)
"""

import cv2
import numpy as np

FORENSIC_DIM = 73
_FFT_BINS = 64

# Three standard SRM high-pass filters (spam-like), normalized.
_SRM_KERNELS = [
    np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], np.float32) / 4.0,
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float32) / 12.0,
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], np.float32) / 2.0,
]


def _radial_fft_profile(gray, nbins=_FFT_BINS):
    """Azimuthally-averaged log-power spectrum — a rotation-robust freq fingerprint."""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    y, x = np.indices((h, w))
    r = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
    bins = (r / r.max() * (nbins - 1)).astype(np.int32).ravel()
    summed = np.bincount(bins, mag.ravel(), minlength=nbins)
    counts = np.bincount(bins, minlength=nbins)
    return summed / np.maximum(counts, 1)


def frame_forensics(rgb_uint8):
    """rgb_uint8: (H,W,3) uint8 RGB frame -> (73,) float32 forensic descriptor."""
    if rgb_uint8.shape[:2] != (224, 224):
        rgb_uint8 = cv2.resize(rgb_uint8, (224, 224))
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

    fft = _radial_fft_profile(gray)  # 64

    srm = []
    for k in _SRM_KERNELS:
        res = cv2.filter2D(gray.astype(np.float32), -1, k)
        srm += [np.mean(np.abs(res)), np.std(res)]  # 6

    rgb = rgb_uint8.astype(np.float32)
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    noise_std = (rgb - blur).std(axis=(0, 1))  # 3

    return np.concatenate([fft, srm, noise_std]).astype(np.float32)
