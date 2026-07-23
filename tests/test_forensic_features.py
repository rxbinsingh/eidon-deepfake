"""Tests for src/forensic_features.py.

Runnable directly (`python tests/test_forensic_features.py`) or under pytest.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.forensic_features import frame_forensics, FORENSIC_DIM


def test_shape_and_finite():
    img = (np.random.default_rng(0).random((224, 224, 3)) * 255).astype(np.uint8)
    v = frame_forensics(img)
    assert v.shape == (FORENSIC_DIM,)
    assert v.dtype == np.float32
    assert np.isfinite(v).all()


def test_deterministic():
    img = (np.random.default_rng(1).random((224, 224, 3)) * 255).astype(np.uint8)
    assert np.allclose(frame_forensics(img), frame_forensics(img))


def test_resizes_non_224_input():
    img = (np.random.default_rng(2).random((100, 160, 3)) * 255).astype(np.uint8)
    v = frame_forensics(img)  # should resize internally, not raise
    assert v.shape == (FORENSIC_DIM,) and np.isfinite(v).all()


def test_smooth_image_has_lower_high_freq_residual_than_noise():
    # A flat/smooth image should have far less high-pass (SRM/noise) energy than
    # a high-frequency noise image — sanity that the residual features respond.
    rng = np.random.default_rng(3)
    flat = np.full((224, 224, 3), 128, np.uint8)
    noise = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
    # last 3 dims are per-channel noise-residual std
    flat_noise = frame_forensics(flat)[-3:].mean()
    rand_noise = frame_forensics(noise)[-3:].mean()
    assert rand_noise > flat_noise


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all forensic-feature tests passed")
