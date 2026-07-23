"""Tests for src/attn_head.py DeepfakeHead.

Runnable directly (`python tests/test_attn_head.py`) or under pytest.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.attn_head import DeepfakeHead, NUM_FRAMES, EMBED_DIM

FORENSIC = 73


def _batch(dim, B=6):
    face = torch.randn(B, NUM_FRAMES, dim)
    full = torch.randn(B, NUM_FRAMES, dim)
    mask = torch.zeros(B, NUM_FRAMES, dtype=torch.bool)
    mask[0, :] = True  # row 0 is a fully-missing-face example (all padded)
    valid = torch.ones(B, dtype=torch.bool)
    valid[0] = False
    return face, full, mask, valid


def test_baseline_head_is_unchanged_identity_projection():
    # No forensic stream => input projection must be Identity (byte-for-byte baseline)
    h = DeepfakeHead(dim=EMBED_DIM)
    assert isinstance(h.face_in, nn.Identity)
    assert isinstance(h.full_in, nn.Identity)


def test_forensic_head_projects_wide_input():
    h = DeepfakeHead(dim=EMBED_DIM + FORENSIC)
    assert isinstance(h.face_in, nn.Linear)
    assert h.face_in.in_features == EMBED_DIM + FORENSIC
    assert h.face_in.out_features == EMBED_DIM


def test_forward_shape_and_finite_with_missing_face():
    for dim in (EMBED_DIM, EMBED_DIM + FORENSIC):
        h = DeepfakeHead(dim=dim)
        face, full, mask, valid = _batch(dim)
        out = h(face, full, mask, mask, valid)
        assert out.shape == (face.shape[0], 2)
        assert torch.isfinite(out).all(), f"non-finite logits at dim={dim}"


def test_backward_no_nan_grads_with_missing_face():
    # The all-padded (missing-face) row must not poison gradients with NaNs.
    h = DeepfakeHead(dim=EMBED_DIM)
    face, full, mask, valid = _batch(EMBED_DIM)
    out = h(face, full, mask, mask, valid)
    out.sum().backward()
    for n, p in h.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN grad in {n}"


if __name__ == "__main__":
    torch.manual_seed(0)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all attn_head tests passed")
