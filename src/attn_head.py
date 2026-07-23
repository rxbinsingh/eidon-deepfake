"""
Attention-pooling classifier head for cached CLIP embeddings.

Trains on the per-frame embeddings produced by `embed_cache.py`. Two design
choices that matter for AUC over the old mean-pool + linear head:

  1. Attention pooling over the 16 frames (per stream) instead of mean pooling —
     the model learns which frames carry the deepfake signal instead of averaging
     it away.
  2. Explicit missing-face handling — when face detection failed, a learned
     `no_face` vector replaces the face stream instead of feeding zeros.

Everything lives in memory (the whole embedding set is ~100 MB fp16), so training
runs directly on GPU tensors with no DataLoader — ~1 s/epoch.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 768
NUM_FRAMES = 16


# ── Model ─────────────────────────────────────────────────────────────────────

class AttnPool(nn.Module):
    """Learnable-query attention pooling over the frame axis."""

    def __init__(self, dim=EMBED_DIM, heads=4, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x, key_padding_mask):
        # x: (B, T, D); key_padding_mask: (B, T) True where padded (ignored)
        q = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return out.squeeze(1)  # (B, D)


class DeepfakeHead(nn.Module):
    """Dual-stream (face + full-frame) attention head over cached CLIP embeddings."""

    def __init__(self, dim=EMBED_DIM, heads=4, hidden=256, dropout=0.3, attn_dim=EMBED_DIM):
        super().__init__()
        # When a forensic stream is concatenated, dim != attn_dim: project each stream
        # down to attn_dim (also keeps attention embed_dim divisible by heads). With no
        # forensic stream (dim == attn_dim) this is Identity, so the head is unchanged.
        self.face_in = nn.Linear(dim, attn_dim) if dim != attn_dim else nn.Identity()
        self.full_in = nn.Linear(dim, attn_dim) if dim != attn_dim else nn.Identity()
        self.face_norm = nn.LayerNorm(attn_dim)
        self.full_norm = nn.LayerNorm(attn_dim)
        self.face_pool = AttnPool(attn_dim, heads, dropout=0.1)
        self.full_pool = AttnPool(attn_dim, heads, dropout=0.1)
        self.no_face = nn.Parameter(torch.randn(attn_dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.LayerNorm(2 * attn_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * attn_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, face, full, face_mask, full_mask, face_valid):
        # *_mask: (B, T) True where padded. face_valid: (B,) bool.
        f = self.face_pool(self.face_norm(self.face_in(face)), face_mask)
        g = self.full_pool(self.full_norm(self.full_in(full)), full_mask)
        f = torch.where(face_valid.unsqueeze(1), f, self.no_face.unsqueeze(0).to(f.dtype))
        return self.mlp(torch.cat([f, g], dim=-1))


# ── In-memory embedding bank ──────────────────────────────────────────────────

class EmbeddingBank:
    """Loads the whole embedding cache into GPU tensors for index-based batching."""

    def __init__(self, stems, embed_cache_dir, device="cuda", forensic_dir=None):
        embed_cache_dir = Path(embed_cache_dir)
        forensic_dir = Path(forensic_dir) if forensic_dir else None
        face, full, face_f, full_f = [], [], [], []
        n_face, n_full, valid, label = [], [], [], []
        face_forensic, full_forensic = [], []
        self.stems, self.categories = [], []

        for stem, cat in stems:
            p = embed_cache_dir / f"{stem}.npz"
            if not p.exists():
                continue
            if forensic_dir is not None and not (forensic_dir / f"{stem}.npz").exists():
                continue  # need both caches aligned
            d = np.load(p)
            face.append(d["face"]); full.append(d["full"])
            face_f.append(d["face_flip"] if "face_flip" in d else d["face"])
            full_f.append(d["full_flip"] if "full_flip" in d else d["full"])
            n_face.append(int(d["n_face"])); n_full.append(int(d["n_full"]))
            valid.append(bool(d["face_valid"])); label.append(int(d["label"]))
            if forensic_dir is not None:
                fd = np.load(forensic_dir / f"{stem}.npz")
                face_forensic.append(fd["face"].astype(np.float32))
                full_forensic.append(fd["full"].astype(np.float32))
            self.stems.append(stem); self.categories.append(cat)

        to = lambda a, t=torch.float16: torch.tensor(np.stack(a)).to(device, t)
        self.face = to(face); self.full = to(full)
        self.face_flip = to(face_f); self.full_flip = to(full_f)

        if forensic_dir is not None:
            # Standardize forensic features (per-dim z-score over all frames), then
            # concat onto both CLIP streams so the head sees semantic + forensic
            # evidence per frame. Flip variant reuses forensic (freq/noise ~flip-invariant).
            ff = np.stack(face_forensic); gf = np.stack(full_forensic)
            allf = np.concatenate([ff.reshape(-1, ff.shape[-1]), gf.reshape(-1, gf.shape[-1])], 0)
            mu, sd = allf.mean(0), allf.std(0) + 1e-6
            ff = ((ff - mu) / sd).astype(np.float32); gf = ((gf - mu) / sd).astype(np.float32)
            fft = torch.tensor(ff).to(device, torch.float16)
            gft = torch.tensor(gf).to(device, torch.float16)
            self.face = torch.cat([self.face, fft], dim=-1)
            self.full = torch.cat([self.full, gft], dim=-1)
            self.face_flip = torch.cat([self.face_flip, fft], dim=-1)
            self.full_flip = torch.cat([self.full_flip, gft], dim=-1)

        self.feat_dim = self.face.shape[-1]  # 768, or 768+forensic when enabled
        self.n_face = torch.tensor(n_face, device=device)
        self.n_full = torch.tensor(n_full, device=device)
        self.face_valid = torch.tensor(valid, device=device, dtype=torch.bool)
        self.label = torch.tensor(label, device=device, dtype=torch.long)
        self.device = device
        print(f"[EmbeddingBank] {len(self.stems)} videos "
              f"({int((self.label == 0).sum())} real / {int((self.label == 1).sum())} fake)"
              f" | feat_dim={self.feat_dim}{' (CLIP+forensic)' if forensic_dir is not None else ''}")

    def __len__(self):
        return len(self.stems)

    @staticmethod
    def _mask(counts):
        # (B,) counts -> (B, T) True where padded. Guarantee >=1 valid key (avoids
        # empty-attention NaN for missing-face rows; those get overwritten by no_face).
        counts = counts.clamp(min=1)
        idx = torch.arange(NUM_FRAMES, device=counts.device).unsqueeze(0)
        return idx >= counts.unsqueeze(1)

    def batch(self, idx, train=False, noise_frac=0.0, frame_drop=0.0, flip=None):
        """Gather a batch by index. Applies embedding-space augmentation if train.

        flip: None -> random flip when train (augmentation); True/False forces the
        clean or flipped variant (used for flip-TTA at eval).
        """
        use_flip = flip if flip is not None else (train and (torch.rand(1).item() < 0.5))
        face = (self.face_flip if use_flip else self.face)[idx].float()
        full = (self.full_flip if use_flip else self.full)[idx].float()
        face_mask = self._mask(self.n_face[idx])
        full_mask = self._mask(self.n_full[idx])

        if train and noise_frac > 0:
            face = face + torch.randn_like(face) * face.std() * noise_frac
            full = full + torch.randn_like(full) * full.std() * noise_frac
        if train and frame_drop > 0:
            face_mask = _drop_frames(face_mask, self.n_face[idx], frame_drop)
            full_mask = _drop_frames(full_mask, self.n_full[idx], frame_drop)

        return {
            "face": face, "full": full,
            "face_mask": face_mask, "full_mask": full_mask,
            "face_valid": self.face_valid[idx], "label": self.label[idx],
        }


def _drop_frames(mask, counts, p):
    """Randomly mark valid frames as padded (keep >=1 valid per row)."""
    drop = (torch.rand_like(mask.float()) < p) & (~mask)
    keep_one = (~mask & ~drop).any(dim=1) | (counts <= 1)
    new_mask = mask | drop
    # if a row lost all valid frames, un-drop its first originally-valid frame
    bad = ~(~new_mask).any(dim=1)
    if bad.any():
        first_valid = (~mask).float().argmax(dim=1)
        new_mask[bad, first_valid[bad]] = False
    return new_mask
