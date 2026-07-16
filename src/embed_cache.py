"""
Embedding cache builder for LNCLIP-DF (frozen-backbone fast path).

The CLIP ViT-L/14 backbone is fully frozen, so its per-frame features never
change between epochs. Instead of pushing 32 frames/video through ViT-L/14 every
epoch (the ~20 min/epoch bottleneck), we run the backbone ONCE here and cache the
per-frame 768-d embeddings to disk. A lightweight head then trains on these in
~1 s/epoch, enabling cross-validation and hyperparameter sweeps.

Input : the existing pixel cache produced by the preprocessing step
        ({stem}.npz with `full_frames`, optional `face_crops`, `face_valid`, `label`).
Output: one {stem}.npz per video with per-frame embeddings for both streams,
        clean + horizontally-flipped (flip enables consistent flip-TTA later).

Run once (a few minutes on a T4). Re-runs skip videos already cached.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

NUM_FRAMES = 16
EMBED_DIM = 768  # ViT-L/14 image feature dim

# CLIP preprocessing (frames are already 224x224 RGB uint8 in the pixel cache).
CLIP_TF = T.Compose([
    T.ToTensor(),
    T.Normalize([0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]),
])


def _pad_stack(frames_uint8, num_frames=NUM_FRAMES):
    """(n,224,224,3) RGB uint8 -> (num_frames,3,224,224) tensor + true count.

    Pads by repeating the last frame; truncates if too many.
    """
    n = int(min(len(frames_uint8), num_frames))
    tensors = [CLIP_TF(Image.fromarray(frames_uint8[i])) for i in range(n)]
    while len(tensors) < num_frames:
        tensors.append(tensors[-1].clone())
    return torch.stack(tensors[:num_frames]), n


@torch.no_grad()
def _encode(visual, batch, device, chunk=64):
    """Encode (N,3,224,224) through frozen CLIP visual -> (N,768) float16 (cpu)."""
    out = []
    for i in range(0, batch.shape[0], chunk):
        x = batch[i:i + chunk].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            feats = visual(x)  # (chunk, 768)
        out.append(feats.float().cpu())
    return torch.cat(out, dim=0).half()


def build_embedding_cache(
    samples,
    pixel_cache_dir,
    embed_cache_dir,
    device="cuda",
    include_flip=True,
    overwrite=False,
):
    """Build the per-frame embedding cache from the pixel cache.

    Args:
        samples: iterable of (video_path, label, category) as produced by the
                 notebook's get_all_videos(). Only the stem + label are used.
        pixel_cache_dir: dir with the existing {stem}.npz pixel cache.
        embed_cache_dir: output dir for {stem}.npz embedding cache.
        include_flip: also cache horizontally-flipped embeddings (for flip-TTA).
        overwrite: recompute even if the output already exists.

    Returns: dict with counts {ok, skipped, missing_pixel, failed}.
    """
    import open_clip

    pixel_cache_dir = Path(pixel_cache_dir)
    embed_cache_dir = Path(embed_cache_dir)
    embed_cache_dir.mkdir(parents=True, exist_ok=True)

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    visual = clip_model.visual.to(device).eval()
    for p in visual.parameters():
        p.requires_grad = False

    stats = {"ok": 0, "skipped": 0, "missing_pixel": 0, "failed": 0}

    for vp, label, _cat in tqdm(list(samples), desc="Embedding"):
        stem = Path(vp).stem
        out_path = embed_cache_dir / f"{stem}.npz"
        if out_path.exists() and not overwrite:
            stats["skipped"] += 1
            continue

        pixel_path = pixel_cache_dir / f"{stem}.npz"
        if not pixel_path.exists():
            stats["missing_pixel"] += 1
            continue

        try:
            data = np.load(pixel_path)
            full_frames = data["full_frames"]
            face_valid = bool(data["face_valid"]) if "face_valid" in data else False
            face_crops = data["face_crops"] if ("face_crops" in data and face_valid) else None
        except Exception:
            stats["failed"] += 1
            continue

        try:
            full_batch, n_full = _pad_stack(full_frames)
            if face_crops is not None and len(face_crops) >= 4:
                face_batch, n_face = _pad_stack(face_crops)
            else:
                face_batch, n_face, face_valid = torch.zeros_like(full_batch), 0, False

            record = {
                "full": _encode(visual, full_batch, device).numpy(),
                "face": _encode(visual, face_batch, device).numpy(),
                "n_full": np.int64(n_full),
                "n_face": np.int64(n_face),
                "label": np.int64(label),
                "face_valid": np.bool_(face_valid),
            }
            if include_flip:
                record["full_flip"] = _encode(visual, torch.flip(full_batch, [-1]), device).numpy()
                record["face_flip"] = _encode(visual, torch.flip(face_batch, [-1]), device).numpy()

            np.savez_compressed(out_path, **record)
            stats["ok"] += 1
        except Exception:
            stats["failed"] += 1

    print(f"[embed_cache] {stats}")
    return stats
