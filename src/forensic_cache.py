"""
Build the per-frame forensic feature cache from the existing pixel cache.

Mirrors embed_cache.py but computes hand-crafted forensic descriptors instead of
CLIP features (CPU-only, no GPU needed). Run once; EmbeddingBank concatenates
these onto the CLIP streams when `forensic_dir` is given.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.forensic_features import frame_forensics, FORENSIC_DIM

NUM_FRAMES = 16


def _pad(frames_uint8, n=NUM_FRAMES):
    """(m,224,224,3) -> (n, FORENSIC_DIM) float32 + true count (pad by repeat)."""
    m = int(min(len(frames_uint8), n))
    feats = [frame_forensics(frames_uint8[i]) for i in range(m)]
    while len(feats) < n:
        feats.append(feats[-1].copy() if feats else np.zeros(FORENSIC_DIM, np.float32))
    return np.stack(feats[:n]), m


def _process_one(vp, label, pixel_cache_dir, forensic_dir, overwrite, retries=3, retry_wait=2.0):
    """Process a single video. Retries transient Drive disconnects; never raises.

    Returns one of "skipped", "missing_pixel", "ok", "failed".
    """
    import time

    stem = Path(vp).stem
    out = forensic_dir / f"{stem}.npz"

    for attempt in range(retries):
        try:
            if out.exists() and not overwrite:
                return "skipped"
            pix = pixel_cache_dir / f"{stem}.npz"
            if not pix.exists():
                return "missing_pixel"

            d = np.load(pix)
            full = d["full_frames"]
            face_valid = bool(d["face_valid"]) if "face_valid" in d else False
            face = d["face_crops"] if ("face_crops" in d and face_valid) else None

            full_f, n_full = _pad(full)
            if face is not None and len(face) >= 4:
                face_f, n_face = _pad(face)
            else:
                face_f, n_face = np.zeros((NUM_FRAMES, FORENSIC_DIM), np.float32), 0

            np.savez_compressed(out, face=face_f.astype(np.float16),
                                full=full_f.astype(np.float16),
                                n_face=np.int64(n_face), n_full=np.int64(n_full),
                                label=np.int64(label))
            return "ok"
        except OSError:
            # Google Drive FUSE mount hiccup ("Transport endpoint not connected").
            # Transient — back off and retry rather than aborting the whole cache build.
            if attempt < retries - 1:
                time.sleep(retry_wait)
                continue
            return "failed"
        except Exception:
            return "failed"
    return "failed"


def build_forensic_cache(samples, pixel_cache_dir, forensic_dir, overwrite=False):
    pixel_cache_dir = Path(pixel_cache_dir)
    forensic_dir = Path(forensic_dir)
    forensic_dir.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "skipped": 0, "missing_pixel": 0, "failed": 0}

    for vp, label, _cat in tqdm(list(samples), desc="Forensic"):
        result = _process_one(vp, label, pixel_cache_dir, forensic_dir, overwrite)
        stats[result] += 1

    print(f"[forensic_cache] {stats}")
    return stats
