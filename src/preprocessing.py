"""
VIPER — preprocessing.py
Frame extraction, face detection, and face crop pipeline.
Uses InsightFace buffalo_sc (same as SynID) for face detection.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import insightface
from insightface.app import FaceAnalysis


# ── Face detector (singleton) ─────────────────────────────────────────────────

_face_app: Optional[FaceAnalysis] = None


def get_face_app() -> FaceAnalysis:
    """Lazy-load InsightFace buffalo_sc. Same model used in SynID."""
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(
            name="buffalo_sc",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
    return _face_app


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    num_frames: int = 16,
    start_frac: float = 0.05,
    end_frac: float = 0.95,
) -> list[np.ndarray]:
    """
    Extract `num_frames` evenly spaced frames from a video.
    Skips the first and last 5% to avoid title cards / black frames.

    Returns list of BGR frames (H, W, 3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        num_frames = max(1, total)

    start_idx = int(total * start_frac)
    end_idx   = int(total * end_frac)
    indices   = np.linspace(start_idx, end_idx, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def extract_anchor_frames(
    video_path: str,
    n_anchor: int = 8,
) -> list[np.ndarray]:
    """
    Extract the first `n_anchor` frames for identity anchor formation.
    Skips the very first frame (often a cut or black frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2)  # skip frame 0 and 1
    while len(frames) < n_anchor:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


# ── Face detection and cropping ───────────────────────────────────────────────

def detect_face(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the largest face in a frame and return a 224×224 aligned crop.
    Returns None if no face is detected.
    """
    app = get_face_app()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)

    if not faces:
        return None

    # Take the largest face by bounding box area
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    # Add 20% padding around the face
    pad_x = int((x2 - x1) * 0.2)
    pad_y = int((y2 - y1) * 0.2)
    h, w  = frame.shape[:2]
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (224, 224))
    return crop


def detect_face_with_landmarks(
    frame: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (face_crop_224, arcface_embedding) or (None, None).
    The ArcFace embedding is 512-dim, L2-normalized.
    """
    app = get_face_app()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)

    if not faces:
        return None, None

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # ArcFace embedding (512-dim, already normalized by InsightFace)
    embedding = face.normed_embedding  # shape (512,)

    # Face crop
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    pad_x = int((x2 - x1) * 0.2)
    pad_y = int((y2 - y1) * 0.2)
    h, w  = frame.shape[:2]
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    crop = cv2.resize(crop, (224, 224))
    return crop, embedding


# ── Full preprocessing pipeline ───────────────────────────────────────────────

def preprocess_video(
    video_path: str,
    num_frames: int = 16,
    n_anchor: int = 8,
) -> dict:
    """
    Full preprocessing pipeline for a single video.
    
    DUAL-INPUT: extracts both face crops AND full frames.
    If face detection fails, face_crops will be empty but full_frames still valid.

    Returns dict with:
        face_crops      : list of face crops (224×224 BGR) — empty if no face detected
        full_frames     : list of full frames resized to (224×224 BGR) — always available
        anchor_embeddings: list of ArcFace embeddings from anchor frames
        video_embeddings: list of ArcFace embeddings from video frames
        valid           : bool — True if at least full_frames were extracted
        face_valid      : bool — True if face detection succeeded
    """
    result = {
        "face_crops": [],
        "full_frames": [],
        "anchor_frames": [],
        "anchor_embeddings": [],
        "video_frames": [],
        "video_embeddings": [],
        "raw_frames": [],
        "valid": False,
        "face_valid": False,
    }

    # Extract frames
    video_raw = extract_frames(video_path, num_frames=num_frames)
    if len(video_raw) < 4:
        return result

    # Full frames (always available — no face detection needed)
    for frame in video_raw:
        resized = cv2.resize(frame, (224, 224))
        result["full_frames"].append(resized)

    result["valid"] = True  # We have full frames at minimum

    # Face crops (may fail for full-body/scene videos)
    anchor_raw = extract_anchor_frames(video_path, n_anchor=n_anchor)
    for frame in anchor_raw:
        crop, emb = detect_face_with_landmarks(frame)
        if crop is not None:
            result["anchor_frames"].append(crop)
            result["anchor_embeddings"].append(emb)

    for frame in video_raw:
        crop, emb = detect_face_with_landmarks(frame)
        if crop is not None:
            result["face_crops"].append(crop)
            result["video_frames"].append(crop)
            result["video_embeddings"].append(emb)
            result["raw_frames"].append(frame)

    # Face detection succeeded if we got at least 4 face crops
    if len(result["face_crops"]) >= 4:
        result["face_valid"] = True

    return result
