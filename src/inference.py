"""
Production inference: video file -> calibrated P(fake) + label.

Wraps the whole pipeline into one call:
  preprocess (face + full frames) -> frozen CLIP embeddings (+ optional forensic
  stream) -> 5-fold head ensemble with flip-TTA -> calibrated probability.

Usage:
    from src.inference import DeepfakeDetector
    det = DeepfakeDetector("checkpoints_fast_forensic", use_forensic=True,
                           forensic_stats_path=".../forensic_stats.npz",
                           calibrator_path=".../calibration.json")
    out = det.predict("some_video.mp4")   # {'p_fake':0.87,'label':'fake',...}

Heavy deps (torch, open_clip, cv2, insightface) are imported lazily so the module
can be imported anywhere; they're only needed when a detector is built/run.
"""

from pathlib import Path

import numpy as np

EMBED_DIM = 768
NUM_FRAMES = 16


class DeepfakeDetector:
    def __init__(self, checkpoint_dir, device="cuda", use_forensic=False,
                 forensic_stats_path=None, calibrator_path=None,
                 hidden=128, heads=2):
        import torch
        import open_clip
        from src.attn_head import DeepfakeHead
        from src.forensic_features import FORENSIC_DIM

        self.torch = torch
        self.device = device
        self.use_forensic = use_forensic
        self.feat_dim = EMBED_DIM + (FORENSIC_DIM if use_forensic else 0)

        # Frozen CLIP visual encoder
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.visual = clip_model.visual.to(device).eval()
        for p in self.visual.parameters():
            p.requires_grad = False

        # Fold-ensemble heads
        ckpts = sorted(Path(checkpoint_dir).glob("head_fold*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No head_fold*.pt in {checkpoint_dir}")
        self.heads = []
        for cp in ckpts:
            head = DeepfakeHead(dim=self.feat_dim, hidden=hidden, heads=heads).to(device).eval()
            head.load_state_dict(torch.load(cp, map_location=device))
            self.heads.append(head)

        # Forensic standardization stats (must match training)
        self.forensic_mu = self.forensic_sd = None
        if use_forensic:
            if forensic_stats_path is None:
                raise ValueError("use_forensic=True requires forensic_stats_path")
            s = np.load(forensic_stats_path)
            self.forensic_mu, self.forensic_sd = s["mu"], s["sd"]

        # Optional calibrator
        self.calibrator = None
        if calibrator_path is not None:
            from src.calibration import Calibrator
            self.calibrator = Calibrator.load(calibrator_path)

    # ── feature extraction ────────────────────────────────────────────────────
    def _clip_tf(self):
        from torchvision import transforms as T
        return T.Compose([T.ToTensor(),
                          T.Normalize([0.48145466, 0.4578275, 0.40821073],
                                      [0.26862954, 0.26130258, 0.27577711])])

    def _pad_stack(self, frames_rgb, tf):
        from PIL import Image
        n = int(min(len(frames_rgb), NUM_FRAMES))
        ts = [tf(Image.fromarray(frames_rgb[i])) for i in range(n)]
        while len(ts) < NUM_FRAMES:
            ts.append(ts[-1].clone())
        return self.torch.stack(ts[:NUM_FRAMES]), n

    def _encode_clip(self, batch):
        with self.torch.no_grad(), self.torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            return self.visual(batch.to(self.device)).float()  # (16,768)

    def _forensic_block(self, frames_rgb):
        from src.forensic_features import frame_forensics, FORENSIC_DIM
        n = int(min(len(frames_rgb), NUM_FRAMES))
        feats = [frame_forensics(frames_rgb[i]) for i in range(n)]
        while len(feats) < NUM_FRAMES:
            feats.append(feats[-1].copy() if feats else np.zeros(FORENSIC_DIM, np.float32))
        arr = np.stack(feats[:NUM_FRAMES])
        arr = (arr - self.forensic_mu) / self.forensic_sd
        return self.torch.tensor(arr, dtype=self.torch.float32, device=self.device)

    def _stream(self, frames_rgb, forensic_block, flip):
        """One stream (face or full) -> (16, feat_dim) feature tensor."""
        batch, n = self._pad_stack(frames_rgb, self._clip_tf())
        if flip:
            batch = self.torch.flip(batch, [-1])
        emb = self._encode_clip(batch)  # (16,768)
        if self.use_forensic:
            emb = self.torch.cat([emb, forensic_block], dim=-1)  # forensic ~flip-invariant
        return emb, n

    def _embed_video(self, video_path):
        import cv2
        from src.preprocessing import preprocess_video

        pre = preprocess_video(video_path, num_frames=NUM_FRAMES)
        if not pre["valid"]:
            return None

        full_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in pre["full_frames"]]
        face_valid = bool(pre["face_valid"]) and len(pre["face_crops"]) >= 4
        face_rgb = ([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in pre["face_crops"]]
                    if face_valid else full_rgb)  # placeholder frames; masked out below

        full_for = self._forensic_block(full_rgb) if self.use_forensic else None
        face_for = self._forensic_block(face_rgb) if self.use_forensic else None
        return full_rgb, face_rgb, face_valid, full_for, face_for

    def _mask(self, count):
        idx = self.torch.arange(NUM_FRAMES, device=self.device)
        return (idx >= max(count, 1)).unsqueeze(0)  # (1,16) True where padded

    # ── prediction ────────────────────────────────────────────────────────────
    def predict(self, video_path, tta=True):
        import torch.nn.functional as F

        emb = self._embed_video(video_path)
        if emb is None:
            return {"p_fake": None, "label": "unknown", "reason": "no valid frames"}
        full_rgb, face_rgb, face_valid, full_for, face_for = emb

        flips = [False, True] if tta else [False]
        raw = []
        for flip in flips:
            full_emb, n_full = self._stream(full_rgb, full_for, flip)
            face_emb, n_face = self._stream(face_rgb, face_for, flip)
            if not face_valid:
                n_face = 0
            full_t = full_emb.unsqueeze(0)  # (1,16,D)
            face_t = face_emb.unsqueeze(0)
            fm, gm = self._mask(n_face), self._mask(n_full)
            fv = self.torch.tensor([face_valid], device=self.device, dtype=self.torch.bool)
            for head in self.heads:
                with self.torch.no_grad(), self.torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                    logits = head(face_t, full_t, fm, gm, fv)
                raw.append(float(F.softmax(logits.float(), dim=-1)[0, 1]))

        p_raw = float(np.mean(raw))
        p_fake = float(self.calibrator.transform([p_raw])[0]) if self.calibrator else p_raw
        thr = self.calibrator.threshold if self.calibrator else 0.5
        return {
            "p_fake": round(p_fake, 4),
            "label": "fake" if p_fake >= thr else "real",
            "p_raw": round(p_raw, 4),
            "threshold": round(thr, 4),
            "face_detected": face_valid,
            "n_models": len(self.heads) * len(flips),
        }
