"""
LNCLIP-DF Model — LayerNorm-tuned CLIP for Deepfake Detection.

Based on:
  - "Deepfake Detection that Generalizes Across Benchmarks" (2025, arxiv 2508.06248)
  - "Unlocking Hidden Potential of CLIP in Generalizable Deepfake Detection" (2025, arxiv 2503.19683)

Architecture:
  - CLIP ViT-L/14 backbone
  - ONLY LayerNorm parameters in layers 18-23 are trainable (0.03% of model)
  - L2 normalization → all embeddings live on unit hypersphere
  - Angular margin classifier (ArcFace-style loss)
  - Temporal mean pooling for video (16 frames)

Key differences from VIPER v3:
  - No external MLP head
  - No BCE loss
  - Angular margin creates wide decision boundary
  - LN-tuning adapts CLIP internals for the task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ── Angular Margin Product (ArcFace-style classifier) ─────────────────────────

class ArcMarginProduct(nn.Module):
    """
    ArcFace-style angular margin classifier.
    Maps L2-normalized embeddings to class logits with angular margin.

    This creates a WIDE decision boundary on the hypersphere between
    real and fake embeddings, preventing overfitting to specific artifacts.
    """

    def __init__(self, in_features: int, num_classes: int = 2,
                 scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        # Class weight vectors on the hypersphere
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Args:
            embeddings: (B, D) L2-normalized embeddings
            labels: (B,) integer class labels (0=real, 1=fake). None during inference.

        Returns:
            logits: (B, num_classes) scaled cosine logits with angular margin
        """
        # Normalize weight vectors
        weight_norm = F.normalize(self.weight, dim=1)

        # Cosine similarity between embeddings and class vectors
        cosine = F.linear(embeddings, weight_norm)  # (B, num_classes)

        if labels is None:
            # Inference: just return scaled cosine
            return cosine * self.scale

        # Training: add angular margin to the target class
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        # Numerical stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding of labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)

        # Apply margin only to target class
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.scale

        return logits


# ── Latent Space Augmentation ─────────────────────────────────────────────────

class SphericalAugmentation(nn.Module):
    """
    Augments embeddings on the unit hypersphere by adding small random rotations.
    This forces the decision boundary to be robust to embedding perturbations.
    Only active during training.
    """

    def __init__(self, noise_std: float = 0.05):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return embeddings

        # Add Gaussian noise then re-normalize to sphere
        noise = torch.randn_like(embeddings) * self.noise_std
        augmented = embeddings + noise
        augmented = F.normalize(augmented, dim=-1)
        return augmented


# ── LNCLIP-DF Model ───────────────────────────────────────────────────────────

class LNCLIPDeepfakeDetector(nn.Module):
    """
    LayerNorm-tuned CLIP ViT-L/14 for Deepfake Detection.
    
    DUAL-INPUT architecture:
      Path A: Face crop (224×224) — catches face-level artifacts
      Path B: Full frame (224×224) — catches body/scene/context artifacts
    
    Both paths use the SAME CLIP encoder (shared weights).
    Combined embedding: [768 face + 768 frame] = 1536-dim → classifier.
    
    This handles:
      - Face-swap (detected via face crop path)
      - Expression transfer (detected via face crop path)
      - Full-body GAN (detected via full frame path — no face detection needed)
      - Background manipulation (detected via full frame path)
    
    If face detection fails, face path gets zeros and full frame carries detection.
    """

    def __init__(
        self,
        clip_model,
        num_trainable_layers: int = 6,
        arc_scale: float = 30.0,
        arc_margin: float = 0.3,
        sphere_noise: float = 0.05,
    ):
        super().__init__()

        self.visual = clip_model.visual
        self.embed_dim = 768  # ViT-L/14 output dimension
        self.combined_dim = 768 * 2  # face + frame = 1536

        # Freeze everything
        for param in self.visual.parameters():
            param.requires_grad = False

        # Unfreeze LayerNorm in last N transformer layers
        total_layers = len(self.visual.transformer.resblocks)
        start_layer = total_layers - num_trainable_layers

        trainable_count = 0
        for i, block in enumerate(self.visual.transformer.resblocks):
            if i >= start_layer:
                for name, param in block.named_parameters():
                    if 'ln_' in name or 'norm' in name.lower():
                        param.requires_grad = True
                        trainable_count += param.numel()

        # Also unfreeze the final LayerNorm (ln_post)
        if hasattr(self.visual, 'ln_post'):
            for param in self.visual.ln_post.parameters():
                param.requires_grad = True
                trainable_count += param.numel()

        print(f"[LNCLIP] Trainable LN parameters: {trainable_count:,} "
              f"({trainable_count / sum(p.numel() for p in self.visual.parameters()) * 100:.3f}%)")

        # Spherical augmentation
        self.sphere_aug = SphericalAugmentation(noise_std=sphere_noise)

        # Angular margin classifier — input is 1536 (face 768 + frame 768)
        self.classifier = ArcMarginProduct(
            in_features=self.combined_dim,
            num_classes=2,
            scale=arc_scale,
            margin=arc_margin,
        )

    def encode_frames(self, face_crops: torch.Tensor, full_frames: torch.Tensor) -> torch.Tensor:
        """
        Dual-path encoding: face crops + full frames through shared CLIP.

        Args:
            face_crops:  (B, T, C, H, W) face crops (224×224), zeros if detection failed
            full_frames: (B, T, C, H, W) full frames resized to 224×224

        Returns:
            (B, 1536) L2-normalized combined embedding on unit sphere
        """
        B, T, C, H, W = face_crops.shape

        # Path A: Face crops
        face_flat = face_crops.view(B * T, C, H, W)
        face_embs = self.visual(face_flat)  # (B*T, 768)
        face_embs = face_embs.view(B, T, -1).mean(dim=1)  # (B, 768)

        # Path B: Full frames
        frame_flat = full_frames.view(B * T, C, H, W)
        frame_embs = self.visual(frame_flat)  # (B*T, 768)
        frame_embs = frame_embs.view(B, T, -1).mean(dim=1)  # (B, 768)

        # Combine: concatenate face + frame
        combined = torch.cat([face_embs, frame_embs], dim=-1)  # (B, 1536)

        # L2 normalize to unit hypersphere
        combined = F.normalize(combined.float(), dim=-1)

        return combined

    def forward(self, face_crops: torch.Tensor, full_frames: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        """
        Args:
            face_crops:  (B, T, C, H, W) face crops
            full_frames: (B, T, C, H, W) full frames
            labels: (B,) integer labels (0=real, 1=fake). None for inference.

        Returns:
            logits: (B, 2) class logits
            embeddings: (B, 1536) L2-normalized combined embeddings
        """
        embeddings = self.encode_frames(face_crops, full_frames)

        # Apply spherical augmentation during training
        aug_embeddings = self.sphere_aug(embeddings)

        # Angular margin classification
        logits = self.classifier(aug_embeddings, labels)

        return logits, embeddings

    def predict(self, face_crops: torch.Tensor, full_frames: torch.Tensor) -> torch.Tensor:
        """Inference: returns P(fake) for each video."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(face_crops, full_frames, labels=None)
            probs = F.softmax(logits, dim=-1)
            return probs[:, 1]  # P(fake)


# ── Model loading ─────────────────────────────────────────────────────────────

def build_lnclip_model(
    device: str = "cuda",
    num_trainable_layers: int = 6,
    arc_scale: float = 30.0,
    arc_margin: float = 0.3,
    sphere_noise: float = 0.05,
):
    """Build the dual-input LNCLIP-DF model with frozen CLIP + trainable LayerNorms."""
    import open_clip

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()

    model = LNCLIPDeepfakeDetector(
        clip_model=clip_model,
        num_trainable_layers=num_trainable_layers,
        arc_scale=arc_scale,
        arc_margin=arc_margin,
        sphere_noise=sphere_noise,
    ).to(device)

    return model, preprocess
