"""
LNCLIP-DF Training Script — LayerNorm-tuned CLIP for Deepfake Detection.

Based on SOTA papers (2025):
  - arxiv 2508.06248 (LNCLIP-DF, tested on 14 benchmarks)
  - arxiv 2503.19683 (LN-tuning + hyperspherical manifold)

Usage:
    python train_lnclip.py \
        --data_dirs dataset_production dataset_week2 dfd_dataset \
        --save_dir checkpoints_lnclip \
        --epochs 20

Runs on free Colab T4 in ~30 minutes with all 1,056 videos.
"""

import os
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms as T
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from PIL import Image

from src.lnclip_model import build_lnclip_model


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Compression Augmentation ──────────────────────────────────────────────────

class CompressionAugmentation:
    """
    Applies random compression artifacts during training.
    Makes the model robust to real-world video quality variations.
    Based on Wang et al. CVPR 2020 finding that augmentation is critical.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """img: PIL Image"""
        if random.random() > self.p:
            return img

        img_np = np.array(img)

        # Random JPEG compression (quality 30-95)
        if random.random() < 0.5:
            quality = random.randint(30, 95)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), encode_param)
            img_np = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Random Gaussian blur (sigma 0.5-2.0)
        if random.random() < 0.3:
            sigma = random.uniform(0.5, 2.0)
            ksize = int(sigma * 4) | 1  # ensure odd
            img_np = cv2.GaussianBlur(img_np, (ksize, ksize), sigma)

        # Random downscale then upscale (0.5x-0.9x)
        if random.random() < 0.3:
            scale = random.uniform(0.5, 0.9)
            h, w = img_np.shape[:2]
            small = cv2.resize(img_np, (int(w*scale), int(h*scale)))
            img_np = cv2.resize(small, (w, h))

        return Image.fromarray(img_np)


# ── Dataset ───────────────────────────────────────────────────────────────────

CLIP_NORMALIZE = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)


def load_all_samples(data_dirs: list, seed: int = 42):
    """Load samples from multiple dataset directories."""
    all_samples = []

    for data_dir in data_dirs:
        meta_path = Path(data_dir) / "metadata.csv"
        if not meta_path.exists():
            print(f"  [Skip] No metadata.csv in {data_dir}")
            continue

        meta = pd.read_csv(meta_path)
        count = 0
        for _, row in meta.iterrows():
            label_str = row.get("label", row.get("category", "unknown"))
            label = 0 if label_str == "real" else 1
            category = row.get("category", label_str)

            # Find the video file
            filename = row["filename"]
            video_path = Path(data_dir) / category / filename
            if not video_path.exists():
                video_path = Path(data_dir) / label_str / filename
            if not video_path.exists():
                continue

            all_samples.append((str(video_path), label, category, data_dir))
            count += 1

        print(f"  [{Path(data_dir).name}] Loaded {count} videos")

    random.seed(seed)
    random.shuffle(all_samples)
    return all_samples


class LNCLIPDataset(Dataset):
    """
    Dataset for LNCLIP-DF training.
    Extracts face crops on-the-fly with caching.
    Applies compression augmentation during training.
    """

    def __init__(self, samples, cache_dir, num_frames=16, train=True):
        self.samples = samples
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_frames = num_frames
        self.train = train

        # Augmentation
        self.compress_aug = CompressionAugmentation(p=0.5) if train else None
        self.spatial_aug = T.Compose([
            T.RandomHorizontalFlip(0.5),
        ]) if train else T.Compose([])

        self.to_tensor = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            CLIP_NORMALIZE,
        ])

    def _cache_path(self, video_path):
        return self.cache_dir / f"{Path(video_path).stem}.npz"

    def _extract_faces(self, video_path):
        """Extract face crops using InsightFace. Cache to disk."""
        cp = self._cache_path(video_path)
        if cp.exists():
            try:
                data = np.load(cp)
                return data["crops"]
            except:
                pass

        from src.preprocessing import preprocess_video
        preprocessed = preprocess_video(video_path, num_frames=self.num_frames, n_anchor=8)
        if not preprocessed["valid"]:
            return None

        crops_rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in preprocessed["video_frames"]]
        crops_arr = np.stack(crops_rgb)
        np.savez_compressed(cp, crops=crops_arr)
        return crops_arr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label, category, source = self.samples[idx]
        crops = self._extract_faces(video_path)

        if crops is None:
            # Return zeros for failed samples
            return {
                "crops": torch.zeros(self.num_frames, 3, 224, 224),
                "label": torch.tensor(label, dtype=torch.long),
                "valid": False,
            }

        T_actual = min(len(crops), self.num_frames)
        tensors = []
        for i in range(T_actual):
            img = Image.fromarray(crops[i])

            # Compression augmentation (training only)
            if self.compress_aug:
                img = self.compress_aug(img)

            # Spatial augmentation
            img = self.spatial_aug(img)

            # Convert to CLIP tensor
            tensors.append(self.to_tensor(img))

        # Pad to num_frames
        while len(tensors) < self.num_frames:
            tensors.append(tensors[-1])

        return {
            "crops": torch.stack(tensors[:self.num_frames]),
            "label": torch.tensor(label, dtype=torch.long),
            "valid": True,
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    all_labels, all_probs, total_loss = [], [], 0.0

    for batch in tqdm(loader, leave=False, desc="Train"):
        crops = batch["crops"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits, embeddings = model(crops, labels)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        probs = F.softmax(logits.detach(), dim=-1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), auc


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []

    for batch in tqdm(loader, leave=False, desc="Eval"):
        crops = batch["crops"].to(device)
        labels = batch["label"]

        with torch.amp.autocast("cuda"):
            logits, _ = model(crops, labels=None)
        probs = F.softmax(logits, dim=-1)[:, 1]

        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    return auc, acc, all_labels, all_probs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train LNCLIP-DF")
    parser.add_argument("--data_dirs", nargs="+", default=["dataset_production", "dataset_week2", "dfd_dataset"])
    parser.add_argument("--cache_dir", type=str, default="./cache_lnclip")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_lnclip")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--arc_scale", type=float, default=30.0)
    parser.add_argument("--arc_margin", type=float, default=0.3)
    parser.add_argument("--sphere_noise", type=float, default=0.05)
    parser.add_argument("--num_ln_layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.save_dir).mkdir(exist_ok=True)

    print("=" * 60)
    print("LNCLIP-DF — LayerNorm-tuned CLIP for Deepfake Detection")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data dirs: {args.data_dirs}")
    print()

    # Load all samples
    print("Loading datasets...")
    all_samples = load_all_samples(args.data_dirs, seed=args.seed)
    n = len(all_samples)
    print(f"Total: {n} videos ({sum(1 for _,l,_,_ in all_samples if l==0)} real, "
          f"{sum(1 for _,l,_,_ in all_samples if l==1)} fake)")

    # Split: 70/10/20
    train_samples = all_samples[:int(0.7 * n)]
    val_samples = all_samples[int(0.7 * n):int(0.8 * n)]
    test_samples = all_samples[int(0.8 * n):]
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # Datasets
    train_ds = LNCLIPDataset(train_samples, args.cache_dir, args.num_frames, train=True)
    val_ds = LNCLIPDataset(val_samples, args.cache_dir, args.num_frames, train=False)
    test_ds = LNCLIPDataset(test_samples, args.cache_dir, args.num_frames, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Build model
    print("\nBuilding model...")
    model, _ = build_lnclip_model(
        device=device,
        num_trainable_layers=args.num_ln_layers,
        arc_scale=args.arc_scale,
        arc_margin=args.arc_margin,
        sphere_noise=args.sphere_noise,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable: {trainable_params:,} ({trainable_params/total_params*100:.3f}%)")

    # Optimizer (only trainable params)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # Training
    best_auc = 0.0
    patience, no_improve = 7, 0
    history = []

    print(f"\nTraining for up to {args.epochs} epochs (patience={patience})...\n")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_auc = train_epoch(model, train_loader, optimizer, device, scaler)
        vl_auc, vl_acc, _, _ = eval_epoch(model, val_loader, device)
        scheduler.step()

        flag = ""
        if vl_auc > best_auc:
            best_auc = vl_auc
            no_improve = 0
            torch.save(model.state_dict(), f"{args.save_dir}/lnclip_best.pt")
            flag = "  ← best"
        else:
            no_improve += 1

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"Loss={tr_loss:.4f}  Train AUC={tr_auc:.4f}  "
              f"Val AUC={vl_auc:.4f}  Val Acc={vl_acc:.4f}{flag}")

        history.append({"epoch": epoch, "tr_loss": tr_loss, "tr_auc": tr_auc,
                        "vl_auc": vl_auc, "vl_acc": vl_acc})

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    model.load_state_dict(torch.load(f"{args.save_dir}/lnclip_best.pt", map_location=device))
    test_auc, test_acc, test_labels, test_probs = eval_epoch(model, test_loader, device)

    print(f"Test AUC:      {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    # Save results
    results = {
        "model": "LNCLIP-DF (LN-tuned CLIP ViT-L/14 + Angular Margin)",
        "test_auc": round(test_auc, 4),
        "test_accuracy": round(test_acc, 4),
        "best_val_auc": round(best_auc, 4),
        "total_videos": n,
        "trainable_params": trainable_params,
        "trainable_pct": round(trainable_params / total_params * 100, 3),
        "epochs_trained": len(history),
        "config": vars(args),
    }

    torch.save(model.state_dict(), f"{args.save_dir}/lnclip_final.pt")
    with open(f"{args.save_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{args.save_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved to {args.save_dir}/")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
