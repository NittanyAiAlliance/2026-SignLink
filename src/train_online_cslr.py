#!/usr/bin/env python3
"""
Train model for Online CSLR (Continuous Sign Language Recognition)

Based on Zuo et al. (EMNLP 2024) approach:
1. Background class for "not signing"
2. Saliency loss - forces model to focus on core sign, not transitions
3. Frame masking augmentation - makes model robust to partial observations
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# Define model here to avoid import issues
class SignGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        feat = out.mean(dim=1)
        return self.classifier(feat)


# 7 phrases + background = 8 classes
LABELS = [
    "hello",
    "how_are_you",
    "my_name_is",
    "kat",
    "nice_to_meet_you",
    "welcome_to_signlink",
    "thank_you",
    "background"
]

DATA_DIR = Path("data_pose_personal")
MODEL_DIR = Path("models_signlink_pose")


class OnlineCSLRDataset(Dataset):
    """Dataset with augmentations for online CSLR training."""

    def __init__(self, samples, max_frames=30, augment=False, mask_ratio=0.2):
        self.samples = samples
        self.max_frames = max_frames
        self.augment = augment
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.samples)

    def _augment(self, x, is_background=False):
        """Data augmentation with saliency-inspired masking."""
        T, F = x.shape

        # 1. Frame masking (saliency loss inspiration)
        if np.random.random() < 0.5 and not is_background:
            num_mask = int(T * self.mask_ratio)
            mask_indices = np.random.choice(T, num_mask, replace=False)
            for idx in mask_indices:
                if np.random.random() < 0.5:
                    x[idx] = 0
                else:
                    x[idx] = x[idx] + np.random.normal(0, 0.1, F)

        # 2. Add global noise
        if np.random.random() < 0.5:
            x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)

        # 3. Scale variation
        if np.random.random() < 0.5:
            x = x * np.random.uniform(0.9, 1.1)

        # 4. Time shift
        if np.random.random() < 0.3:
            shift = np.random.randint(-3, 4)
            if shift > 0:
                x = np.concatenate([np.zeros((shift, F), dtype=np.float32), x[:-shift]])
            elif shift < 0:
                x = np.concatenate([x[-shift:], np.zeros((-shift, F), dtype=np.float32)])

        return x.astype(np.float32)

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]
        x = np.load(path).astype(np.float32)

        T, F = x.shape
        is_background = (label_idx == LABELS.index("background"))

        if T > self.max_frames:
            if self.augment:
                start = np.random.randint(0, T - self.max_frames + 1)
            else:
                start = (T - self.max_frames) // 2
            x = x[start:start + self.max_frames]
        elif T < self.max_frames:
            pad = np.zeros((self.max_frames - T, F), dtype=np.float32)
            x = np.concatenate([x, pad])

        if self.augment:
            x = self._augment(x, is_background)

        return torch.from_numpy(x), torch.tensor(label_idx, dtype=torch.long)


class SaliencyLoss(nn.Module):
    """Combined loss with saliency component."""

    def __init__(self, label_smoothing=0.1, saliency_weight=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.saliency_weight = saliency_weight

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
        saliency = entropy * self.saliency_weight
        return ce + saliency


def main():
    print("=" * 60)
    print("TRAIN ONLINE CSLR MODEL")
    print("(Zuo et al. EMNLP 2024 inspired)")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"\nERROR: Data directory not found: {DATA_DIR}")
        return

    # Load samples
    print("\nLoading samples...")
    samples = []
    class_counts = []

    for i, label in enumerate(LABELS):
        label_dir = DATA_DIR / label
        if label_dir.exists():
            files = list(label_dir.glob("*.npy"))
            valid_files = []
            for f in files:
                data = np.load(f)
                if data.shape[1] == 156:
                    valid_files.append(f)

            samples.extend([(str(f), i) for f in valid_files])
            class_counts.append(len(valid_files))
            print(f"  {label}: {len(valid_files)} samples")
        else:
            class_counts.append(0)
            print(f"  {label}: 0 samples")

    if len(samples) == 0:
        print("\nERROR: No samples found!")
        return

    bg_idx = LABELS.index("background")
    if class_counts[bg_idx] == 0:
        print("\nERROR: No background samples!")
        print("Run 'python src/record_background.py' first.")
        return

    print(f"\nTotal: {len(samples)} samples across {len(LABELS)} classes")

    # Split train/val
    np.random.seed(42)
    np.random.shuffle(samples)
    n_val = max(len(LABELS), int(len(samples) * 0.15))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Weighted sampler
    train_labels = [s[1] for s in train_samples]
    class_counts_train = [train_labels.count(i) for i in range(len(LABELS))]
    weights = [1.0 / class_counts_train[label] if class_counts_train[label] > 0 else 0
               for _, label in train_samples]
    sampler = WeightedRandomSampler(weights, len(train_samples) * 2, replacement=True)

    # Datasets
    train_ds = OnlineCSLRDataset(train_samples, max_frames=30, augment=True, mask_ratio=0.2)
    val_ds = OnlineCSLRDataset(val_samples, max_frames=30, augment=False)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = SignGRU(
        input_size=156,
        hidden_size=128,
        num_layers=2,
        num_classes=len(LABELS),
        dropout=0.3
    ).to(device)

    criterion = SaliencyLoss(label_smoothing=0.1, saliency_weight=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-5)

    # Training
    epochs = 120
    best_acc = 0.0
    best_state = None

    print(f"\nTraining for {epochs} epochs with saliency loss...")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += yb.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        per_class_correct = [0] * len(LABELS)
        per_class_total = [0] * len(LABELS)

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)

                for p, t in zip(preds.cpu(), yb.cpu()):
                    per_class_total[t.item()] += 1
                    if p.item() == t.item():
                        per_class_correct[t.item()] += 1
                        val_correct += 1
                    val_total += 1

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        per_class_acc = [c/t if t > 0 else 0 for c, t in zip(per_class_correct, per_class_total)]
        bal_acc = np.mean(per_class_acc)

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | bal_acc={bal_acc:.3f}")

        if bal_acc > best_acc:
            best_acc = bal_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best! bal_acc={bal_acc:.3f}")

    print("\n" + "=" * 60)
    print(f"Training complete! Best balanced accuracy: {best_acc*100:.1f}%")

    # Per-class results
    print("\nPer-class accuracy:")
    model.load_state_dict(best_state)
    model.eval()

    per_class_correct = [0] * len(LABELS)
    per_class_total = [0] * len(LABELS)

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            for p, t in zip(preds.cpu(), yb.cpu()):
                per_class_total[t.item()] += 1
                if p.item() == t.item():
                    per_class_correct[t.item()] += 1

    for i, label in enumerate(LABELS):
        if per_class_total[i] > 0:
            acc = 100 * per_class_correct[i] / per_class_total[i]
            marker = "+" if acc >= 80 else "-"
            print(f"  {marker} {label:25s}: {per_class_correct[i]}/{per_class_total[i]} = {acc:.1f}%")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "online_cslr.pt"
    labels_path = MODEL_DIR / "online_cslr_labels.json"
    config_path = MODEL_DIR / "online_cslr_config.json"

    torch.save(best_state, model_path)
    with open(labels_path, "w") as f:
        json.dump(LABELS, f, indent=2)
    with open(config_path, "w") as f:
        json.dump({
            "input_size": 156,
            "hidden_size": 128,
            "num_layers": 2,
            "num_classes": len(LABELS),
            "has_background": True,
            "background_index": bg_idx,
            "training": "online_cslr_saliency"
        }, f, indent=2)

    print(f"\nModel saved to: {MODEL_DIR}/")
    print("  - online_cslr.pt")
    print("  - online_cslr_labels.json")
    print("  - online_cslr_config.json")
    print()
    print("Next: Run 'python src/app_live_online_cslr_vcam.py'")
    print("=" * 60)


if __name__ == "__main__":
    main()
