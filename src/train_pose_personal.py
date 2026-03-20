#!/usr/bin/env python3
"""
Train GRU model on personal 7 phrases with POSE features (156-dim).

This trains a model using body + hands features for better accuracy.
Saves to models_signlink_pose/ so app_live_body_hands_vcam.py can use it.
"""

import json
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.model import SignGRU


# Your 7 personal phrases
LABELS = [
    "hello",
    "how_are_you",
    "my_name_is",
    "kat",
    "nice_to_meet_you",
    "welcome_to_signlink",
    "thank_you"
]

DATA_DIR = Path("data_pose_personal")
MODEL_DIR = Path("models_signlink_pose")


class SignDataset(Dataset):
    def __init__(self, samples, max_frames=60, augment=False):
        self.samples = samples
        self.max_frames = max_frames
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _augment(self, x):
        """Data augmentation for pose features."""
        T, F = x.shape

        # Add noise
        if np.random.random() < 0.5:
            x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)

        # Scale
        if np.random.random() < 0.5:
            x = x * np.random.uniform(0.9, 1.1)

        # Time shift (small)
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

        # Pad or truncate to max_frames
        if T > self.max_frames:
            start = (T - self.max_frames) // 2
            x = x[start:start + self.max_frames]
        elif T < self.max_frames:
            pad = np.zeros((self.max_frames - T, F), dtype=np.float32)
            x = np.concatenate([x, pad])

        if self.augment:
            x = self._augment(x)

        return torch.from_numpy(x), torch.tensor(label_idx, dtype=torch.long)


def main():
    print("=" * 60)
    print("TRAIN POSE MODEL - 7 Personal Phrases")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"\nERROR: Data directory not found: {DATA_DIR}")
        print("Run record_dataset_pose.py first to record samples.")
        return

    # Load samples
    print("\nLoading samples...")
    samples = []
    class_counts = []

    for i, label in enumerate(LABELS):
        label_dir = DATA_DIR / label
        if label_dir.exists():
            files = list(label_dir.glob("*.npy"))
            # Only load 156-dim files
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
            print(f"  {label}: 0 samples (directory not found)")

    if len(samples) == 0:
        print("\nERROR: No samples found!")
        print("Run record_dataset_pose.py first to record samples.")
        return

    # Check minimum samples
    min_samples = min(class_counts)
    if min_samples < 5:
        print(f"\nWARNING: Some phrases have less than 5 samples.")
        print("Consider recording more samples for better accuracy.")

    print(f"\nTotal: {len(samples)} samples")

    # Split train/val
    np.random.seed(42)
    np.random.shuffle(samples)
    n_val = max(len(LABELS), int(len(samples) * 0.15))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Create weighted sampler for balanced training
    train_labels = [s[1] for s in train_samples]
    class_counts_train = [train_labels.count(i) for i in range(len(LABELS))]
    weights = [1.0 / class_counts_train[label] if class_counts_train[label] > 0 else 0
               for _, label in train_samples]
    sampler = WeightedRandomSampler(weights, len(train_samples) * 2, replacement=True)

    # Create datasets
    train_ds = SignDataset(train_samples, max_frames=30, augment=True)
    val_ds = SignDataset(val_samples, max_frames=30, augment=False)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = SignGRU(
        input_size=156,  # Pose features
        hidden_size=128,
        num_layers=2,
        num_classes=len(LABELS),
        dropout=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    # Training
    epochs = 100
    best_acc = 0.0
    best_state = None

    print(f"\nTraining for {epochs} epochs...")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # Train
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

        # Validate
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

        # Balanced accuracy
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
            print(f"  {marker} {label:20s}: {per_class_correct[i]}/{per_class_total[i]} = {acc:.1f}%")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "pose_gru.pt"
    labels_path = MODEL_DIR / "pose_labels.json"
    config_path = MODEL_DIR / "pose_config.json"

    torch.save(best_state, model_path)
    with open(labels_path, "w") as f:
        json.dump(LABELS, f, indent=2)
    with open(config_path, "w") as f:
        json.dump({
            "input_size": 156,
            "hidden_size": 128,
            "num_layers": 2,
            "num_classes": len(LABELS)
        }, f, indent=2)

    print(f"\nModel saved to: {MODEL_DIR}/")
    print("  - pose_gru.pt")
    print("  - pose_labels.json")
    print("  - pose_config.json")
    print("\nYou can now run: python3 src/app_live_body_hands_vcam.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
