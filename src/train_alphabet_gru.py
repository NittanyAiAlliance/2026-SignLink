#!/usr/bin/env python3
"""
Train GRU for ASL alphabet recognition using MediaPipe hand features.

Uses the same SignGRU architecture as word recognition for consistency.
Treats static alphabet images as short sequences (repeated frames).
"""

import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.model import SignGRU


# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data_alphabet"
MODEL_DIR = PROJECT_DIR / "models_alphabet"

# Labels: a-z + special classes
LABELS = list("abcdefghijklmnopqrstuvwxyz") + ["del", "nothing", "space"]

# Sequence length - repeat static frame this many times
# This simulates a short recording of a static pose
SEQUENCE_LENGTH = 10


class AlphabetSequenceDataset(Dataset):
    """Dataset that converts single-frame features to sequences for GRU."""

    def __init__(self, samples, seq_len=SEQUENCE_LENGTH, augment=False):
        self.samples = samples
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label_idx = self.samples[idx]
        features = np.load(npy_path).astype(np.float32)  # (63,)

        # Create sequence by repeating the frame
        # Add small noise during training to simulate real variation
        if self.augment:
            # Create slightly varied sequence (simulates hand tremor/movement)
            sequence = np.stack([
                features + np.random.normal(0, 0.01, features.shape).astype(np.float32)
                for _ in range(self.seq_len)
            ])
        else:
            # Just repeat the frame
            sequence = np.stack([features] * self.seq_len)

        return (
            torch.tensor(sequence, dtype=torch.float32),  # (seq_len, 63)
            torch.tensor(label_idx, dtype=torch.long)
        )


def main():
    print("=" * 60)
    print("TRAIN ALPHABET GRU (Same architecture as words)")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"\nERROR: Data directory not found: {DATA_DIR}")
        print("Run scripts/extract_alphabet_features.py first!")
        return

    # Collect samples
    print("\nCollecting samples...")
    samples = []
    class_counts = []

    for i, label in enumerate(LABELS):
        label_dir = DATA_DIR / label
        if label_dir.exists():
            files = list(label_dir.glob("*.npy"))
            samples.extend([(str(f), i) for f in files])
            class_counts.append(len(files))
            print(f"  {label}: {len(files)} samples")
        else:
            class_counts.append(0)
            print(f"  {label}: 0 (not found)")

    if len(samples) == 0:
        print("\nERROR: No samples found!")
        print("Run scripts/extract_alphabet_features.py first!")
        return

    print(f"\nTotal: {len(samples)} samples across {sum(1 for c in class_counts if c > 0)} classes")

    # Split train/val (90/10)
    np.random.seed(42)
    np.random.shuffle(samples)
    n_val = int(len(samples) * 0.1)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"Sequence length: {SEQUENCE_LENGTH} (repeated frames)")

    # Create datasets
    train_ds = AlphabetSequenceDataset(train_samples, augment=True)
    val_ds = AlphabetSequenceDataset(val_samples, augment=False)

    # Weighted sampler for balanced training
    train_labels = [s[1] for s in train_samples]
    class_counts_train = [train_labels.count(i) for i in range(len(LABELS))]
    weights = [1.0 / class_counts_train[label] if class_counts_train[label] > 0 else 0
               for _, label in train_samples]
    sampler = WeightedRandomSampler(weights, len(train_samples), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    # Model - same architecture as word model
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    num_classes = len(LABELS)
    input_size = 63  # MediaPipe hand landmarks (21 * 3)
    hidden_size = 256
    num_layers = 2

    model = SignGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=0.3
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: SignGRU(input={input_size}, hidden={hidden_size}, layers={num_layers})")
    print(f"Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    # Training
    epochs = 50
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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += yb.size(0)

            pbar.set_postfix(loss=loss.item(), acc=train_correct/train_total)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        scheduler.step()

        print(f"Epoch {epoch:02d} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best! val_acc={val_acc:.3f}")

    print("\n" + "=" * 60)
    print(f"Training complete! Best accuracy: {best_acc*100:.1f}%")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "alphabet_gru.pt"
    labels_path = MODEL_DIR / "alphabet_labels.json"
    config_path = MODEL_DIR / "alphabet_config.json"

    torch.save(best_state, model_path)
    with open(labels_path, "w") as f:
        json.dump(LABELS, f, indent=2)
    with open(config_path, "w") as f:
        json.dump({
            "num_classes": num_classes,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "sequence_length": SEQUENCE_LENGTH,
            "model_type": "SignGRU",
            "feature_type": "mediapipe_hands_63dim"
        }, f, indent=2)

    print(f"\nModel saved to: {MODEL_DIR}/")
    print("  - alphabet_gru.pt")
    print("  - alphabet_labels.json")
    print("  - alphabet_config.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
