#!/usr/bin/env python3
"""
Train MLP for ASL alphabet recognition using MediaPipe hand features.

Uses 63-dim hand landmark features (21 landmarks * 3 coords) extracted
from images. Compatible with real-time MediaPipe inference.
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


# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data_alphabet"
MODEL_DIR = PROJECT_DIR / "models_alphabet"

# Labels: a-z + special classes
LABELS = list("abcdefghijklmnopqrstuvwxyz") + ["del", "nothing", "space"]


class AlphabetMLP(nn.Module):
    """MLP classifier for 63-dim hand landmark features."""

    def __init__(self, input_dim=63, num_classes=29, hidden_dims=[256, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AlphabetFeaturesDataset(Dataset):
    """Dataset for pre-extracted .npy hand features."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label_idx = self.samples[idx]
        features = np.load(npy_path)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)


def normalize_landmarks(features):
    """Normalize hand landmarks relative to wrist (landmark 0)."""
    features = features.copy()

    # Reshape to (21, 3) for easier processing
    landmarks = features.reshape(21, 3)

    # Translate so wrist is at origin
    wrist = landmarks[0].copy()
    landmarks -= wrist

    # Scale by hand size (distance from wrist to middle finger tip)
    middle_tip = landmarks[12]
    scale = np.linalg.norm(middle_tip)
    if scale > 0:
        landmarks /= scale

    return landmarks.flatten()


def main():
    print("=" * 60)
    print("TRAIN ALPHABET MLP (MediaPipe Features)")
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

    # Create datasets
    train_ds = AlphabetFeaturesDataset(train_samples)
    val_ds = AlphabetFeaturesDataset(val_samples)

    # Weighted sampler for balanced training
    train_labels = [s[1] for s in train_samples]
    class_counts_train = [train_labels.count(i) for i in range(len(LABELS))]
    weights = [1.0 / class_counts_train[label] if class_counts_train[label] > 0 else 0
               for _, label in train_samples]
    sampler = WeightedRandomSampler(weights, len(train_samples), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    # Model
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    num_classes = len(LABELS)
    model = AlphabetMLP(input_dim=63, num_classes=num_classes).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

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

    model_path = MODEL_DIR / "alphabet_mlp.pt"
    labels_path = MODEL_DIR / "alphabet_labels.json"
    config_path = MODEL_DIR / "alphabet_config.json"

    torch.save(best_state, model_path)
    with open(labels_path, "w") as f:
        json.dump(LABELS, f, indent=2)
    with open(config_path, "w") as f:
        json.dump({
            "num_classes": num_classes,
            "input_dim": 63,
            "hidden_dims": [256, 128, 64],
            "model_type": "AlphabetMLP",
            "feature_type": "mediapipe_hands_63dim"
        }, f, indent=2)

    print(f"\nModel saved to: {MODEL_DIR}/")
    print("  - alphabet_mlp.pt")
    print("  - alphabet_labels.json")
    print("  - alphabet_config.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
