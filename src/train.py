# src/train.py
import os
import sys
import glob
import json
import random
from dataclasses import asdict
from pathlib import Path

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import AppConfig
from src.model import SignGRU


class NPSequenceDataset(Dataset):
    def __init__(self, samples):
        """
        samples: list of (path, class_index)
        each file is a numpy array of shape (T, F)
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        x = np.load(path)  # (T, F)
        x = x.astype(np.float32)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def make_splits(cfg: AppConfig, seed: int = 42, val_ratio: float = 0.2):
    """
    Stratified split per label so every class appears in train/val.
    Expects data/<label>/*.npy files.
    """
    rng = random.Random(seed)

    label_to_files = {}
    for i, lab in enumerate(cfg.labels):
        files = sorted(glob.glob(os.path.join("data", lab, "*.npy")))
        label_to_files[i] = files

    train, val = [], []
    for y, files in label_to_files.items():
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_ratio)) if len(files) >= 5 else 1 if len(files) > 1 else 0
        val_files = files[:n_val]
        train_files = files[n_val:]

        # if a class has only 1 file, keep it in train
        if len(files) == 1:
            val_files = []
            train_files = files

        train += [(p, y) for p in train_files]
        val += [(p, y) for p in val_files]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def main():
    cfg = AppConfig()

    # --- settings you can tweak ---
    seed = 42
    val_ratio = 0.2
    batch_size = 32
    lr = 1e-3
    epochs = 50  # More epochs for better accuracy
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    weight_decay = 1e-5  # Regularization
    # -----------------------------

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_samples, val_samples = make_splits(cfg, seed=seed, val_ratio=val_ratio)

    if len(train_samples) == 0:
        raise RuntimeError("No training samples found. Did you record data into data/<label>/*.npy ?")

    print(f"Labels ({len(cfg.labels)}): {cfg.labels}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

    # show per-class counts
    def count_per_class(samples):
        counts = [0] * len(cfg.labels)
        for _, y in samples:
            counts[y] += 1
        return counts

    print("Train per class:", count_per_class(train_samples))
    print("Val per class:  ", count_per_class(val_samples))

    train_ds = NPSequenceDataset(train_samples)
    val_ds = NPSequenceDataset(val_samples) if len(val_samples) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False) if val_ds else None

    # infer input dims from first sample
    x0, _ = train_ds[0]
    T, F = x0.shape
    num_classes = len(cfg.labels)
    print(f"Input shape: T={T}, F={F}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    model = SignGRU(input_size=F, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, dropout=dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)  # (B, C)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(logits.detach(), yb)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item()
                    val_acc += accuracy(logits, yb)
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
        else:
            val_loss, val_acc = 0.0, 0.0

        print(f"Epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")

        # Learning rate scheduling
        if val_loader:
            scheduler.step(val_acc)
        
        # save best
        if val_loader and val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "models/signlink_gru.pt")
            with open("models/labels.json", "w") as f:
                json.dump(cfg.labels, f, indent=2)
            print(f"  ✓ saved best (val_acc={val_acc:.3f}) -> models/signlink_gru.pt")

    # if no val set, still save final
    if not val_loader:
        torch.save(model.state_dict(), "models/signlink_gru.pt")
        with open("models/labels.json", "w") as f:
            json.dump(cfg.labels, f, indent=2)
        print("✓ saved -> models/signlink_gru.pt")

    print("Training complete.")


if __name__ == "__main__":
    main()
