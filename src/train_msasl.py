# src/train_msasl.py
# Training script for MS-ASL + How2Sign dataset (combined)
# Includes fixes for class imbalance via weighted sampling

import os
import sys
import glob
import json
import random
from pathlib import Path

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.config_msasl import MSASLConfig
from src.model import SignGRU


class NPSequenceDataset(Dataset):
    def __init__(self, samples, max_frames=300, augment=False):
        """
        samples: list of (path, class_index)
        each file is a numpy array of shape (T, F)
        max_frames: maximum number of frames to use (truncate or pad)
        augment: whether to apply data augmentation
        """
        self.samples = samples
        self.max_frames = max_frames
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _smart_sample_frames(self, x, target_frames):
        """
        Intelligently sample frames from long sequences.
        Uses activity-based sampling to keep the most important frames.
        """
        T, F = x.shape

        if T <= target_frames:
            return x

        # Calculate per-frame activity (motion/change between frames)
        if T > 1:
            motion = np.sum(np.abs(np.diff(x, axis=0)), axis=1)
            motion = np.concatenate([[motion[0]], motion])  # Pad to match length
        else:
            motion = np.ones(T)

        # Normalize motion to create sampling probability
        motion = motion + 0.1  # Add small constant to avoid zero probability
        prob = motion / motion.sum()

        # Sample frames based on activity (more active = more likely to be sampled)
        # But also ensure temporal ordering is preserved
        indices = np.sort(np.random.choice(T, size=target_frames, replace=False, p=prob))

        return x[indices]

    def _find_active_region(self, x):
        """Find the region with most hand activity (non-zero frames)."""
        # Calculate activity per frame (sum of absolute values)
        activity = np.sum(np.abs(x), axis=1)

        # Use a sliding window to find the most active region
        window_size = min(self.max_frames, len(activity))
        if len(activity) <= window_size:
            return 0, len(activity)

        # Compute cumulative sum for efficient window sum calculation
        cumsum = np.cumsum(activity)
        cumsum = np.insert(cumsum, 0, 0)

        # Find window with maximum activity
        window_sums = cumsum[window_size:] - cumsum[:-window_size]
        best_start = np.argmax(window_sums)

        return best_start, best_start + window_size

    def _augment(self, x):
        """Apply aggressive data augmentation to hand landmark sequence."""
        T, F = x.shape

        # 1. Random noise (jitter landmarks) - MORE AGGRESSIVE
        if np.random.random() < 0.8:
            noise = np.random.normal(0, 0.03, x.shape).astype(np.float32)
            x = x + noise

        # 2. Random scaling (simulate distance from camera) - MORE AGGRESSIVE
        if np.random.random() < 0.7:
            scale = np.random.uniform(0.7, 1.3)
            x = x * scale

        # 3. Random horizontal flip (mirror x coordinates)
        if np.random.random() < 0.5:
            for i in range(0, min(63, F), 3):
                x[:, i] = 1.0 - x[:, i]

        # 4. Random frame dropout (simulate occlusion) - MORE AGGRESSIVE
        if np.random.random() < 0.5:
            dropout_rate = np.random.uniform(0.05, 0.15)
            mask = np.random.random(T) > dropout_rate
            x = x * mask[:, np.newaxis]

        # 5. Random time warping (speed variation) - NEW
        if np.random.random() < 0.5:
            speed = np.random.uniform(0.8, 1.2)
            new_len = int(T * speed)
            new_len = max(10, min(new_len, T * 2))
            indices = np.linspace(0, T - 1, new_len, dtype=int)
            x_warped = x[indices]
            # Resize back to original length
            if len(x_warped) != T:
                indices2 = np.linspace(0, len(x_warped) - 1, T, dtype=int)
                x = x_warped[indices2]

        # 6. Random translation (shift x, y coordinates) - NEW
        if np.random.random() < 0.5:
            shift_x = np.random.uniform(-0.1, 0.1)
            shift_y = np.random.uniform(-0.1, 0.1)
            for i in range(0, min(63, F), 3):
                x[:, i] = x[:, i] + shift_x      # x coordinate
                x[:, i + 1] = x[:, i + 1] + shift_y  # y coordinate

        # 7. Random rotation (rotate around center) - NEW
        if np.random.random() < 0.4:
            angle = np.random.uniform(-15, 15) * np.pi / 180  # degrees to radians
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for i in range(0, min(63, F), 3):
                # Center coordinates around 0.5
                cx = x[:, i] - 0.5
                cy = x[:, i + 1] - 0.5
                # Rotate
                x[:, i] = cx * cos_a - cy * sin_a + 0.5
                x[:, i + 1] = cx * sin_a + cy * cos_a + 0.5

        # 8. Mixup-style interpolation within same sample - NEW
        if np.random.random() < 0.3:
            # Shuffle frames and blend
            alpha = np.random.uniform(0.7, 1.0)
            perm = np.random.permutation(T)
            x = alpha * x + (1 - alpha) * x[perm]

        # 9. Random cutout (zero out random frames) - NEW
        if np.random.random() < 0.3:
            cutout_len = np.random.randint(5, min(30, T // 4))
            start = np.random.randint(0, T - cutout_len)
            x[start:start + cutout_len] = 0

        return x

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        x = np.load(path)  # (T, F)
        x = x.astype(np.float32)

        T, F = x.shape

        if T > self.max_frames:
            if self.augment:
                # During training: use smart sampling for very long sequences
                if T > self.max_frames * 3:
                    # For very long sequences, use activity-based sampling
                    x = self._smart_sample_frames(x, self.max_frames)
                else:
                    # For moderately long sequences, use sliding window with jitter
                    start, end = self._find_active_region(x)
                    jitter = np.random.randint(-50, 50) if T > self.max_frames + 100 else 0
                    start = max(0, min(start + jitter, T - self.max_frames))
                    x = x[start:start + self.max_frames]
            else:
                # During validation: use the most active region
                start, end = self._find_active_region(x)
                x = x[start:start + self.max_frames]
        elif T < self.max_frames:
            # Pad with zeros
            padding = np.zeros((self.max_frames - T, F), dtype=np.float32)
            x = np.concatenate([x, padding], axis=0)

        # Apply augmentation if enabled
        if self.augment:
            x = self._augment(x)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def make_splits(cfg: MSASLConfig, seed: int = 42, val_ratio: float = 0.2):
    """
    Stratified split per label so every class appears in train/val.
    Loads msasl_*.npy, how2sign_*.npy, wlasl_*.npy, and aslcitizen_*.npy files from data_msasl/<label>/ folders.
    """
    rng = random.Random(seed)

    label_to_files = {}
    for i, lab in enumerate(cfg.labels):
        # Get MS-ASL, How2Sign, WLASL, and ASL Citizen files
        msasl_files = sorted(glob.glob(os.path.join("data_msasl", lab, "msasl_*.npy")))
        how2sign_files = sorted(glob.glob(os.path.join("data_msasl", lab, "how2sign_*.npy")))
        wlasl_files = sorted(glob.glob(os.path.join("data_msasl", lab, "wlasl_*.npy")))
        aslcitizen_files = sorted(glob.glob(os.path.join("data_msasl", lab, "aslcitizen_*.npy")))
        files = msasl_files + how2sign_files + wlasl_files + aslcitizen_files
        if len(files) > 0:
            label_to_files[i] = files

    if len(label_to_files) == 0:
        raise RuntimeError(
            "No training samples found! "
            "Expected msasl_*.npy or how2sign_*.npy files in data_msasl/<label>/ directories."
        )

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


def compute_sample_weights(samples, num_classes):
    """
    Compute sample weights for WeightedRandomSampler.
    Each sample gets weight = 1 / (class_frequency)
    This makes minority classes sampled more often.
    """
    # Count samples per class
    class_counts = [0] * num_classes
    for _, y in samples:
        class_counts[y] += 1

    # Compute weight for each class (inverse frequency)
    class_weights = []
    for count in class_counts:
        if count > 0:
            class_weights.append(1.0 / count)
        else:
            class_weights.append(0.0)

    # Assign weight to each sample based on its class
    sample_weights = [class_weights[y] for _, y in samples]

    return sample_weights, class_counts


def compute_class_weights(class_counts, device):
    """
    Compute class weights for CrossEntropyLoss.
    Uses inverse frequency with smoothing.
    """
    total = sum(class_counts)
    weights = []
    for count in class_counts:
        if count > 0:
            # Inverse frequency with sqrt smoothing to avoid extreme weights
            w = np.sqrt(total / (len(class_counts) * count))
            weights.append(w)
        else:
            weights.append(1.0)

    # Normalize so mean weight is 1
    weights = np.array(weights)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32, device=device)


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def main():
    print("=" * 70)
    print("🚀 SIGN LANGUAGE WORD RECOGNITION MODEL TRAINING")
    print("=" * 70)
    print("Dataset: MS-ASL + How2Sign (combined)")
    print("Training on msasl_*.npy and how2sign_*.npy files")
    print("=" * 70)
    print()

    cfg = MSASLConfig()

    # --- Training settings ---
    seed = 42
    val_ratio = 0.2
    batch_size = 8  # Smaller batch for more gradient updates
    lr = 5e-4  # Learning rate
    epochs = 200  # More epochs with heavy augmentation
    hidden_size = 128  # Smaller model to prevent overfitting
    num_layers = 2  # GRU layers
    dropout = 0.5  # Higher dropout for regularization
    weight_decay = 1e-3  # Stronger weight decay
    max_frames = 150  # Frames per sequence
    label_smoothing = 0.1  # Label smoothing for small dataset
    # -------------------------

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("📂 Loading MS-ASL data...")
    train_samples, val_samples = make_splits(cfg, seed=seed, val_ratio=val_ratio)

    print(f"\n📊 Dataset Statistics:")
    print(f"   Labels: {len(cfg.labels)} signs")
    print(f"   Train samples: {len(train_samples)}")
    print(f"   Val samples: {len(val_samples)}")

    # show per-class counts
    def count_per_class(samples):
        counts = [0] * len(cfg.labels)
        for _, y in samples:
            counts[y] += 1
        return counts

    train_counts = count_per_class(train_samples)
    val_counts = count_per_class(val_samples)

    print(f"\n📈 Samples per sign:")
    for i, label in enumerate(cfg.labels):
        total = train_counts[i] + val_counts[i]
        if total > 0:
            print(f"   {label:15s}: {train_counts[i]:2d} train + {val_counts[i]:2d} val = {total:2d} total")

    train_ds = NPSequenceDataset(train_samples, max_frames=max_frames, augment=True)
    val_ds = NPSequenceDataset(val_samples, max_frames=max_frames, augment=False) if len(val_samples) > 0 else None

    print(f"\n🔄 Data Augmentation: ENABLED for training")

    # Compute sample weights for balanced sampling
    sample_weights, class_counts = compute_sample_weights(train_samples, len(cfg.labels))
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_samples) * 2,  # Oversample to see minority classes more
        replacement=True
    )

    print(f"⚖️  Weighted Sampling: ENABLED (balancing class imbalance)")
    print(f"   Class counts: {dict(zip(cfg.labels, class_counts))}")

    # Use sampler instead of shuffle
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False) if val_ds else None

    # infer input dims from first sample
    x0, _ = train_ds[0]
    T, F = x0.shape
    num_classes = len(cfg.labels)
    print(f"\n🔧 Model Configuration:")
    print(f"   Input shape: T={T}, F={F}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Num layers: {num_layers}")
    print(f"   Dropout: {dropout}")
    print(f"   Classes: {num_classes}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   Device: {device}")

    model = SignGRU(input_size=F, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, dropout=dropout)
    model.to(device)

    # Compute class weights for loss function
    class_weights = compute_class_weights(class_counts, device)
    print(f"⚖️  Class-weighted Loss: ENABLED")
    print(f"   Weights: {dict(zip(cfg.labels, [f'{w:.2f}' for w in class_weights.tolist()]))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    best_val = 0.0
    # Save MS-ASL model to a SEPARATE folder
    model_dir = "models_msasl"
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n🎯 Training for {epochs} epochs...")
    print(f"   Model will be saved to: {model_dir}/")
    print("=" * 70)

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
            # Track per-class accuracy
            per_class_correct = [0] * num_classes
            per_class_total = [0] * num_classes

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item()
                    val_acc += accuracy(logits, yb)

                    # Per-class tracking
                    preds = logits.argmax(dim=1)
                    for pred, true in zip(preds.cpu().numpy(), yb.cpu().numpy()):
                        per_class_total[true] += 1
                        if pred == true:
                            per_class_correct[true] += 1

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            # Calculate balanced accuracy (mean of per-class accuracies)
            per_class_acc = []
            for i in range(num_classes):
                if per_class_total[i] > 0:
                    per_class_acc.append(per_class_correct[i] / per_class_total[i])
            balanced_acc = np.mean(per_class_acc) if per_class_acc else 0.0
        else:
            val_loss, val_acc, balanced_acc = 0.0, 0.0, 0.0

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f} bal_acc {balanced_acc:.3f} | lr {current_lr:.2e}")

        # Learning rate scheduling (cosine annealing)
        scheduler.step(epoch)

        # save best based on BALANCED accuracy (important for imbalanced data!)
        if val_loader and balanced_acc > best_val:
            best_val = balanced_acc
            torch.save(model.state_dict(), f"{model_dir}/msasl_gru.pt")
            with open(f"{model_dir}/msasl_labels.json", "w") as f:
                json.dump(cfg.labels, f, indent=2)
            print(f"  ✓ saved best (balanced_acc={balanced_acc:.3f}) -> {model_dir}/msasl_gru.pt")

        # Print per-class accuracy every 20 epochs
        if epoch % 20 == 0 and val_loader:
            print(f"\n  📊 Per-class accuracy at epoch {epoch}:")
            for i, label in enumerate(cfg.labels):
                if per_class_total[i] > 0:
                    acc = 100 * per_class_correct[i] / per_class_total[i]
                    status = "✓" if acc >= 50 else "✗"
                    print(f"     {status} {label:15s}: {per_class_correct[i]:2d}/{per_class_total[i]:2d} = {acc:5.1f}%")
            print()

    # if no val set, still save final
    if not val_loader:
        torch.save(model.state_dict(), f"{model_dir}/msasl_gru.pt")
        with open(f"{model_dir}/msasl_labels.json", "w") as f:
            json.dump(cfg.labels, f, indent=2)
        print(f"✓ saved -> {model_dir}/msasl_gru.pt")

    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print(f"📁 Model saved to: {model_dir}/msasl_gru.pt")
    print(f"📁 Labels saved to: {model_dir}/msasl_labels.json")
    print(f"🎯 Best BALANCED validation accuracy: {best_val:.3f}")
    print("   (Balanced accuracy = mean of per-class accuracies)")
    print("=" * 70)


if __name__ == "__main__":
    main()

