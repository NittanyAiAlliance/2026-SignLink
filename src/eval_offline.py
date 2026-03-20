import os, glob, json
import numpy as np
import torch
from collections import defaultdict

from src.config import AppConfig
from src.model import SignGRU

def main():
    cfg = AppConfig()

    labels_path = os.path.join("models", "labels.json")
    ckpt_path = os.path.join("models", "signlink_gru.pt")

    if not (os.path.exists(labels_path) and os.path.exists(ckpt_path)):
        print("Missing models/labels.json or models/signlink_gru.pt")
        return

    with open(labels_path, "r") as f:
        labels = json.load(f)
    cfg.labels = labels

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = SignGRU(input_size=128, hidden_size=128, num_layers=2, num_classes=len(cfg.labels), dropout=0.2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # load all samples
    paths = []
    ys = []
    for i, lab in enumerate(cfg.labels):
        files = glob.glob(os.path.join("data", lab, "**", "*.npy"), recursive=True)
        for p in files:
            paths.append(p)
            ys.append(i)

    if not paths:
        print("No data found in data/<label>/*.npy")
        return

    # evaluate
    correct = 0
    total = 0
    per_label = defaultdict(lambda: {"correct":0, "total":0})

    with torch.no_grad():
        for p, y in zip(paths, ys):
            x = np.load(p).astype(np.float32)         # (T,F)
            xt = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,T,F)
            logits = model(xt)[0]
            pred = int(torch.argmax(logits).item())

            total += 1
            per_label[y]["total"] += 1
            if pred == y:
                correct += 1
                per_label[y]["correct"] += 1

    print(f"\nOverall accuracy (on your recorded data): {correct}/{total} = {correct/total:.3f}")

    print("\nPer-label accuracy:")
    for i, lab in enumerate(cfg.labels):
        c = per_label[i]["correct"]
        t = per_label[i]["total"]
        acc = c / t if t else 0
        print(f"{lab:24s} {c:4d}/{t:4d}  acc={acc:.3f}")

if __name__ == "__main__":
    main()
