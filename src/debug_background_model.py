#!/usr/bin/env python3
"""
Debug tool to see what the background model is predicting in real-time.
Shows confidence for ALL classes so you can see what's happening.
"""

import json
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from collections import deque

import torch

from src.model import SignGRU


def softmax_np(x: np.ndarray):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


def main():
    print("=" * 60)
    print("DEBUG: Background Model Predictions")
    print("=" * 60)

    T = 30

    # Load model with background
    model_path = "models_signlink_pose/pose_gru_with_bg.pt"
    labels_path = "models_signlink_pose/pose_labels_with_bg.json"

    if not os.path.exists(model_path):
        print("ERROR: Background model not found")
        return

    from src.features_hands_pose import HandsPoseFeatureExtractor
    extractor = HandsPoseFeatureExtractor()

    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    with open(labels_path, "r") as f:
        labels = json.load(f)

    model = SignGRU(
        input_size=156,
        hidden_size=128,
        num_layers=2,
        num_classes=len(labels),
        dropout=0.2
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Labels: {labels}")
    print("\nShowing confidence for ALL classes")
    print("Watch what happens when you sign vs. when you don't")
    print("Press Q to quit\n")

    window = deque(maxlen=T)
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        feat, info = extractor.extract(rgb)
        window.append(feat)

        left = int(info["left_present"])
        right = int(info["right_present"])
        hands_detected = left > 0 or right > 0

        frame_count += 1

        # Run inference
        if hands_detected and len(window) >= T and frame_count % 3 == 0:
            x = np.stack(list(window), axis=0).astype(np.float32)
            xt = torch.from_numpy(x).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(xt).cpu().numpy()[0]

            probs = softmax_np(logits)

            # Print all confidences
            print("\n" + "=" * 50)
            for i, label in enumerate(labels):
                conf = probs[i]
                bar = "█" * int(conf * 30)
                marker = " <-- TOP" if i == np.argmax(probs) else ""
                print(f"{label:25s} {conf:5.1%} {bar}{marker}")

            # Draw on frame
            y_offset = 100
            for i, label in enumerate(labels):
                conf = probs[i]
                color = (0, 255, 0) if i == np.argmax(probs) else (200, 200, 200)
                bar_width = int(200 * conf)

                cv2.rectangle(frame, (10, y_offset), (10 + bar_width, y_offset + 20), color, -1)
                cv2.rectangle(frame, (10, y_offset), (210, y_offset + 20), (255, 255, 255), 1)
                cv2.putText(frame, f"{label}: {conf:.0%}", (220, y_offset + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 30

        # Status
        cv2.putText(frame, f"L={left} R={right} | frames={len(window)}/{T}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Debug Background Model", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
