#!/usr/bin/env python3
"""
SignLink Virtual Camera - Continuous Prediction (Debug Version)

This version shows predictions continuously as you sign.
Used to verify the model is working before adding velocity filtering.
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
import pyvirtualcam

from src.overlay import draw_caption, draw_status
from src.model import SignGRU


def softmax_np(x: np.ndarray):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


# Display labels
DISPLAY_LABELS = {
    "hello": "Hello",
    "how_are_you": "How are you?",
    "my_name_is": "My name is",
    "kat": "Kat",
    "nice_to_meet_you": "Nice to meet you",
    "welcome_to_signlink": "Welcome to SignLink",
    "thank_you": "Thank you",
}


def format_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def main():
    print("=" * 70)
    print("SIGNLINK - CONTINUOUS PREDICTION (DEBUG)")
    print("=" * 70)
    print()
    print("Shows predictions in REAL-TIME as you sign")
    print("This is to verify your model is working")
    print()

    T = 30  # Window size
    camera_index = 0
    width, height = 1280, 720

    # Model setup
    pose_model_path = "models_signlink_pose/pose_gru.pt"
    pose_labels_path = "models_signlink_pose/pose_labels.json"

    if not os.path.exists(pose_model_path):
        print(f"ERROR: Model not found at {pose_model_path}")
        return

    print("Using POSE model (body + hands)")
    from src.features_hands_pose import HandsPoseFeatureExtractor
    extractor = HandsPoseFeatureExtractor()
    input_size = 156
    model_path = pose_model_path
    labels_path = pose_labels_path

    # Camera
    print(f"\nOpening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"Camera: {actual_w}x{actual_h} @ {fps}fps")

    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    with open(labels_path, "r") as f:
        labels = json.load(f)

    model = SignGRU(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=len(labels),
        dropout=0.2
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded: {len(labels)} signs")
    print(f"Labels: {labels}")

    # State
    window = deque(maxlen=T)
    prev_feat = None
    frame_count = 0

    # Virtual camera
    print("\nInitializing virtual camera...")
    try:
        try:
            cam = pyvirtualcam.Camera(
                width=actual_w, height=actual_h, fps=fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
                backend=pyvirtualcam.Backend.OBS
            )
        except (AttributeError, ValueError):
            cam = pyvirtualcam.Camera(
                width=actual_w, height=actual_h, fps=fps,
                fmt=pyvirtualcam.PixelFormat.BGR
            )
        print(f"Virtual camera: {cam.device}")
    except Exception as e:
        print(f"ERROR: Virtual camera failed: {e}")
        cap.release()
        return

    print()
    print("=" * 70)
    print("CONTINUOUS MODE:")
    print("  - Predictions update in real-time")
    print("  - Watch the subtitle change as you sign")
    print("  - Velocity bar shows hand movement speed")
    print()
    print("CONTROLS: Q=quit, C=clear")
    print("=" * 70 + "\n")

    try:
        with cam:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Extract features
                feat, info = extractor.extract(rgb)
                window.append(feat)

                left = int(info["left_present"])
                right = int(info["right_present"])
                hands_detected = left > 0 or right > 0

                # Compute velocity
                velocity = 0.0
                if hands_detected and prev_feat is not None:
                    if left > 0:
                        left_vel = np.linalg.norm(feat[0:63] - prev_feat[0:63])
                        velocity += left_vel
                    if right > 0:
                        right_vel = np.linalg.norm(feat[63:126] - prev_feat[63:126])
                        velocity += right_vel
                    if left > 0 and right > 0:
                        velocity /= 2

                prev_feat = feat.copy() if hands_detected else None

                # Run inference every 3 frames when we have enough data
                caption = ""
                conf = None

                if hands_detected and len(window) >= T and frame_count % 3 == 0:
                    x = np.stack(list(window), axis=0).astype(np.float32)
                    xt = torch.from_numpy(x).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(xt).cpu().numpy()[0]

                    probs = softmax_np(logits)
                    pred = int(np.argmax(probs))
                    conf = float(probs[pred])

                    caption = format_label(labels[pred])

                    # Print every 15 frames
                    if frame_count % 15 == 0:
                        print(f"Prediction: {caption} ({conf:.2f}) | vel={velocity:.4f}")

                # Build status
                status = f"L={left} R={right} | vel={velocity:.4f} | frames={len(window)}/{T}"

                draw_status(frame, status)

                if caption and conf is not None:
                    draw_caption(frame, caption, conf)

                # Border
                if hands_detected:
                    color = (0, 255, 0)  # Green when hands detected
                else:
                    color = (128, 128, 128)  # Gray
                cv2.rectangle(frame, (0, 0), (actual_w-1, actual_h-1), color, 6)

                # Velocity bar
                if hands_detected:
                    bar_x = 20
                    bar_y = actual_h - 60
                    bar_max_width = 200

                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + 20), (50, 50, 50), -1)

                    vel_normalized = min(velocity / 0.1, 1.0)
                    bar_width = int(bar_max_width * vel_normalized)
                    bar_color = (0, 255, 0) if velocity > 0.01 else (0, 165, 255)

                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), bar_color, -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + 20), (255, 255, 255), 2)
                    cv2.putText(frame, f"Velocity: {velocity:.4f}", (bar_x, bar_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cam.send(frame)
                cv2.imshow("SignLink Continuous", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    window.clear()

                frame_count += 1
                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nStopped")


if __name__ == "__main__":
    main()
