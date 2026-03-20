#!/usr/bin/env python3
"""
SignLink Live App - Zuo et al. Online CSLR Method

Uses sliding window with:
1. Background elimination
2. Majority voting
3. Duplicate removal

Usage:
  python src/app_live_zuo.py
"""

import json
import os
import sys
from pathlib import Path
from collections import deque, Counter
import importlib.util

import cv2
import numpy as np
import torch
import torch.nn as nn
import pyvirtualcam


# Load feature extractor without triggering src/__init__.py
def load_feature_extractor():
    module_path = Path(__file__).parent / "features_hands_pose.py"
    spec = importlib.util.spec_from_file_location("features_hands_pose", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HandsPoseFeatureExtractor()


# Model definition (same as training)
class SignGRUOnline(nn.Module):
    def __init__(self, input_dim=156, hidden_dim=128,
                 num_classes=8, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features, _ = self.gru(x)
        pooled = features.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, features


# Display labels
DISPLAY_LABELS = {
    "hello": "Hello",
    "how_are_you": "How are you?",
    "my_name_is": "My name is",
    "kat": "Kat",
    "nice_to_meet_you": "Nice to meet you",
    "welcome_to_signlink": "Welcome to SignLink",
    "thank_you": "Thank you",
    "background": None,
}


def format_label(label):
    if label == "background":
        return None
    return DISPLAY_LABELS.get(label, label)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


# Inline overlay functions
def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=15, alpha=0.85):
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_caption(frame, caption, conf=None):
    if not caption or caption.strip() == "":
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    thickness = 2
    (tw, th), _ = cv2.getTextSize(caption, font, scale, thickness)
    x = (w - tw) // 2
    y = h - 60
    pad_x, pad_y = 25, 18
    box_x1 = max(0, x - pad_x)
    box_y1 = max(0, y - th - pad_y)
    box_x2 = min(w, x + tw + pad_x)
    box_y2 = min(h, y + pad_y)
    draw_rounded_rect(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1, radius=8, alpha=0.7)
    cv2.putText(frame, caption, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    print("=" * 60)
    print("SIGNLINK - ZUO et al. ONLINE CSLR")
    print("=" * 60)

    # Settings
    WINDOW_SIZE = 30
    STRIDE = 1  # Process every frame
    BAG_SIZE = 7  # Majority voting over this many predictions
    HOLD_FRAMES = 60  # Show subtitle for ~2 seconds
    CONF_THRESHOLD = 0.85  # Minimum confidence to accept a prediction

    # Model paths
    model_path = "models_zuo/online_cslr_zuo.pt"
    labels_path = "models_zuo/online_cslr_zuo_labels.json"

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Run 'python scripts/train_zuo.py' first.")
        return

    # Load model
    print(f"\nLoading model: {model_path}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    with open(labels_path, "r") as f:
        labels = json.load(f)

    bg_idx = labels.index("background") if "background" in labels else -1
    print(f"Labels: {labels}")
    print(f"Background index: {bg_idx}")

    model = SignGRUOnline(
        input_dim=156,
        hidden_dim=128,
        num_classes=len(labels),
        num_layers=2,
        dropout=0.3
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded: {len(labels)} classes")

    # Feature extractor
    print("\nInitializing feature extractor...")
    extractor = load_feature_extractor()

    # Camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"Camera: {actual_w}x{actual_h} @ {fps}fps")

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

    # State
    frame_buffer = deque(maxlen=WINDOW_SIZE)
    prediction_bag = deque(maxlen=BAG_SIZE)
    last_output = None
    last_was_blank = True

    # Display
    subtitle_text = ""
    subtitle_hold = 0
    gesture_count = 0

    print()
    print("=" * 60)
    print("ZUO et al. ONLINE CSLR:")
    print(f"  Window size: {WINDOW_SIZE} frames")
    print(f"  Bag size: {BAG_SIZE} predictions")
    print(f"  Stride: {STRIDE}")
    print()
    print("METHOD:")
    print("  1. Sliding window classification")
    print("  2. Majority voting over prediction bag")
    print("  3. Background elimination")
    print("  4. Duplicate removal")
    print()
    print("CONTROLS: Q=quit, C=clear")
    print("=" * 60 + "\n")

    try:
        with cam:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Extract features
                feat, info = extractor.extract(rgb)
                frame_buffer.append(feat)

                left = int(info["left_present"])
                right = int(info["right_present"])
                hands_detected = left > 0 or right > 0

                # Subtitle hold countdown
                if subtitle_hold > 0:
                    subtitle_hold -= 1
                    if subtitle_hold == 0:
                        subtitle_text = ""

                # Current prediction display
                current_pred = "ready"
                current_conf = 0.0

                # Run inference when buffer is full
                if hands_detected and len(frame_buffer) >= WINDOW_SIZE:
                    window = np.array(list(frame_buffer), dtype=np.float32)
                    window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits, _ = model(window_tensor)
                        probs = softmax(logits.cpu().numpy()[0])
                        pred_idx = int(np.argmax(probs))
                        conf = float(probs[pred_idx])

                    current_pred = labels[pred_idx]
                    current_conf = conf

                    # Add to prediction bag only if confidence is high enough
                    if conf >= CONF_THRESHOLD:
                        prediction_bag.append(pred_idx)
                    else:
                        prediction_bag.append(bg_idx)

                    # Majority voting when bag is full
                    if len(prediction_bag) >= BAG_SIZE:
                        counts = Counter(prediction_bag)
                        most_common_idx, count = counts.most_common(1)[0]

                        # Only accept if majority (> half)
                        if count > BAG_SIZE // 2:
                            vote = most_common_idx
                        else:
                            vote = bg_idx  # No clear majority = background

                        # Background elimination
                        if vote == bg_idx:
                            last_was_blank = True
                        else:
                            # New sign detected (not background)
                            is_new = (vote != last_output) or last_was_blank

                            if is_new:
                                gesture_count += 1
                                gloss = labels[vote]
                                display = format_label(gloss)

                                if display:
                                    subtitle_text = display
                                    subtitle_hold = HOLD_FRAMES
                                    print(f"\n[{gesture_count}] {display}")

                                last_output = vote
                                last_was_blank = False

                # Display
                display_frame = frame.copy()

                # Status bar
                status = f"L={left} R={right}"
                if current_pred != "ready" and current_pred != "background":
                    status += f" | {format_label(current_pred)} ({current_conf:.0%})"
                elif current_pred == "background":
                    status += " | ready"

                cv2.putText(display_frame, status, (15, 30),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)

                # Subtitle
                if subtitle_text:
                    draw_caption(display_frame, subtitle_text)

                # Border color
                if subtitle_text and subtitle_hold > 0:
                    color = (0, 255, 0)  # Green - showing
                elif hands_detected and current_pred != "background":
                    color = (0, 255, 255)  # Yellow - detecting
                elif hands_detected:
                    color = (150, 150, 150)  # Gray - ready
                else:
                    color = (80, 80, 80)  # Dark - no hands

                cv2.rectangle(display_frame, (0, 0), (actual_w-1, actual_h-1), color, 4)

                # Confidence bar
                if hands_detected and len(frame_buffer) >= WINDOW_SIZE:
                    bar_x = actual_w // 2 - 150
                    bar_y = 50
                    bar_w = 300
                    bar_h = 20

                    cv2.rectangle(display_frame, (bar_x, bar_y),
                                 (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

                    fill = int(bar_w * current_conf)
                    bar_color = (0, 255, 0) if current_conf >= 0.9 else (0, 165, 255)
                    cv2.rectangle(display_frame, (bar_x, bar_y),
                                 (bar_x + fill, bar_y + bar_h), bar_color, -1)

                    cv2.rectangle(display_frame, (bar_x, bar_y),
                                 (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

                    pred_display = format_label(current_pred) if current_pred != "background" else "ready"
                    cv2.putText(display_frame, f"{pred_display}: {current_conf:.0%}",
                               (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cam.send(display_frame)
                cv2.imshow("SignLink Zuo CSLR", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    frame_buffer.clear()
                    prediction_bag.clear()
                    subtitle_text = ""
                    subtitle_hold = 0
                    last_output = None
                    last_was_blank = True

                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nStopped. Recognized {gesture_count} signs.")


if __name__ == "__main__":
    main()
