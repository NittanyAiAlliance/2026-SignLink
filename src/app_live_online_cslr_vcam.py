#!/usr/bin/env python3
"""
SignLink Virtual Camera - Online CSLR (Sliding Window)

Based on Zuo et al. (EMNLP 2024) inference approach:
1. Sliding window runs continuously
2. Background class suppression
3. Duplicate removal (same sign twice → show once)
4. Confidence thresholding
5. Prediction smoothing

This is the state-of-the-art approach for continuous sign recognition.
"""

import json
import os
import sys
from pathlib import Path
import importlib.util

import cv2
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import pyvirtualcam


def load_feature_extractor():
    """Load HandsPoseFeatureExtractor directly without triggering src/__init__.py"""
    module_path = Path(__file__).parent / "features_hands_pose.py"
    spec = importlib.util.spec_from_file_location("features_hands_pose", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HandsPoseFeatureExtractor()


# Define model inline to avoid import issues
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


# Inline overlay functions to avoid importing through src/__init__.py
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


def draw_caption(frame_bgr, caption, conf=None):
    if not caption or caption.strip() == "":
        return
    h, w = frame_bgr.shape[:2]
    text = f"{caption}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = h - 60
    pad_x, pad_y = 25, 18
    box_x1 = max(0, x - pad_x)
    box_y1 = max(0, y - th - pad_y)
    box_x2 = min(w, x + tw + pad_x)
    box_y2 = min(h, y + pad_y)
    draw_rounded_rect(frame_bgr, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1, radius=8, alpha=0.7)
    cv2.putText(frame_bgr, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_status(frame_bgr, status):
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    thickness = 1
    cv2.putText(frame_bgr, status, (15, 30), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)


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
    "background": None,
}


def format_label(label: str) -> str:
    if label == "background":
        return None
    return DISPLAY_LABELS.get(label, label)


def main():
    print("=" * 70)
    print("SIGNLINK - ONLINE CSLR (Zuo et al. EMNLP 2024)")
    print("=" * 70)

    # ==========================================================
    # SETTINGS
    # ==========================================================
    T = 30                          # Sliding window size
    STRIDE = 3                      # Inference every N frames

    # Thresholds - using same as confidence-based approach that worked
    COMMIT_THRESHOLD = 0.91         # Show subtitle at 91%+
    STABLE_FRAMES = 3               # Same prediction for 3 frames to commit

    HOLD_FRAMES = 60                # Show subtitle for 2 seconds

    camera_index = 0
    width, height = 1280, 720

    # ==========================================================
    # MODEL SETUP
    # ==========================================================
    model_path = "models_signlink_pose/online_cslr.pt"
    labels_path = "models_signlink_pose/online_cslr_labels.json"

    # Fallback to background model if online_cslr not trained yet
    if not os.path.exists(model_path):
        print("Online CSLR model not found, trying background model...")
        model_path = "models_signlink_pose/pose_gru_with_bg.pt"
        labels_path = "models_signlink_pose/pose_labels_with_bg.json"

    if not os.path.exists(model_path):
        print(f"\nERROR: No model found!")
        print("Run 'python src/train_online_cslr.py' first.")
        return

    print(f"Loading model: {model_path}")
    extractor = load_feature_extractor()

    # ==========================================================
    # CAMERA
    # ==========================================================
    print(f"Opening camera {camera_index}...")
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

    # ==========================================================
    # LOAD MODEL
    # ==========================================================
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    with open(labels_path, "r") as f:
        labels = json.load(f)

    bg_idx = labels.index("background") if "background" in labels else -1
    print(f"Labels: {labels}")
    print(f"Background index: {bg_idx}")

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
    print(f"Model loaded: {len(labels)} classes")

    # ==========================================================
    # STATE
    # ==========================================================
    window = deque(maxlen=T)

    # Tracking state
    last_pred_idx = -1
    stable_count = 0
    committed_label = ""  # Track what was last committed to avoid re-triggering

    # Current state
    current_pred = "background"
    current_conf = 0.0

    # Display state
    committed_caption = ""
    committed_conf = None
    hold_counter = 0

    # Stats
    gesture_count = 0
    frame_count = 0

    # ==========================================================
    # VIRTUAL CAMERA
    # ==========================================================
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
    print("CONTINUOUS RECOGNITION SETTINGS:")
    print(f"  Window size:        {T} frames")
    print(f"  Commit threshold:   {COMMIT_THRESHOLD*100:.0f}%")
    print(f"  Stable frames:      {STABLE_FRAMES}")
    print()
    print("HOW IT WORKS:")
    print("  1. Ignores 'background' predictions - always ready")
    print("  2. Shows subtitle when sign reaches 91%+ confidence")
    print("  3. NO PAUSE between signs - continuous detection")
    print("  4. Same sign can be repeated after subtitle fades")
    print()
    print("CONTROLS: Q=quit, C=clear")
    print("=" * 70 + "\n")

    # ==========================================================
    # MAIN LOOP
    # ==========================================================
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

                frame_count += 1

                # ==================================================
                # HOLD COUNTER
                # ==================================================
                if hold_counter > 0:
                    hold_counter -= 1
                    if hold_counter == 0:
                        committed_caption = ""
                        committed_conf = None
                        committed_label = ""  # Allow re-detection of same sign

                # ==================================================
                # SLIDING WINDOW INFERENCE
                # ==================================================
                if hands_detected and len(window) >= T and frame_count % STRIDE == 0:
                    x = np.stack(list(window), axis=0).astype(np.float32)
                    xt = torch.from_numpy(x).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(xt).cpu().numpy()[0]

                    probs = softmax_np(logits)
                    pred_idx = int(np.argmax(probs))
                    conf = float(probs[pred_idx])

                    current_pred = labels[pred_idx]
                    current_conf = conf

                    # ==================================================
                    # SIMPLE CONFIDENCE-BASED COMMIT (like the version that worked)
                    # ==================================================

                    is_background = (pred_idx == bg_idx)
                    is_new_sign = current_pred != committed_label

                    # Only track non-background predictions with high confidence
                    if not is_background and conf >= COMMIT_THRESHOLD and is_new_sign:
                        if pred_idx == last_pred_idx:
                            stable_count += 1
                        else:
                            stable_count = 1
                            last_pred_idx = pred_idx

                        # Commit when stable
                        if stable_count >= STABLE_FRAMES:
                            gesture_count += 1
                            committed_caption = format_label(current_pred)
                            committed_conf = conf
                            committed_label = current_pred
                            hold_counter = HOLD_FRAMES
                            stable_count = 0
                            last_pred_idx = -1
                            print(f"\n[{gesture_count}] {committed_caption} ({conf:.0%})")
                    else:
                        # Reset stability if prediction changed or is background
                        if pred_idx != last_pred_idx:
                            stable_count = 0
                            last_pred_idx = pred_idx

                # ==================================================
                # DISPLAY
                # ==================================================
                # Status - show current prediction and stability
                status = f"L={left} R={right}"

                if current_pred != "background":
                    status += f" | {format_label(current_pred)} ({current_conf:.0%})"
                    if stable_count > 0:
                        status += f" [stable: {stable_count}/{STABLE_FRAMES}]"
                else:
                    status += " | ready"

                draw_status(frame, status)

                # Caption
                if committed_caption:
                    draw_caption(frame, committed_caption, committed_conf)

                # Border color
                if committed_caption and hold_counter > 0:
                    color = (0, 255, 0)      # Green - showing result
                elif stable_count > 0:
                    color = (0, 255, 255)    # Yellow - building stability
                elif hands_detected and current_pred != "background" and current_conf >= 0.7:
                    color = (255, 200, 0)    # Blue - detecting sign
                elif hands_detected:
                    color = (150, 150, 150)  # Gray - ready
                else:
                    color = (80, 80, 80)     # Dark - no hands

                cv2.rectangle(frame, (0, 0), (actual_w-1, actual_h-1), color, 6)

                # Confidence bar
                if hands_detected and len(window) >= T:
                    bar_x = actual_w // 2 - 150
                    bar_y = 50
                    bar_width = 300
                    bar_height = 25

                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

                    fill = int(bar_width * current_conf)
                    if current_pred == "background":
                        bar_color = (128, 128, 128)
                    elif current_conf >= COMMIT_THRESHOLD:
                        bar_color = (0, 255, 0)
                    else:
                        bar_color = (0, 165, 255)

                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + fill, bar_y + bar_height), bar_color, -1)

                    # Threshold marker (91%)
                    commit_x = bar_x + int(bar_width * COMMIT_THRESHOLD)
                    cv2.line(frame, (commit_x, bar_y - 5), (commit_x, bar_y + bar_height + 5), (255, 255, 255), 2)

                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

                    # Label
                    pred_display = format_label(current_pred) if current_pred != "background" else "ready"
                    cv2.putText(frame, f"{pred_display}: {current_conf:.0%}",
                               (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cam.send(frame)
                cv2.imshow("SignLink Online CSLR", frame)

                # Keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    window.clear()
                    committed_caption = ""
                    committed_conf = None
                    committed_label = ""
                    hold_counter = 0
                    stable_count = 0
                    last_pred_idx = -1

                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nStopped. Recognized {gesture_count} signs.")


if __name__ == "__main__":
    main()
