#!/usr/bin/env python3
"""
SignLink Virtual Camera - 90% Confidence Trigger

Upgraded version of app_live_body_hands_vcam.py:
- Shows subtitle when confidence reaches 90% (not when hands exit)
- Uses pose features (body + hands) for better accuracy
- Continuous detection - no need to remove hands between signs
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


# Display labels with proper formatting
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
    """Convert internal label to display format."""
    return DISPLAY_LABELS.get(label, label)


def main():
    print("=" * 70)
    print("SIGNLINK - 90% CONFIDENCE TRIGGER")
    print("=" * 70)

    # ==========================================================
    # SETTINGS
    # ==========================================================
    T = 30                          # Window size: 30 frames
    CONFIDENCE_THRESHOLD = 0.90     # Show subtitle at 90%+
    STABLE_FRAMES = 3               # Need 3 stable predictions to commit
    HOLD_FRAMES = 60                # Show subtitle for ~2 seconds

    # Camera settings
    camera_index = 0
    width = 1280
    height = 720

    # ==========================================================
    # MODEL SETUP
    # ==========================================================
    pose_model_path = "models_signlink_pose/pose_gru.pt"
    pose_labels_path = "models_signlink_pose/pose_labels.json"

    if not os.path.exists(pose_model_path):
        print(f"ERROR: Model not found at {pose_model_path}")
        return

    print("Using POSE model (body + hands)")
    from src.features_hands_pose import HandsPoseFeatureExtractor
    extractor = HandsPoseFeatureExtractor()
    input_size = 156

    # ==========================================================
    # CAMERA SETUP
    # ==========================================================
    print(f"\nOpening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    print(f"Camera: {actual_width}x{actual_height} @ {fps}fps")

    # ==========================================================
    # LOAD MODEL
    # ==========================================================
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    with open(pose_labels_path, "r") as f:
        labels = json.load(f)

    model = SignGRU(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=len(labels),
        dropout=0.2
    )
    model.load_state_dict(torch.load(pose_model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded: {len(labels)} signs")
    print(f"Labels: {labels}")

    # ==========================================================
    # STATE VARIABLES
    # ==========================================================
    window = deque(maxlen=T)

    # Tracking for stable predictions
    last_pred_idx = -1
    stable_count = 0
    committed_label = ""  # Track what was last shown to avoid re-triggering

    # Display state
    display_caption = ""
    display_conf = None
    hold_counter = 0

    # Current prediction (for display)
    current_pred = ""
    current_conf = 0.0

    gesture_count = 0
    frame_i = 0

    # ==========================================================
    # VIRTUAL CAMERA SETUP
    # ==========================================================
    print("\nInitializing virtual camera...")

    try:
        try:
            cam = pyvirtualcam.Camera(
                width=actual_width,
                height=actual_height,
                fps=fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
                backend=pyvirtualcam.Backend.OBS
            )
        except (AttributeError, ValueError):
            cam = pyvirtualcam.Camera(
                width=actual_width,
                height=actual_height,
                fps=fps,
                fmt=pyvirtualcam.PixelFormat.BGR
            )

        print(f"Virtual camera ready: {cam.device}")
        print()
        print("=" * 70)
        print("SETTINGS:")
        print(f"  Window: {T} frames")
        print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
        print(f"  Stable frames needed: {STABLE_FRAMES}")
        print()
        print("HOW IT WORKS:")
        print("  - Subtitle appears when confidence >= 90%")
        print("  - No need to remove hands between signs")
        print("  - Continuous detection")
        print("=" * 70)
        print("\nControls: Q=quit, C=clear")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"ERROR: Virtual camera failed: {e}")
        cap.release()
        return

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

                feat, info = extractor.extract(rgb)
                window.append(feat)

                left = int(info["left_present"])
                right = int(info["right_present"])
                hands_detected = left > 0 or right > 0

                pose_status = ""
                if "pose_present" in info:
                    pose_status = f" P={int(info['pose_present'])}"

                frame_i += 1

                # ==================================================
                # HOLD COUNTER (subtitle display)
                # ==================================================
                if hold_counter > 0:
                    hold_counter -= 1
                    if hold_counter == 0:
                        display_caption = ""
                        display_conf = None
                        committed_label = ""  # Allow same sign to be detected again

                # ==================================================
                # INFERENCE
                # ==================================================
                current_pred = ""
                current_conf = 0.0

                if hands_detected and len(window) >= T:
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
                    # 90% CONFIDENCE TRIGGER
                    # ==================================================
                    is_new_sign = current_pred != committed_label

                    if conf >= CONFIDENCE_THRESHOLD and is_new_sign:
                        # Check stability
                        if pred_idx == last_pred_idx:
                            stable_count += 1
                        else:
                            stable_count = 1
                            last_pred_idx = pred_idx

                        # Commit when stable enough
                        if stable_count >= STABLE_FRAMES:
                            gesture_count += 1
                            display_caption = format_label(current_pred)
                            display_conf = conf
                            committed_label = current_pred
                            hold_counter = HOLD_FRAMES
                            stable_count = 0
                            last_pred_idx = -1
                            print(f"\n[{gesture_count}] {display_caption} ({conf:.0%})")
                    else:
                        # Reset stability if prediction changed
                        if pred_idx != last_pred_idx:
                            stable_count = 0
                            last_pred_idx = pred_idx

                # ==================================================
                # DISPLAY
                # ==================================================
                status = f"L={left} R={right}{pose_status}"

                if hands_detected and len(window) >= T:
                    status += f" | {format_label(current_pred)} ({current_conf:.0%})"
                    if stable_count > 0:
                        status += f" [stable: {stable_count}/{STABLE_FRAMES}]"

                draw_status(frame, status)

                # Show committed caption
                if display_caption:
                    draw_caption(frame, display_caption, display_conf)

                # Border color
                h, w = frame.shape[:2]
                if display_caption and hold_counter > 0:
                    color = (0, 255, 0)      # Green - showing result
                elif hands_detected and current_conf >= CONFIDENCE_THRESHOLD:
                    color = (0, 255, 255)    # Yellow - high confidence
                elif hands_detected:
                    color = (200, 150, 0)    # Blue - detecting
                else:
                    color = (128, 128, 128)  # Gray - waiting

                cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 6)

                # Confidence bar
                if hands_detected and current_conf > 0:
                    bar_x = w // 2 - 150
                    bar_y = 50
                    bar_width = 300
                    bar_height = 25

                    # Background
                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

                    # Fill
                    fill_width = int(bar_width * current_conf)
                    bar_color = (0, 255, 0) if current_conf >= CONFIDENCE_THRESHOLD else (0, 165, 255)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + fill_width, bar_y + bar_height), bar_color, -1)

                    # Threshold marker (90%)
                    thresh_x = bar_x + int(bar_width * CONFIDENCE_THRESHOLD)
                    cv2.line(frame, (thresh_x, bar_y - 5), (thresh_x, bar_y + bar_height + 5),
                            (255, 255, 255), 2)

                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

                    # Current prediction above bar
                    pred_text = format_label(current_pred)
                    cv2.putText(frame, f"{pred_text}: {current_conf:.0%}",
                               (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cam.send(frame)
                cv2.imshow("SignLink 90% Confidence", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    window.clear()
                    last_pred_idx = -1
                    stable_count = 0
                    display_caption = ""
                    display_conf = None
                    committed_label = ""
                    hold_counter = 0

                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nStopped. Recognized {gesture_count} signs.")


if __name__ == "__main__":
    main()
