#!/usr/bin/env python3
"""
SignLink Virtual Camera - Sliding Window with Background Class

This app uses a model trained with an explicit "background" class.
The sliding window runs continuously, and subtitles appear when:
1. Prediction changes from "background" to a sign
2. Sign confidence is high and stable
3. Then immediately ready for next sign

This is the cleanest detection method from the literature.
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
    "background": None,  # Don't display background
}


def format_label(label: str) -> str:
    if label == "background":
        return None
    return DISPLAY_LABELS.get(label, label)


def main():
    print("=" * 70)
    print("SIGNLINK - SLIDING WINDOW WITH BACKGROUND CLASS")
    print("=" * 70)

    # ==========================================================
    # SETTINGS
    # ==========================================================
    T = 30                          # Sliding window size
    STRIDE = 2                      # Run inference every N frames

    SIGN_CONF_THRESHOLD = 0.85      # Confidence needed for sign
    STABLE_PREDICTIONS = 3          # Same prediction for N windows

    HOLD_FRAMES = 60                # Show subtitle for 2 seconds

    camera_index = 0
    width, height = 1280, 720

    # ==========================================================
    # MODEL SETUP (with background class)
    # ==========================================================
    model_path = "models_signlink_pose/pose_gru_with_bg.pt"
    labels_path = "models_signlink_pose/pose_labels_with_bg.json"

    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        print("\nYou need to:")
        print("  1. Run 'python src/record_background.py' to record background samples")
        print("  2. Run 'python src/train_pose_with_background.py' to train the model")
        return

    print("Loading model with background class...")
    from src.features_hands_pose import HandsPoseFeatureExtractor
    extractor = HandsPoseFeatureExtractor()

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

    # Find background index
    bg_idx = labels.index("background") if "background" in labels else -1
    print(f"Labels: {labels}")
    print(f"Background class index: {bg_idx}")

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
    print(f"Model loaded: {len(labels)} classes (including background)")

    # ==========================================================
    # STATE
    # ==========================================================
    window = deque(maxlen=T)
    prediction_history = deque(maxlen=10)  # (label, confidence) tuples

    # Display state
    committed_caption = ""
    committed_conf = None
    committed_label = ""
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
    print("SLIDING WINDOW DETECTION:")
    print(f"  Window size:            {T} frames")
    print(f"  Stride:                 {STRIDE} frames")
    print(f"  Sign confidence:        {SIGN_CONF_THRESHOLD*100:.0f}%")
    print(f"  Stable predictions:     {STABLE_PREDICTIONS}")
    print()
    print("HOW IT WORKS:")
    print("  1. Model predicts 'background' or one of 7 signs")
    print("  2. When sign detected with high confidence → show subtitle")
    print("  3. Immediately continues detecting next sign")
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
                        # Keep committed_label to prevent re-triggering

                # ==================================================
                # SLIDING WINDOW INFERENCE
                # ==================================================
                current_pred = "background"
                current_conf = 0.0
                is_sign = False

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
                    is_sign = (pred_idx != bg_idx)

                    # Add to history
                    prediction_history.append((current_pred, current_conf, is_sign))

                    # ==================================================
                    # CHECK FOR COMMIT
                    # ==================================================
                    if len(prediction_history) >= STABLE_PREDICTIONS:
                        recent = list(prediction_history)[-STABLE_PREDICTIONS:]

                        # Check: all same sign (not background), high confidence
                        all_same = all(p[0] == recent[0][0] for p in recent)
                        all_sign = all(p[2] for p in recent)  # All are signs (not background)
                        avg_conf = np.mean([p[1] for p in recent])
                        is_new = recent[0][0] != committed_label

                        if all_same and all_sign and avg_conf >= SIGN_CONF_THRESHOLD and is_new:
                            gesture_count += 1
                            committed_label = recent[0][0]
                            committed_caption = format_label(committed_label)
                            committed_conf = avg_conf
                            hold_counter = HOLD_FRAMES

                            # Clear history
                            prediction_history.clear()

                            print(f"\n[{gesture_count}] {committed_caption} ({committed_conf:.0%})")

                # ==================================================
                # DISPLAY
                # ==================================================
                # Status line
                status = f"L={left} R={right}"

                if current_pred == "background":
                    status += " | background"
                else:
                    status += f" | {format_label(current_pred)} ({current_conf:.0%})"

                # Show stability progress
                if len(prediction_history) > 0:
                    recent_signs = sum(1 for p in list(prediction_history)[-STABLE_PREDICTIONS:] if p[2])
                    if recent_signs > 0:
                        status += f" [stable: {recent_signs}/{STABLE_PREDICTIONS}]"

                draw_status(frame, status)

                # Caption
                if committed_caption:
                    draw_caption(frame, committed_caption, committed_conf)

                # Border color
                if committed_caption:
                    color = (0, 255, 0)      # Green - showing result
                elif is_sign and current_conf >= SIGN_CONF_THRESHOLD:
                    color = (0, 255, 255)    # Yellow - sign detected, building stability
                elif is_sign:
                    color = (255, 200, 0)    # Blue - sign detected, low confidence
                elif hands_detected:
                    color = (150, 150, 150)  # Gray - background
                else:
                    color = (80, 80, 80)     # Dark gray - no hands

                cv2.rectangle(frame, (0, 0), (actual_w-1, actual_h-1), color, 6)

                # Confidence bar
                if hands_detected and current_conf > 0:
                    bar_x = actual_w // 2 - 150
                    bar_y = 50
                    bar_width = 300
                    bar_height = 25

                    # Background
                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

                    # Fill
                    fill_width = int(bar_width * current_conf)
                    if current_pred == "background":
                        bar_color = (128, 128, 128)  # Gray for background
                    elif current_conf >= SIGN_CONF_THRESHOLD:
                        bar_color = (0, 255, 0)      # Green for high conf sign
                    else:
                        bar_color = (0, 165, 255)    # Orange for low conf

                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + fill_width, bar_y + bar_height), bar_color, -1)

                    # Threshold marker
                    thresh_x = bar_x + int(bar_width * SIGN_CONF_THRESHOLD)
                    cv2.line(frame, (thresh_x, bar_y - 3), (thresh_x, bar_y + bar_height + 3),
                            (255, 255, 255), 2)

                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

                    # Label
                    pred_display = format_label(current_pred) if current_pred != "background" else "background"
                    cv2.putText(frame, f"{pred_display}: {current_conf:.0%}",
                               (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cam.send(frame)
                cv2.imshow("SignLink Sliding Window", frame)

                # ==================================================
                # KEY HANDLING
                # ==================================================
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('c'):
                    window.clear()
                    prediction_history.clear()
                    committed_caption = ""
                    committed_conf = None
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
