#!/usr/bin/env python3
"""
SignLink Virtual Camera - Combined Velocity + Confidence Detection

Shows subtitle when BOTH conditions are met:
1. Confidence ≥ 90% (stable for N frames)
2. Velocity drops below threshold (hands slowing down = sign ending)

This combines the best of both approaches for more accurate detection.
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


def compute_hand_velocity(feat_curr, feat_prev, left_present, right_present):
    """Compute velocity of hand landmarks between frames."""
    if feat_prev is None:
        return 0.0

    velocity = 0.0
    n_hands = 0

    # Left hand velocity
    if left_present > 0:
        left_vel = np.linalg.norm(feat_curr[0:63] - feat_prev[0:63])
        velocity += left_vel
        n_hands += 1

    # Right hand velocity
    if right_present > 0:
        right_vel = np.linalg.norm(feat_curr[63:126] - feat_prev[63:126])
        velocity += right_vel
        n_hands += 1

    if n_hands > 0:
        velocity /= n_hands

    return velocity


def main():
    print("=" * 70)
    print("SIGNLINK - COMBINED VELOCITY + CONFIDENCE DETECTION")
    print("=" * 70)

    # ==========================================================
    # SETTINGS
    # ==========================================================
    T = 30                          # Window size (frames)

    # Confidence settings
    CONFIDENCE_THRESHOLD = 0.90     # Must be this confident
    STABLE_FRAMES = 4               # Same prediction for N frames

    # Velocity settings (based on your data: still ~0.005-0.01, moving ~0.02-0.18)
    VEL_LOW = 0.015                 # Below this = hands slowing/stopped

    HOLD_FRAMES = 60                # Show subtitle for 2 seconds

    camera_index = 0
    width, height = 1280, 720

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
    # CAMERA
    # ==========================================================
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

    # ==========================================================
    # STATE
    # ==========================================================
    window = deque(maxlen=T)
    velocity_history = deque(maxlen=5)  # Smooth velocity

    # Prediction tracking
    last_pred = -1
    stable_count = 0
    high_conf_pred = None           # The prediction when confidence was high
    high_conf_value = 0.0

    # Velocity tracking
    prev_feat = None

    # Display state
    committed_caption = ""
    committed_conf = None
    committed_label = ""
    hold_counter = 0

    # Stats
    gesture_count = 0

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
    print("COMBINED DETECTION (Velocity + Confidence):")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  Velocity threshold:   {VEL_LOW} (hands slowing)")
    print(f"  Stable frames needed: {STABLE_FRAMES}")
    print()
    print("HOW IT WORKS:")
    print("  1. Detection runs continuously")
    print("  2. Tracks when confidence reaches 90%+")
    print("  3. Shows subtitle when hands SLOW DOWN after high confidence")
    print("  4. Immediately starts detecting next sign")
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

                # ==================================================
                # COMPUTE VELOCITY
                # ==================================================
                velocity = compute_hand_velocity(
                    feat, prev_feat,
                    info["left_present"],
                    info["right_present"]
                )
                velocity_history.append(velocity)
                smooth_velocity = np.mean(velocity_history) if velocity_history else 0

                prev_feat = feat.copy() if hands_detected else None

                # ==================================================
                # HOLD COUNTER
                # ==================================================
                if hold_counter > 0:
                    hold_counter -= 1
                    if hold_counter == 0:
                        committed_caption = ""
                        committed_conf = None
                        committed_label = ""

                # ==================================================
                # INFERENCE
                # ==================================================
                current_pred = ""
                current_conf = 0.0
                ready_to_commit = False

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

                    is_new_sign = current_pred != committed_label

                    # Track stability at high confidence
                    if pred_idx == last_pred and conf >= CONFIDENCE_THRESHOLD and is_new_sign:
                        stable_count += 1
                        # Remember this high-confidence prediction
                        high_conf_pred = current_pred
                        high_conf_value = conf
                    elif pred_idx != last_pred:
                        stable_count = 0
                        last_pred = pred_idx

                    # Check if ready to commit:
                    # - Had stable high confidence prediction
                    # - Velocity has dropped (hands slowing = sign ending)
                    if (stable_count >= STABLE_FRAMES and
                        high_conf_pred is not None and
                        smooth_velocity < VEL_LOW and
                        high_conf_pred != committed_label):
                        ready_to_commit = True

                    # COMMIT
                    if ready_to_commit:
                        gesture_count += 1
                        committed_caption = format_label(high_conf_pred)
                        committed_conf = high_conf_value
                        committed_label = high_conf_pred
                        hold_counter = HOLD_FRAMES
                        stable_count = 0
                        high_conf_pred = None
                        high_conf_value = 0.0
                        last_pred = -1
                        print(f"\n[{gesture_count}] Committed: {committed_caption} ({committed_conf:.0%}) | vel={smooth_velocity:.4f}")

                # ==================================================
                # DISPLAY
                # ==================================================
                # Status line
                status = f"L={left} R={right} | vel={smooth_velocity:.3f}"

                if hands_detected and len(window) >= T:
                    status += f" | {format_label(current_pred)} ({current_conf:.0%})"
                    if stable_count > 0:
                        status += f" [ready:{stable_count}]"
                    if smooth_velocity < VEL_LOW and high_conf_pred:
                        status += " SLOW!"

                draw_status(frame, status)

                # Caption
                if committed_caption:
                    draw_caption(frame, committed_caption, committed_conf)

                # Border color
                if committed_caption:
                    color = (0, 255, 0)      # Green - showing result
                elif hands_detected:
                    if stable_count >= STABLE_FRAMES and smooth_velocity < VEL_LOW:
                        color = (0, 255, 255)    # Yellow - about to commit
                    elif stable_count > 0:
                        color = (255, 200, 0)    # Blue - building confidence
                    else:
                        color = (200, 150, 0)    # Darker blue - detecting
                else:
                    color = (128, 128, 128)  # Gray - waiting

                cv2.rectangle(frame, (0, 0), (actual_w-1, actual_h-1), color, 6)

                # Confidence + Velocity bars
                if hands_detected and current_conf > 0:
                    # Confidence bar (top)
                    bar_x = actual_w // 2 - 150
                    bar_y = 50
                    bar_width = 300
                    bar_height = 20

                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

                    fill_width = int(bar_width * current_conf)
                    conf_color = (0, 255, 0) if current_conf >= CONFIDENCE_THRESHOLD else (0, 165, 255)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + fill_width, bar_y + bar_height), conf_color, -1)

                    # Threshold marker
                    thresh_x = bar_x + int(bar_width * CONFIDENCE_THRESHOLD)
                    cv2.line(frame, (thresh_x, bar_y - 3), (thresh_x, bar_y + bar_height + 3),
                            (255, 255, 255), 2)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                    cv2.putText(frame, f"Conf: {current_conf:.0%}", (bar_x + bar_width + 10, bar_y + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Velocity bar (below confidence)
                    vel_y = bar_y + 30
                    cv2.rectangle(frame, (bar_x, vel_y),
                                 (bar_x + bar_width, vel_y + bar_height), (50, 50, 50), -1)

                    vel_normalized = min(smooth_velocity / 0.1, 1.0)
                    vel_fill = int(bar_width * vel_normalized)
                    vel_color = (0, 0, 255) if smooth_velocity < VEL_LOW else (0, 255, 0)
                    cv2.rectangle(frame, (bar_x, vel_y),
                                 (bar_x + vel_fill, vel_y + bar_height), vel_color, -1)

                    # Velocity threshold marker
                    vel_thresh_x = bar_x + int(bar_width * (VEL_LOW / 0.1))
                    cv2.line(frame, (vel_thresh_x, vel_y - 3), (vel_thresh_x, vel_y + bar_height + 3),
                            (255, 255, 255), 2)
                    cv2.rectangle(frame, (bar_x, vel_y),
                                 (bar_x + bar_width, vel_y + bar_height), (255, 255, 255), 2)
                    cv2.putText(frame, f"Vel: {smooth_velocity:.3f}", (bar_x + bar_width + 10, vel_y + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Current prediction label
                    pred_text = format_label(current_pred)
                    text_size = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_x = actual_w // 2 - text_size[0] // 2
                    cv2.putText(frame, pred_text, (text_x, bar_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Ready indicator
                    if stable_count >= STABLE_FRAMES and high_conf_pred:
                        ready_text = "HIGH CONF - slow down to commit!"
                        if smooth_velocity < VEL_LOW:
                            ready_text = "COMMITTING..."
                        text_size = cv2.getTextSize(ready_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        text_x = actual_w // 2 - text_size[0] // 2
                        cv2.putText(frame, ready_text, (text_x, vel_y + 45),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cam.send(frame)
                cv2.imshow("SignLink Combined Detection", frame)

                # ==================================================
                # KEY HANDLING
                # ==================================================
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('c'):
                    window.clear()
                    velocity_history.clear()
                    last_pred = -1
                    stable_count = 0
                    high_conf_pred = None
                    high_conf_value = 0.0
                    committed_caption = ""
                    committed_conf = None
                    committed_label = ""
                    hold_counter = 0
                    prev_feat = None

                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nStopped. Recognized {gesture_count} signs.")


if __name__ == "__main__":
    main()
