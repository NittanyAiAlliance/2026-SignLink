#!/usr/bin/env python3
"""
SignLink Virtual Camera - Confidence-Based Detection (Optimized)

Shows subtitle when:
- Confidence reaches 91%+
- Hands are actually moving (not completely still)

No pause between signs - continuous detection.
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

    if left_present > 0:
        left_vel = np.linalg.norm(feat_curr[0:63] - feat_prev[0:63])
        velocity += left_vel
        n_hands += 1

    if right_present > 0:
        right_vel = np.linalg.norm(feat_curr[63:126] - feat_prev[63:126])
        velocity += right_vel
        n_hands += 1

    if n_hands > 0:
        velocity /= n_hands

    return velocity


def main():
    print("=" * 70)
    print("SIGNLINK - CONFIDENCE-BASED DETECTION")
    print("=" * 70)

    # ==========================================================
    # SETTINGS
    # ==========================================================
    T = 30                          # Window size (frames)

    CONFIDENCE_THRESHOLD = 0.91     # Show subtitle at 91%+
    STABLE_FRAMES = 3               # Same prediction for 3 frames to commit
    HOLD_FRAMES = 60                # Show subtitle for 2 seconds

    # Velocity threshold - ignore predictions when hands are too still
    MIN_VELOCITY = 0.3              # Must have significant movement to count as signing

    camera_index = 0
    width, height = 1280, 720

    # ==========================================================
    # MODEL SETUP (original 7-class model)
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
    velocity_history = deque(maxlen=10)

    # Prediction tracking
    last_pred = -1
    stable_count = 0

    # Velocity tracking
    prev_feat = None
    had_movement = False  # Track if there was movement recently

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
    print("SETTINGS:")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  Stable frames:        {STABLE_FRAMES}")
    print(f"  Subtitle duration:    {HOLD_FRAMES} frames (~2s)")
    print(f"  Min velocity:         {MIN_VELOCITY} (ignores still hands)")
    print()
    print("HOW IT WORKS:")
    print("  1. Sign reaches 91%+ confidence → subtitle appears")
    print("  2. NO PAUSE - immediately detecting next sign")
    print("  3. Ignores predictions when hands are completely still")
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
                # COMPUTE VELOCITY
                # ==================================================
                velocity = compute_hand_velocity(
                    feat, prev_feat,
                    info["left_present"],
                    info["right_present"]
                )
                velocity_history.append(velocity)
                avg_velocity = np.mean(velocity_history) if len(velocity_history) > 0 else 0

                prev_feat = feat.copy() if hands_detected else None

                # Track if there's been movement
                if avg_velocity > MIN_VELOCITY:
                    had_movement = True

                # ==================================================
                # HOLD COUNTER (subtitle display)
                # ==================================================
                if hold_counter > 0:
                    hold_counter -= 1
                    if hold_counter == 0:
                        committed_caption = ""
                        committed_conf = None
                        # Don't clear committed_label - prevents re-trigger

                # ==================================================
                # INFERENCE (always running)
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

                    # Only count predictions if:
                    # 1. Confidence is high enough
                    # 2. There was movement (not just still hands)
                    # 3. It's a different sign than currently shown
                    is_new_sign = current_pred != committed_label
                    is_moving = had_movement and avg_velocity > MIN_VELOCITY * 0.5

                    if conf >= CONFIDENCE_THRESHOLD and is_new_sign and is_moving:
                        if pred_idx == last_pred:
                            stable_count += 1
                        else:
                            stable_count = 1
                            last_pred = pred_idx

                        # Commit immediately when stable enough
                        if stable_count >= STABLE_FRAMES:
                            gesture_count += 1
                            committed_caption = format_label(current_pred)
                            committed_conf = conf
                            committed_label = current_pred
                            hold_counter = HOLD_FRAMES
                            stable_count = 0
                            last_pred = -1
                            had_movement = False  # Reset movement tracking
                            print(f"\n[{gesture_count}] {committed_caption} ({conf:.0%})")
                    else:
                        # Reset stability if conditions not met
                        if pred_idx != last_pred:
                            stable_count = 0
                            last_pred = pred_idx

                # ==================================================
                # DISPLAY
                # ==================================================
                # Status line
                status = f"L={left} R={right} | vel={avg_velocity:.3f}"

                if hands_detected and len(window) >= T:
                    status += f" | {format_label(current_pred)} ({current_conf:.0%})"
                    if stable_count > 0:
                        status += f" [stable: {stable_count}/{STABLE_FRAMES}]"

                draw_status(frame, status)

                # Caption - show committed caption
                if committed_caption:
                    draw_caption(frame, committed_caption, committed_conf)

                # Border color
                if committed_caption and hold_counter > 0:
                    color = (0, 255, 0)      # Green - showing result
                elif hands_detected and current_conf >= CONFIDENCE_THRESHOLD:
                    if stable_count > 0:
                        color = (0, 255, 255)    # Yellow - building stability
                    else:
                        color = (255, 200, 0)    # Blue - high conf but not stable
                elif hands_detected:
                    color = (200, 150, 0)    # Darker blue - detecting
                else:
                    color = (128, 128, 128)  # Gray - waiting

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
                    if current_conf >= CONFIDENCE_THRESHOLD:
                        bar_color = (0, 255, 0)  # Green
                    else:
                        bar_color = (0, 165, 255)  # Orange

                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + fill_width, bar_y + bar_height), bar_color, -1)

                    # Threshold marker (91%)
                    thresh_x = bar_x + int(bar_width * CONFIDENCE_THRESHOLD)
                    cv2.line(frame, (thresh_x, bar_y - 5), (thresh_x, bar_y + bar_height + 5),
                            (255, 255, 255), 2)

                    cv2.rectangle(frame, (bar_x, bar_y),
                                 (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

                    # Percentage
                    cv2.putText(frame, f"{current_conf:.0%}", (bar_x + bar_width + 10, bar_y + 18),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Current prediction above bar
                    pred_text = format_label(current_pred)
                    text_size = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_x = actual_w // 2 - text_size[0] // 2
                    cv2.putText(frame, pred_text, (text_x, bar_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Movement indicator
                    if avg_velocity < MIN_VELOCITY:
                        cv2.putText(frame, "(hands still - waiting for movement)",
                                   (bar_x, bar_y + bar_height + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

                cam.send(frame)
                cv2.imshow("SignLink Confidence Detection", frame)

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
                    committed_caption = ""
                    committed_conf = None
                    committed_label = ""
                    hold_counter = 0
                    had_movement = False
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
