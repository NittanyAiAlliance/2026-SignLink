#!/usr/bin/env python3
"""
SignLink Virtual Camera - Hybrid Detection (Velocity Boundaries + Sliding Window)

Based on sign language recognition literature:
1. Compute hand velocity every frame
2. Use velocity dips as soft boundary candidates
3. Run sliding window with GRU
4. Fuse signals: prediction + velocity pattern (deceleration-then-acceleration)
5. Emit subtitle when:
   - Prediction changes from "background" (low conf) to sign (high conf)
   - Velocity shows sign-ending pattern
   - Prediction stable for N windows

Note: Uses low confidence (<70%) as proxy for "background" class.
For best results, retrain model with actual background samples.
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


def detect_velocity_pattern(velocity_history, window_size=10):
    """
    Detect velocity patterns that indicate sign boundaries.

    Returns:
        pattern: 'decel' (slowing down), 'accel' (speeding up),
                 'stable_high', 'stable_low', or 'transition'
        trend: numerical trend value (negative = decelerating)
    """
    if len(velocity_history) < window_size:
        return 'unknown', 0.0

    recent = list(velocity_history)[-window_size:]

    # Split into first half and second half
    first_half = np.mean(recent[:window_size//2])
    second_half = np.mean(recent[window_size//2:])

    trend = second_half - first_half  # Negative = decelerating

    avg_vel = np.mean(recent)

    if avg_vel < 0.01:
        return 'stable_low', trend
    elif abs(trend) < 0.005:
        return 'stable_high', trend
    elif trend < -0.005:
        return 'decel', trend  # Slowing down = sign ending
    else:
        return 'accel', trend  # Speeding up = sign starting


def main():
    print("=" * 70)
    print("SIGNLINK - HYBRID DETECTION")
    print("(Velocity Boundaries + Sliding Window)")
    print("=" * 70)

    # ==========================================================
    # SETTINGS
    # ==========================================================
    T = 30                          # Sliding window size
    STRIDE = 2                      # Run inference every N frames

    # Confidence thresholds
    SIGN_CONF_THRESHOLD = 0.88      # High confidence = sign detected
    BACKGROUND_CONF_THRESHOLD = 0.70  # Below this = "background" (no clear sign)

    # Stability requirements
    STABLE_WINDOWS = 3              # Same prediction for N consecutive windows

    # Velocity thresholds
    VEL_DECEL_THRESHOLD = -0.003    # Trend below this = decelerating

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
    velocity_history = deque(maxlen=20)

    # Prediction history (for stability check)
    prediction_history = deque(maxlen=10)  # (label, confidence) tuples

    # State tracking
    prev_feat = None
    in_background = True  # Start in background state

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
    print("HYBRID DETECTION (Literature-Based):")
    print(f"  Sign confidence threshold:  {SIGN_CONF_THRESHOLD*100:.0f}%")
    print(f"  Background threshold:       <{BACKGROUND_CONF_THRESHOLD*100:.0f}%")
    print(f"  Stable windows needed:      {STABLE_WINDOWS}")
    print(f"  Sliding window:             {T} frames, stride {STRIDE}")
    print()
    print("HOW IT WORKS:")
    print("  1. Sliding window runs continuously")
    print("  2. Detects 'background' (low conf) vs 'sign' (high conf)")
    print("  3. Monitors velocity pattern (decel = sign ending)")
    print("  4. Commits when: background→sign + deceleration + stable")
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
                smooth_velocity = np.mean(list(velocity_history)[-5:]) if velocity_history else 0

                # Detect velocity pattern
                vel_pattern, vel_trend = detect_velocity_pattern(velocity_history)

                prev_feat = feat.copy() if hands_detected else None

                # ==================================================
                # HOLD COUNTER
                # ==================================================
                if hold_counter > 0:
                    hold_counter -= 1
                    if hold_counter == 0:
                        committed_caption = ""
                        committed_conf = None
                        # Don't clear committed_label - prevents re-triggering same sign

                # ==================================================
                # SLIDING WINDOW INFERENCE
                # ==================================================
                current_pred = ""
                current_conf = 0.0
                current_state = "background"

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

                    # Classify as background or sign
                    if conf < BACKGROUND_CONF_THRESHOLD:
                        current_state = "background"
                    elif conf >= SIGN_CONF_THRESHOLD:
                        current_state = "sign"
                    else:
                        current_state = "uncertain"

                    # Add to prediction history
                    prediction_history.append((current_pred, current_conf, current_state))

                    # ==================================================
                    # FUSION: Check commit conditions
                    # ==================================================
                    should_commit = False

                    if len(prediction_history) >= STABLE_WINDOWS:
                        recent_preds = list(prediction_history)[-STABLE_WINDOWS:]

                        # Check stability: same sign, all high confidence
                        all_same = all(p[0] == recent_preds[0][0] for p in recent_preds)
                        all_sign_state = all(p[2] == "sign" for p in recent_preds)
                        avg_conf = np.mean([p[1] for p in recent_preds])

                        # Check if this is a NEW sign (different from what's shown)
                        is_new_sign = recent_preds[0][0] != committed_label

                        # Check velocity pattern: should be decelerating (sign ending)
                        # OR stable_low (sign already ended, hands stopped)
                        velocity_good = vel_pattern in ['decel', 'stable_low']

                        # FUSION: Commit if all conditions met
                        if (all_same and
                            all_sign_state and
                            is_new_sign and
                            avg_conf >= SIGN_CONF_THRESHOLD and
                            (velocity_good or in_background)):  # Allow if coming from background
                            should_commit = True

                        # Update background state
                        if current_state == "background":
                            in_background = True
                        elif current_state == "sign" and velocity_good:
                            in_background = False

                    # ==================================================
                    # COMMIT
                    # ==================================================
                    if should_commit:
                        gesture_count += 1
                        committed_caption = format_label(current_pred)
                        committed_conf = current_conf
                        committed_label = current_pred
                        hold_counter = HOLD_FRAMES
                        in_background = False

                        # Clear history to prevent re-triggering
                        prediction_history.clear()

                        print(f"\n[{gesture_count}] {committed_caption} ({committed_conf:.0%})")
                        print(f"    vel_pattern={vel_pattern}, trend={vel_trend:.4f}")

                # ==================================================
                # DISPLAY
                # ==================================================
                # Status line
                status = f"L={left} R={right} | vel={smooth_velocity:.3f} | {vel_pattern}"

                if current_pred:
                    status += f" | {format_label(current_pred)} ({current_conf:.0%}) [{current_state}]"

                draw_status(frame, status)

                # Caption
                if committed_caption:
                    draw_caption(frame, committed_caption, committed_conf)

                # Border color based on state
                if committed_caption:
                    color = (0, 255, 0)      # Green - showing result
                elif current_state == "sign":
                    color = (0, 255, 255)    # Yellow - sign detected
                elif current_state == "uncertain":
                    color = (255, 200, 0)    # Blue - uncertain
                elif hands_detected:
                    color = (200, 150, 0)    # Dark blue - background/detecting
                else:
                    color = (128, 128, 128)  # Gray - no hands

                cv2.rectangle(frame, (0, 0), (actual_w-1, actual_h-1), color, 6)

                # Visual indicators
                if hands_detected:
                    bar_x = actual_w // 2 - 150
                    bar_width = 300
                    bar_height = 18

                    # Confidence bar
                    conf_y = 45
                    cv2.rectangle(frame, (bar_x, conf_y),
                                 (bar_x + bar_width, conf_y + bar_height), (50, 50, 50), -1)

                    if current_conf > 0:
                        fill = int(bar_width * current_conf)
                        if current_state == "sign":
                            c_color = (0, 255, 0)
                        elif current_state == "uncertain":
                            c_color = (0, 255, 255)
                        else:
                            c_color = (0, 165, 255)
                        cv2.rectangle(frame, (bar_x, conf_y),
                                     (bar_x + fill, conf_y + bar_height), c_color, -1)

                    # Threshold markers
                    bg_x = bar_x + int(bar_width * BACKGROUND_CONF_THRESHOLD)
                    sign_x = bar_x + int(bar_width * SIGN_CONF_THRESHOLD)
                    cv2.line(frame, (bg_x, conf_y), (bg_x, conf_y + bar_height), (0, 165, 255), 2)
                    cv2.line(frame, (sign_x, conf_y), (sign_x, conf_y + bar_height), (0, 255, 0), 2)
                    cv2.rectangle(frame, (bar_x, conf_y),
                                 (bar_x + bar_width, conf_y + bar_height), (255, 255, 255), 1)
                    cv2.putText(frame, f"Conf: {current_conf:.0%}", (bar_x + bar_width + 10, conf_y + 14),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Velocity bar
                    vel_y = conf_y + 25
                    cv2.rectangle(frame, (bar_x, vel_y),
                                 (bar_x + bar_width, vel_y + bar_height), (50, 50, 50), -1)

                    vel_norm = min(smooth_velocity / 0.08, 1.0)
                    vel_fill = int(bar_width * vel_norm)

                    if vel_pattern == 'decel':
                        v_color = (0, 0, 255)    # Red - decelerating
                    elif vel_pattern == 'stable_low':
                        v_color = (255, 0, 255)  # Purple - stopped
                    elif vel_pattern == 'accel':
                        v_color = (0, 255, 0)    # Green - accelerating
                    else:
                        v_color = (0, 255, 255)  # Yellow - other

                    cv2.rectangle(frame, (bar_x, vel_y),
                                 (bar_x + vel_fill, vel_y + bar_height), v_color, -1)
                    cv2.rectangle(frame, (bar_x, vel_y),
                                 (bar_x + bar_width, vel_y + bar_height), (255, 255, 255), 1)
                    cv2.putText(frame, f"Vel: {vel_pattern}", (bar_x + bar_width + 10, vel_y + 14),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Current prediction
                    if current_pred:
                        pred_text = format_label(current_pred)
                        text_size = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        text_x = actual_w // 2 - text_size[0] // 2
                        cv2.putText(frame, pred_text, (text_x, conf_y - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cam.send(frame)
                cv2.imshow("SignLink Hybrid Detection", frame)

                # ==================================================
                # KEY HANDLING
                # ==================================================
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('c'):
                    window.clear()
                    velocity_history.clear()
                    prediction_history.clear()
                    committed_caption = ""
                    committed_conf = None
                    committed_label = ""
                    hold_counter = 0
                    in_background = True
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
