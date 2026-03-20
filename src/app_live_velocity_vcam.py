#!/usr/bin/env python3
"""
SignLink Virtual Camera - Velocity-Based Boundary Detection

Based on sign language research (Malaia & Wilbur 2012, Krebs et al. 2025):
- Signs end with rapid deceleration (velocity drops)
- Transition movements have lower velocity than active signing

This version detects sign completion when hand velocity drops below threshold,
allowing hands to stay in frame while showing the translation.
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
    """
    Compute velocity of hand landmarks between frames.

    Features layout:
    - 0-62: Left hand (21 landmarks × 3 coords)
    - 63-125: Right hand (21 landmarks × 3 coords)

    Returns velocity as L2 norm of position change.
    """
    if feat_prev is None:
        return 0.0

    velocity = 0.0
    n_hands = 0

    # Left hand velocity (only if present in both frames)
    if left_present > 0:
        left_curr = feat_curr[0:63]
        left_prev = feat_prev[0:63]
        left_vel = np.linalg.norm(left_curr - left_prev)
        velocity += left_vel
        n_hands += 1

    # Right hand velocity (only if present in both frames)
    if right_present > 0:
        right_curr = feat_curr[63:126]
        right_prev = feat_prev[63:126]
        right_vel = np.linalg.norm(right_curr - right_prev)
        velocity += right_vel
        n_hands += 1

    # Average velocity across detected hands
    if n_hands > 0:
        velocity /= n_hands

    return velocity


def main():
    print("=" * 70)
    print("SIGNLINK - VELOCITY-BASED BOUNDARY DETECTION")
    print("=" * 70)
    print()
    print("Based on research: signs end with rapid deceleration")
    print("Detects when hands slow down to trigger prediction")
    print()

    # ==========================================================
    # VELOCITY DETECTION SETTINGS
    # ==========================================================
    T = 30                          # Window size (frames for inference)

    # Velocity thresholds (in normalized landmark coordinates)
    # These are tuned for MediaPipe's 0-1 normalized coordinates
    # LOWERED thresholds to be more sensitive
    VEL_HIGH = 0.012                # Velocity indicating active signing (lowered)
    VEL_LOW = 0.004                 # Velocity indicating sign completion (lowered)

    # Timing parameters
    MIN_ACTIVE_FRAMES = 8           # Minimum frames of active signing before commit
    MIN_STILL_FRAMES = 8            # Frames below VEL_LOW to confirm completion
    HOLD_N = 60                     # Show result for 2 seconds
    COOLDOWN_FRAMES = 30            # Wait before detecting next sign

    # Confidence threshold
    MIN_CONFIDENCE = 0.5            # Minimum confidence to show prediction

    camera_index = 0
    width, height = 1280, 720

    # ==========================================================
    # MODEL SETUP
    # ==========================================================
    pose_model_path = "models_signlink_pose/pose_gru.pt"
    pose_labels_path = "models_signlink_pose/pose_labels.json"
    hand_model_path = "models/signlink_gru.pt"
    hand_labels_path = "models/labels.json"

    use_pose = os.path.exists(pose_model_path) and os.path.exists(pose_labels_path)

    if use_pose:
        print("Using POSE model (body + hands)")
        from src.features_hands_pose import HandsPoseFeatureExtractor
        extractor = HandsPoseFeatureExtractor()
        input_size = 156
        model_path = pose_model_path
        labels_path = pose_labels_path
    else:
        print("Using HAND-ONLY model")
        from src.features_hands import HandsFeatureExtractor
        extractor = HandsFeatureExtractor()
        input_size = 128
        model_path = hand_model_path
        labels_path = hand_labels_path

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

    if os.path.exists(model_path) and os.path.exists(labels_path):
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
    else:
        model = None
        labels = []
        print(f"WARNING: Model not found")

    # ==========================================================
    # STATE VARIABLES
    # ==========================================================
    window = deque(maxlen=T)        # Sliding window of features
    velocity_history = deque(maxlen=10)  # Smooth velocity readings

    prev_feat = None                # Previous frame features
    active_frames = 0               # Frames of active signing
    still_frames = 0                # Frames of low velocity
    cooldown = 0                    # Cooldown after showing result

    display_caption = ""
    display_conf = None
    hold_frames = 0
    gesture_count = 0

    current_state = "WAITING"       # WAITING, SIGNING, COMPLETING, SHOWING
    debug_frame_count = 0           # For debug output

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
    print("VELOCITY-BASED DETECTION:")
    print(f"  High velocity threshold: {VEL_HIGH}")
    print(f"  Low velocity threshold:  {VEL_LOW}")
    print(f"  Min active frames:       {MIN_ACTIVE_FRAMES}")
    print(f"  Min still frames:        {MIN_STILL_FRAMES}")
    print()
    print("HOW IT WORKS:")
    print("  1. Start signing (hands move = high velocity)")
    print("  2. Finish sign (hands slow down = low velocity)")
    print("  3. Prediction appears automatically")
    print("  4. Hands can stay in frame!")
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

                # Smoothed velocity (average of recent frames)
                smooth_velocity = np.mean(velocity_history) if velocity_history else 0

                prev_feat = feat.copy() if hands_detected else None

                # ==================================================
                # STATE MACHINE
                # ==================================================
                if cooldown > 0:
                    cooldown -= 1

                # Debug: print velocity periodically
                debug_frame_count += 1
                if hands_detected and debug_frame_count % 30 == 0:
                    print(f"  [{current_state}] vel={smooth_velocity:.4f} | thresholds: LOW={VEL_LOW:.4f} HIGH={VEL_HIGH:.4f}")

                if current_state == "WAITING":
                    if hands_detected and smooth_velocity > VEL_HIGH:
                        # Started signing
                        current_state = "SIGNING"
                        active_frames = 1
                        still_frames = 0
                        gesture_count += 1
                        print(f"\nGesture {gesture_count}: signing started (vel={smooth_velocity:.4f})")

                elif current_state == "SIGNING":
                    if not hands_detected:
                        # Hands left - go back to waiting
                        current_state = "WAITING"
                        active_frames = 0
                        print("  Hands left frame, resetting")
                    elif smooth_velocity > VEL_LOW:
                        # Still actively signing
                        active_frames += 1
                        still_frames = 0
                        # Debug: show progress every 10 frames
                        if active_frames % 10 == 0:
                            print(f"  Signing... frames={active_frames}, vel={smooth_velocity:.4f}")
                    else:
                        # Velocity dropped - might be completing
                        still_frames += 1
                        print(f"  Slowing down... still={still_frames}/{MIN_STILL_FRAMES}, vel={smooth_velocity:.4f}")
                        if still_frames >= MIN_STILL_FRAMES and active_frames >= MIN_ACTIVE_FRAMES:
                            current_state = "COMPLETING"
                            print(f"  Sign completing (active={active_frames}, still={still_frames})")

                elif current_state == "COMPLETING":
                    # Run inference
                    if model is not None and len(window) >= T:
                        x = np.stack(list(window)[-T:], axis=0).astype(np.float32)
                        xt = torch.from_numpy(x).unsqueeze(0).to(device)

                        with torch.no_grad():
                            logits = model(xt).cpu().numpy()[0]

                        probs = softmax_np(logits)
                        pred = int(np.argmax(probs))
                        conf = float(probs[pred])

                        if conf >= MIN_CONFIDENCE:
                            display_caption = format_label(labels[pred])
                            display_conf = conf
                            hold_frames = HOLD_N
                            print(f"  Result: '{display_caption}' ({conf:.2f})")
                        else:
                            print(f"  Low confidence ({conf:.2f}), skipping")

                    current_state = "SHOWING"
                    cooldown = COOLDOWN_FRAMES
                    active_frames = 0
                    still_frames = 0

                elif current_state == "SHOWING":
                    if hold_frames > 0:
                        hold_frames -= 1

                    # Can start new sign after cooldown
                    if cooldown == 0 and hands_detected and smooth_velocity > VEL_HIGH:
                        current_state = "SIGNING"
                        active_frames = 1
                        still_frames = 0
                        gesture_count += 1
                        display_caption = ""
                        display_conf = None
                        print(f"\nGesture {gesture_count}: signing started (vel={smooth_velocity:.4f})")
                    elif not hands_detected and hold_frames == 0:
                        current_state = "WAITING"
                        display_caption = ""
                        display_conf = None

                # ==================================================
                # DISPLAY
                # ==================================================
                # Build status line
                status = f"L={left} R={right}"
                if hands_detected:
                    status += f" | vel={smooth_velocity:.4f}"
                status += f" | {current_state}"
                if current_state == "SIGNING":
                    status += f" [{active_frames}]"
                if cooldown > 0:
                    status += f" (cooldown {cooldown})"

                draw_status(frame, status)

                # Show caption
                if display_caption and (hold_frames > 0 or current_state == "SHOWING"):
                    draw_caption(frame, display_caption, display_conf)

                # Border color based on state
                if current_state == "WAITING":
                    color = (128, 128, 128)  # Gray
                elif current_state == "SIGNING":
                    color = (0, 255, 0)      # Green - actively signing
                elif current_state == "COMPLETING":
                    color = (0, 255, 255)    # Yellow - completing
                else:  # SHOWING
                    color = (255, 0, 0)      # Blue - showing result

                cv2.rectangle(frame, (0, 0), (actual_w-1, actual_h-1), color, 6)

                # Velocity indicator bar
                if hands_detected:
                    bar_x = 20
                    bar_y = actual_h - 60
                    bar_max_width = 200

                    # Background
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + 20), (50, 50, 50), -1)

                    # Velocity bar (scaled to 0-0.05 range)
                    vel_normalized = min(smooth_velocity / 0.05, 1.0)
                    bar_width = int(bar_max_width * vel_normalized)

                    # Color based on threshold
                    if smooth_velocity > VEL_HIGH:
                        bar_color = (0, 255, 0)  # Green - active
                    elif smooth_velocity > VEL_LOW:
                        bar_color = (0, 255, 255)  # Yellow - medium
                    else:
                        bar_color = (0, 0, 255)  # Red - still

                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), bar_color, -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + 20), (255, 255, 255), 2)

                    # Threshold markers
                    low_mark = int(bar_max_width * (VEL_LOW / 0.05))
                    high_mark = int(bar_max_width * (VEL_HIGH / 0.05))
                    cv2.line(frame, (bar_x + low_mark, bar_y), (bar_x + low_mark, bar_y + 20), (0, 0, 255), 2)
                    cv2.line(frame, (bar_x + high_mark, bar_y), (bar_x + high_mark, bar_y + 20), (0, 255, 0), 2)

                    cv2.putText(frame, "Velocity", (bar_x, bar_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cam.send(frame)
                cv2.imshow("SignLink Velocity Detection", frame)

                # ==================================================
                # KEY HANDLING
                # ==================================================
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('c'):
                    window.clear()
                    velocity_history.clear()
                    current_state = "WAITING"
                    display_caption = ""
                    display_conf = None
                    hold_frames = 0
                    active_frames = 0
                    still_frames = 0
                    cooldown = 0
                    prev_feat = None

                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nStopped")


if __name__ == "__main__":
    main()
