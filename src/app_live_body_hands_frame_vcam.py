#!/usr/bin/env python3
"""
SignLink Virtual Camera - Body + Hands Features (In-Frame Detection)

This version:
- Uses pose features (body + hands) for better accuracy
- Shows prediction when hands STOP MOVING (not when they exit)
- Hands can stay in frame - subtitle shows when motion stops
"""

import json
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from collections import deque, Counter

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
    print("SIGNLINK VIRTUAL CAMERA - In-Frame Detection")
    print("=" * 70)

    # ==========================================================
    # TIMING SETTINGS
    # ==========================================================
    T = 30                          # Window size: 30 frames
    STRIDE = 2                      # Inference every 2 frames
    MIN_PREDICTIONS = 4             # Min predictions to commit
    HANDS_STILL_FRAMES = 15         # Frames hands must be still to commit (~0.5 sec)
    MOTION_THRESHOLD = 0.015        # Threshold for detecting motion
    HOLD_N = 50                     # How long to show caption

    # Camera settings
    camera_index = 0
    width = 1280
    height = 720

    # ==========================================================
    # DETERMINE WHICH MODEL TO USE
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
        print("Using HAND-ONLY model (pose model not found)")
        from src.features_hands import HandsFeatureExtractor
        extractor = HandsFeatureExtractor()
        input_size = 128
        model_path = hand_model_path
        labels_path = hand_labels_path

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
        print(f"Labels: {labels}")
    else:
        model = None
        labels = []
        print(f"WARNING: Model not found at {model_path}")

    # ==========================================================
    # STATE VARIABLES
    # ==========================================================
    window = deque(maxlen=T)
    pred_hist = deque(maxlen=15)
    conf_hist = deque(maxlen=15)

    current_state = "WAITING"
    hands_still_frames = 0          # How long hands have been still
    prev_feat = None                # Previous frame features for motion detection
    display_caption = ""
    display_conf = None
    hold_frames = 0
    gesture_count = 0
    debug_line = ""
    frozen = False
    frame_i = 0
    already_committed = False       # Prevent multiple commits for same gesture

    def commit(label: str, conf: float):
        nonlocal display_caption, display_conf, hold_frames, current_state, already_committed
        display_caption = format_label(label)
        display_conf = conf
        hold_frames = HOLD_N
        current_state = "SHOWING"
        already_committed = True
        print(f"SHOWING: '{display_caption}' ({conf:.2f})")

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
        print(f"  Min predictions: {MIN_PREDICTIONS}")
        print(f"  Still threshold: {HANDS_STILL_FRAMES} frames (~{HANDS_STILL_FRAMES/30:.1f}s)")
        print(f"  Motion threshold: {MOTION_THRESHOLD}")
        print()
        print("  Shows prediction when hands STOP MOVING")
        print("  (hands can stay in frame)")
        print("=" * 70)
        print("\nControls: Q=quit, C=clear, SPACE=freeze")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"ERROR: Virtual camera failed: {e}")
        print("Start OBS Virtual Camera first, then run this script")
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

                # Check for pose if available
                pose_status = ""
                if "pose_present" in info:
                    pose_status = f" P={int(info['pose_present'])}"

                hands_state = f"L={left} R={right}{pose_status}"

                # ==================================================
                # MOTION DETECTION
                # ==================================================
                motion = 0.0
                if hands_detected and prev_feat is not None:
                    # Calculate motion as mean absolute difference
                    motion = np.mean(np.abs(feat - prev_feat))

                    if motion < MOTION_THRESHOLD:
                        hands_still_frames += 1
                    else:
                        hands_still_frames = 0
                        already_committed = False  # Reset when motion resumes
                else:
                    hands_still_frames = 0

                prev_feat = feat.copy() if hands_detected else None

                # ==================================================
                # STATE MACHINE (IN-FRAME DETECTION)
                # ==================================================

                if current_state == "WAITING":
                    if hands_detected:
                        current_state = "COLLECTING"
                        gesture_count += 1
                        pred_hist.clear()
                        conf_hist.clear()
                        display_caption = ""
                        display_conf = None
                        hold_frames = 0
                        already_committed = False
                        print(f"\nGesture {gesture_count} started")

                elif current_state == "COLLECTING":
                    # Commit when hands stop moving (still in frame)
                    if hands_detected and hands_still_frames >= HANDS_STILL_FRAMES:
                        if len(pred_hist) >= MIN_PREDICTIONS and not already_committed:
                            vote_counts = Counter(pred_hist)
                            best_idx, _ = vote_counts.most_common(1)[0]
                            confs = [c for p, c in zip(pred_hist, conf_hist) if p == best_idx]
                            avg_conf = sum(confs) / len(confs) if confs else 0
                            commit(labels[best_idx], avg_conf)

                    # If hands leave frame completely, go back to waiting
                    if not hands_detected:
                        current_state = "WAITING"
                        pred_hist.clear()
                        conf_hist.clear()

                elif current_state == "SHOWING":
                    # When hands start moving again, start collecting
                    if hands_detected and motion > MOTION_THRESHOLD:
                        current_state = "COLLECTING"
                        gesture_count += 1
                        pred_hist.clear()
                        conf_hist.clear()
                        display_caption = ""
                        display_conf = None
                        hold_frames = 0
                        already_committed = False
                        print(f"\nGesture {gesture_count} started")

                    # If hands leave, go to waiting
                    if not hands_detected:
                        current_state = "WAITING"

                # ==================================================
                # INFERENCE
                # ==================================================
                if current_state == "COLLECTING" and not frozen and model is not None:
                    if len(window) == T and frame_i % STRIDE == 0:
                        x = np.stack(window, axis=0).astype(np.float32)
                        xt = torch.from_numpy(x).unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits = model(xt).cpu().numpy()[0]
                        probs = softmax_np(logits)
                        pred = int(np.argmax(probs))
                        pred_conf = float(probs[pred])

                        pred_hist.append(pred)
                        conf_hist.append(pred_conf)

                        debug_line = f"{format_label(labels[pred])} ({pred_conf:.2f})"

                # ==================================================
                # DISPLAY
                # ==================================================
                if hold_frames > 0:
                    hold_frames -= 1
                    if hold_frames == 0 and not hands_detected:
                        current_state = "WAITING"
                        display_caption = ""
                        display_conf = None

                caption = display_caption if (hold_frames > 0 or current_state == "SHOWING") else ""
                conf = display_conf if (hold_frames > 0 or current_state == "SHOWING") else None

                if current_state == "WAITING":
                    debug_line = ""

                if model is None:
                    caption = "No model"
                    conf = None

                # Build status line
                status = hands_state
                if frozen:
                    status += " | FROZEN"
                if hands_detected:
                    status += f" | motion={motion:.3f}"
                    if hands_still_frames > 0:
                        status += f" | still={hands_still_frames}"
                if debug_line and current_state == "COLLECTING":
                    status += f" | {debug_line}"
                status += f" | {current_state}"

                draw_status(frame, status)
                draw_caption(frame, caption, conf)

                h, w = frame.shape[:2]
                color = (128, 128, 128) if current_state == "WAITING" else \
                        (0, 255, 0) if current_state == "COLLECTING" else (255, 0, 0)
                cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 6)

                # Show "STILL" indicator when close to committing
                if hands_still_frames > HANDS_STILL_FRAMES // 2:
                    progress = min(hands_still_frames / HANDS_STILL_FRAMES, 1.0)
                    bar_width = int(200 * progress)
                    cv2.rectangle(frame, (w//2 - 100, 60), (w//2 - 100 + bar_width, 80), (0, 255, 255), -1)
                    cv2.rectangle(frame, (w//2 - 100, 60), (w//2 + 100, 80), (255, 255, 255), 2)
                    cv2.putText(frame, "Hold still...", (w//2 - 50, 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cam.send(frame)
                cv2.imshow("SignLink In-Frame Detection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    pred_hist.clear()
                    conf_hist.clear()
                    current_state = "WAITING"
                    display_caption = ""
                    hold_frames = 0
                    hands_still_frames = 0
                    debug_line = ""
                    already_committed = False
                elif key == ord(' '):
                    frozen = not frozen

                frame_i += 1
                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nVirtual camera stopped")


if __name__ == "__main__":
    main()
