#!/usr/bin/env python3
"""
SignLink Virtual Camera - Body + Hands Features

This version:
- Uses pose features (body + hands) for better accuracy
- Shows prediction only after hands leave the frame
- Falls back to hand-only model if pose model not available
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
    print("SIGNLINK VIRTUAL CAMERA - Body + Hands Edition")
    print("=" * 70)

    # ==========================================================
    # TIMING SETTINGS
    # ==========================================================
    T = 30                          # Window size: 30 frames
    STRIDE = 2                      # Inference every 2 frames
    MIN_PREDICTIONS = 4             # Min predictions to commit
    HAND_EXIT_CONFIRM_FRAMES = 5    # Frames to confirm hands left
    HOLD_N = 50                     # How long to show caption

    # Camera settings
    camera_index = 0
    width = 1280
    height = 720

    # ==========================================================
    # DETERMINE WHICH MODEL TO USE
    # ==========================================================
    # Priority: 1) Pose model (7 words) 2) Hand-only model (7 words)

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
        print(f"  To use pose: train with pose features and save to {pose_model_path}")
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
    hands_gone_frames = 0
    display_caption = ""
    display_conf = None
    hold_frames = 0
    gesture_count = 0
    debug_line = ""
    frozen = False
    frame_i = 0

    def commit(label: str, conf: float):
        nonlocal display_caption, display_conf, hold_frames, current_state
        display_caption = format_label(label)
        display_conf = conf
        hold_frames = HOLD_N
        current_state = "SHOWING"
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
        print(f"  Shows prediction when hands leave frame")
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

                if not hands_detected:
                    hands_gone_frames += 1
                else:
                    hands_gone_frames = 0

                # ==================================================
                # STATE MACHINE (OPTIMIZED)
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
                        print(f"\nGesture {gesture_count} started")

                elif current_state == "COLLECTING":
                    # Commit prediction only when hands leave the frame
                    if not hands_detected and hands_gone_frames >= HAND_EXIT_CONFIRM_FRAMES:
                        if len(pred_hist) >= MIN_PREDICTIONS:
                            vote_counts = Counter(pred_hist)
                            best_idx, _ = vote_counts.most_common(1)[0]
                            confs = [c for p, c in zip(pred_hist, conf_hist) if p == best_idx]
                            avg_conf = sum(confs) / len(confs) if confs else 0
                            commit(labels[best_idx], avg_conf)
                        else:
                            current_state = "WAITING"

                elif current_state == "SHOWING":
                    if hands_detected:
                        current_state = "COLLECTING"
                        gesture_count += 1
                        pred_hist.clear()
                        conf_hist.clear()
                        display_caption = ""
                        display_conf = None
                        hold_frames = 0
                        print(f"\nGesture {gesture_count} started")

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
                    if hold_frames == 0:
                        current_state = "WAITING"
                        display_caption = ""
                        display_conf = None

                caption = display_caption if hold_frames > 0 else ""
                conf = display_conf if hold_frames > 0 else None

                if current_state == "WAITING":
                    debug_line = ""

                if model is None:
                    caption = "No model"
                    conf = None

                status = hands_state
                if frozen:
                    status += " | FROZEN"
                if debug_line and current_state == "COLLECTING":
                    status += f" | {debug_line}"
                status += f" | {current_state}"

                draw_status(frame, status)
                draw_caption(frame, caption, conf)

                h, w = frame.shape[:2]
                color = (128, 128, 128) if current_state == "WAITING" else \
                        (0, 255, 0) if current_state == "COLLECTING" else (255, 0, 0)
                cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 6)

                cam.send(frame)
                cv2.imshow("SignLink Body+Hands Preview", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    pred_hist.clear()
                    conf_hist.clear()
                    current_state = "WAITING"
                    display_caption = ""
                    hold_frames = 0
                    hands_gone_frames = 0
                    debug_line = ""
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
