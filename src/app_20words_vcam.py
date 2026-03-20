#!/usr/bin/env python3
"""
SignLink Virtual Camera - 20 Words Edition

This version uses the 20-word pose model (body + hands features).
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


# Display labels with proper formatting (capitalize first letter)
DISPLAY_LABELS = {
    "brother": "Brother",
    "family": "Family",
    "father": "Father",
    "friend": "Friend",
    "hello": "Hello",
    "mother": "Mother",
    "school": "School",
    "teacher": "Teacher",
    "tired": "Tired",
    "yes": "Yes",
    "nice": "Nice",
    "happy": "Happy",
    "please": "Please",
    "no": "No",
    "help": "Help",
    "like": "Like",
    "what": "What",
    "good": "Good",
    "sister": "Sister",
    "eat": "Eat",
}


def format_label(label: str) -> str:
    """Convert internal label to display format."""
    return DISPLAY_LABELS.get(label, label.title())


def main():
    print("=" * 70)
    print("SIGNLINK VIRTUAL CAMERA - 20 Words Edition")
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
    # MODEL PATHS (20 Words)
    # ==========================================================
    model_path = "models_20words/pose_gru_20.pt"
    labels_path = "models_20words/labels_20.json"
    config_path = "models_20words/config_20.json"

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the 20-word model first.")
        return

    # Use pose features (156 dimensions)
    print("Using POSE model (body + hands) - 20 words")
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
        print("20 WORDS:")
        for i, label in enumerate(labels):
            print(f"  {i+1:2d}. {format_label(label)}")
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

                pose_status = ""
                if "pose_present" in info:
                    pose_status = f" P={int(info['pose_present'])}"

                hands_state = f"L={left} R={right}{pose_status}"

                if not hands_detected:
                    hands_gone_frames += 1
                else:
                    hands_gone_frames = 0

                # ==================================================
                # STATE MACHINE
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
                if current_state == "COLLECTING" and not frozen:
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
                cv2.imshow("SignLink 20 Words Preview", frame)

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
