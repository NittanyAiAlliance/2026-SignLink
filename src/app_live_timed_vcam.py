#!/usr/bin/env python3
"""
SignLink Virtual Camera - Timed Recording

Press R to start recording for 1 second (30 frames).
After recording completes, the prediction is shown automatically.

This matches your training data (30 frames per sample).
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


def main():
    print("=" * 70)
    print("SIGNLINK - TIMED RECORDING")
    print("=" * 70)

    # ==========================================================
    # SETTINGS
    # ==========================================================
    T = 30                  # Record exactly 30 frames (1 second)
    HOLD_N = 60             # Show result for 2 seconds

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
    # STATE
    # ==========================================================
    recording = False
    record_buffer = []
    display_caption = ""
    display_conf = None
    hold_frames = 0
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
    print("CONTROLS:")
    print("  R = Start recording (records for 1 second)")
    print("  C = Clear display")
    print("  Q = Quit")
    print()
    print("HOW TO USE:")
    print("  1. Position your hands in frame")
    print("  2. Press R to start recording")
    print("  3. Perform your sign (you have 1 second)")
    print("  4. Result appears automatically")
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

                left = int(info["left_present"])
                right = int(info["right_present"])
                hands_detected = left > 0 or right > 0

                # ==================================================
                # RECORDING LOGIC
                # ==================================================
                if recording:
                    record_buffer.append(feat)
                    frames_left = T - len(record_buffer)

                    # Recording complete
                    if len(record_buffer) >= T:
                        recording = False

                        # Run inference
                        if model is not None:
                            x = np.stack(record_buffer[:T], axis=0).astype(np.float32)
                            xt = torch.from_numpy(x).unsqueeze(0).to(device)

                            with torch.no_grad():
                                logits = model(xt).cpu().numpy()[0]

                            probs = softmax_np(logits)
                            pred = int(np.argmax(probs))
                            conf = float(probs[pred])

                            display_caption = format_label(labels[pred])
                            display_conf = conf
                            hold_frames = HOLD_N

                            print(f"Result: '{display_caption}' ({conf:.2f})")

                        record_buffer = []

                # ==================================================
                # DISPLAY
                # ==================================================
                if hold_frames > 0:
                    hold_frames -= 1
                    if hold_frames == 0:
                        display_caption = ""
                        display_conf = None

                # Build status
                status = f"L={left} R={right}"
                if recording:
                    progress = len(record_buffer)
                    status += f" | RECORDING {progress}/{T}"
                elif hold_frames > 0:
                    status += " | SHOWING"
                else:
                    status += " | Press R to record"

                # Draw UI
                draw_status(frame, status)

                # Show caption
                if display_caption:
                    draw_caption(frame, display_caption, display_conf)

                # Border color
                if recording:
                    color = (0, 0, 255)  # Red while recording
                elif hold_frames > 0:
                    color = (0, 255, 0)  # Green showing result
                else:
                    color = (128, 128, 128)  # Gray waiting

                cv2.rectangle(frame, (0, 0), (actual_w-1, actual_h-1), color, 6)

                # Recording progress bar
                if recording:
                    progress = len(record_buffer) / T
                    bar_width = int(400 * progress)
                    bar_x = (actual_w - 400) // 2
                    bar_y = 60

                    # Background
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 400, bar_y + 30), (50, 50, 50), -1)
                    # Progress
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 30), (0, 0, 255), -1)
                    # Border
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 400, bar_y + 30), (255, 255, 255), 2)
                    # Text
                    cv2.putText(frame, f"RECORDING... {len(record_buffer)}/{T}",
                               (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Ready indicator
                if not recording and hold_frames == 0:
                    cv2.putText(frame, "Press R to record",
                               (actual_w // 2 - 100, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cam.send(frame)
                cv2.imshow("SignLink Timed Recording", frame)

                # ==================================================
                # KEY HANDLING
                # ==================================================
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if not recording:
                        recording = True
                        record_buffer = []
                        gesture_count += 1
                        print(f"\nRecording gesture {gesture_count}...")
                elif key == ord('c'):
                    display_caption = ""
                    display_conf = None
                    hold_frames = 0
                    recording = False
                    record_buffer = []

                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nStopped")


if __name__ == "__main__":
    main()
