#!/usr/bin/env python3
"""
SignLink Virtual Camera - MS-ASL 10 Words Edition
Test your trained model with a virtual camera for video calls.

Signs: brother, family, father, friend, hello, mother, school, teacher, tired, yes
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

from src.features_hands import HandsFeatureExtractor
from src.overlay import draw_caption, draw_status
from src.model import SignGRU


def softmax_np(x: np.ndarray):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


def main():
    print("=" * 70)
    print("SIGNLINK VIRTUAL CAMERA - 10 Word Edition")
    print("=" * 70)
    print("Signs: brother, family, father, friend, hello,")
    print("       mother, school, teacher, tired, yes")
    print("=" * 70)

    # Configuration
    camera_index = 0
    width = 1280
    height = 720
    T = 60  # Window size for inference

    # Camera setup
    print(f"\nOpening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_index}")
        print("Make sure your camera is connected and not being used by another app")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Get actual camera dimensions
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    print(f"Camera opened: {actual_width}x{actual_height} @ {fps}fps")

    # Initialize feature extractor
    extractor = HandsFeatureExtractor()

    # Device selection
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load MS-ASL model
    labels_path = os.path.join("models_msasl", "msasl_labels.json")
    ckpt_path = os.path.join("models_msasl", "msasl_gru.pt")

    if os.path.exists(labels_path) and os.path.exists(ckpt_path):
        with open(labels_path, "r") as f:
            labels = json.load(f)

        model = SignGRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            num_classes=len(labels),
            dropout=0.2
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        print(f"\nModel loaded: {len(labels)} signs")
        print(f"Labels: {labels}")
    else:
        model = None
        labels = []
        print(f"\nWARNING: Model not found!")
        print(f"Expected: {ckpt_path}")
        print("Run training first: modal run scripts/modal_train_gpu.py")

    # State variables
    window = deque(maxlen=T)
    pred_hist = deque(maxlen=15)
    conf_hist = deque(maxlen=15)

    frozen = False
    frame_i = 0

    # Configuration
    HOLD_N = 60  # How long to show caption
    MIN_PREDICTIONS = 6
    HAND_EXIT_CONFIRM_FRAMES = 10

    stride_live = 2

    prev_hands_detected = False
    current_state = "WAITING"
    hands_gone_frames = 0
    best_candidate_label = None
    best_candidate_conf = 0.0

    display_caption = ""
    display_conf = None
    hold_frames = 0

    def commit(label: str, conf: float):
        nonlocal display_caption, display_conf, hold_frames
        display_caption = label
        display_conf = conf
        hold_frames = HOLD_N
        print(f"SHOWING: '{label}' (confidence: {conf:.2f})")

    debug_line = ""
    gesture_count = 0

    # ====================================================================
    # VIRTUAL CAMERA INITIALIZATION
    # ====================================================================
    print("\n" + "=" * 70)
    print("Initializing virtual camera...")
    print("=" * 70)

    try:
        # Try to create virtual camera with OBS backend
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

        print(f"Virtual camera created: {cam.device}")
        print(f"Resolution: {actual_width}x{actual_height} @ {fps}fps")
        print()
        print("=" * 70)
        print("READY TO TEST YOUR 10 SIGNS!")
        print("=" * 70)
        print()
        print("To use in video calls:")
        print("  1. Open your video call app (Zoom, Meet, Teams, etc.)")
        print("  2. Go to camera settings")
        print(f"  3. Select '{cam.device}'")
        print("  4. Start signing!")
        print()
        print("Controls:")
        print("  q - Quit")
        print("  c - Clear/reset state")
        print("  SPACE - Freeze/unfreeze")
        print()
        print("Signs to try:")
        for i, label in enumerate(labels):
            print(f"  {i+1:2d}. {label}")
        print("=" * 70 + "\n")

    except Exception as cam_error:
        print(f"\nERROR: Failed to create virtual camera")
        print(f"Error: {cam_error}")
        print()
        print("SOLUTION:")
        print("  On macOS, pyvirtualcam requires OBS Virtual Camera to be STARTED.")
        print()
        print("  Steps:")
        print("  1. Open OBS Studio")
        print("  2. In OBS, go to: Tools -> Start Virtual Camera")
        print("  3. Wait for 'Virtual Camera Active' message")
        print("  4. Then run this script again")
        print()
        print("  OR use the regular preview app:")
        print("    python src/app_msasl_preview.py")
        print("=" * 70 + "\n")
        cap.release()
        return

    # Main loop
    try:
        with cam:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed")
                    break

                # Mirror for natural signing
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                feat, info = extractor.extract(rgb)
                window.append(feat)

                left = int(info["left_present"])
                right = int(info["right_present"])
                hands_detected = (left > 0 or right > 0)
                hands_state = f"L={left} R={right}"

                # Track hands exit
                if not hands_detected:
                    hands_gone_frames += 1
                else:
                    hands_gone_frames = 0

                # ============================================================
                # STATE MACHINE
                # ============================================================

                if current_state == "WAITING":
                    if hands_detected:
                        current_state = "COLLECTING"
                        gesture_count += 1
                        pred_hist.clear()
                        conf_hist.clear()
                        best_candidate_label = None
                        best_candidate_conf = 0.0
                        display_caption = ""
                        display_conf = None
                        hold_frames = 0
                        print(f"\nGESTURE {gesture_count} STARTED")

                elif current_state == "COLLECTING":
                    if not hands_detected and hands_gone_frames >= HAND_EXIT_CONFIRM_FRAMES:
                        if len(pred_hist) >= MIN_PREDICTIONS:
                            vote_counts = Counter(pred_hist)
                            most_common = vote_counts.most_common(1)[0]
                            vote_label_idx, vote_count = most_common

                            label_confs = [c for p, c in zip(pred_hist, conf_hist) if p == vote_label_idx]
                            avg_conf = sum(label_confs) / len(label_confs) if label_confs else 0.0

                            predicted_label = labels[vote_label_idx]

                            commit(predicted_label, avg_conf)
                            current_state = "SHOWING"
                            print(f"GESTURE {gesture_count} ENDED -> '{predicted_label}' ({avg_conf:.2f})")
                        else:
                            print(f"Not enough predictions ({len(pred_hist)})")
                            current_state = "WAITING"

                elif current_state == "SHOWING":
                    if hands_detected:
                        current_state = "COLLECTING"
                        gesture_count += 1
                        pred_hist.clear()
                        conf_hist.clear()
                        best_candidate_label = None
                        best_candidate_conf = 0.0
                        display_caption = ""
                        display_conf = None
                        hold_frames = 0
                        print(f"\nGESTURE {gesture_count} STARTED")

                # ============================================================
                # INFERENCE
                # ============================================================
                if current_state == "COLLECTING" and (not frozen) and (model is not None):
                    if (len(window) == T) and (frame_i % stride_live == 0):
                        x = np.stack(window, axis=0).astype(np.float32)
                        xt = torch.from_numpy(x).unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits = model(xt).cpu().numpy()[0]
                        probs = softmax_np(logits)
                        pred = int(np.argmax(probs))
                        pred_conf = float(probs[pred])

                        pred_hist.append(pred)
                        conf_hist.append(pred_conf)

                        debug_line = f"{labels[pred]} ({pred_conf:.2f})"

                    if len(pred_hist) >= MIN_PREDICTIONS:
                        vote_counts = Counter(pred_hist)
                        most_common = vote_counts.most_common(1)[0]
                        vote_label_idx, vote_count = most_common

                        label_confs = [c for p, c in zip(pred_hist, conf_hist) if p == vote_label_idx]
                        avg_conf = sum(label_confs) / len(label_confs) if label_confs else 0.0

                        if avg_conf > best_candidate_conf:
                            best_candidate_label = vote_label_idx
                            best_candidate_conf = avg_conf

                prev_hands_detected = hands_detected

                # ============================================================
                # DISPLAY LOGIC
                # ============================================================

                if hold_frames > 0:
                    hold_frames -= 1
                    if hold_frames == 0:
                        current_state = "WAITING"
                        display_caption = ""
                        display_conf = None
                        print("Caption cleared\n")

                if current_state == "SHOWING" and hold_frames > 0:
                    caption_to_display = display_caption
                    conf_to_display = display_conf
                else:
                    caption_to_display = ""
                    conf_to_display = None
                    if current_state == "WAITING":
                        debug_line = ""

                if model is None:
                    caption_to_display = "No model loaded"
                    conf_to_display = None

                # Draw overlays
                status_line = (
                    hands_state +
                    (" | FROZEN" if frozen else "") +
                    (f" | {debug_line}" if debug_line else "") +
                    f" | {current_state}"
                )

                draw_status(frame, status_line)
                draw_caption(frame, caption_to_display, conf_to_display)

                # Visual border
                h, w = frame.shape[:2]
                if current_state == "WAITING":
                    cv2.rectangle(frame, (0, 0), (w-1, h-1), (128, 128, 128), 6)
                elif current_state == "COLLECTING":
                    cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 255, 0), 6)
                else:
                    cv2.rectangle(frame, (0, 0), (w-1, h-1), (255, 0, 0), 6)

                # Send to virtual camera
                cam.send(frame)

                # Also show preview window
                cv2.imshow("SignLink MS-ASL Preview", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(f"\n{'='*70}")
                    print(f"Session ended - Total gestures: {gesture_count}")
                    print(f"{'='*70}\n")
                    break
                elif key == ord('c'):
                    pred_hist.clear()
                    conf_hist.clear()
                    display_caption = ""
                    display_conf = None
                    hold_frames = 0
                    current_state = "WAITING"
                    best_candidate_label = None
                    best_candidate_conf = 0.0
                    hands_gone_frames = 0
                    debug_line = ""
                    print("State cleared\n")
                elif key == ord(' '):
                    frozen = not frozen
                    print(f"Frozen: {frozen}")

                frame_i += 1

                # Wait to maintain fps
                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        print("\n\nStopping virtual camera...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Virtual camera stopped\n")


if __name__ == "__main__":
    main()
