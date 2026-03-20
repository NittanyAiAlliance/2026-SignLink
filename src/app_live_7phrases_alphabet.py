#!/usr/bin/env python3
"""
SignLink Virtual Camera - 7 Phrases + Alphabet Combined

This version combines:
- 7 phrases model (body + hands features, 156-dim)
- Alphabet model (hand-only features, 63-dim)

Mode switching:
- Hold hand still for alphabet recognition (static letters)
- Move hands for phrase recognition (dynamic signs)
- Press 'M' to toggle between PHRASE/ALPHABET/AUTO modes
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


# Display labels for phrases
PHRASE_DISPLAY = {
    "hello": "Hello",
    "how_are_you": "How are you?",
    "my_name_is": "My name is",
    "kat": "Kat",
    "nice_to_meet_you": "Nice to meet you",
    "welcome_to_signlink": "Welcome to SignLink",
    "thank_you": "Thank you",
}

# Display labels for alphabet
ALPHABET_DISPLAY = {letter: letter.upper() for letter in "abcdefghijklmnopqrstuvwxyz"}
ALPHABET_DISPLAY.update({"del": "DEL", "nothing": "", "space": "SPACE"})


def format_label(label: str, mode: str) -> str:
    """Convert internal label to display format based on mode."""
    if mode == "ALPHABET":
        return ALPHABET_DISPLAY.get(label, label.upper())
    else:
        return PHRASE_DISPLAY.get(label, label)


def calculate_motion(window, n_frames=5):
    """Calculate hand motion over recent frames to detect static vs dynamic signs."""
    if len(window) < n_frames:
        return 0.0

    recent = list(window)[-n_frames:]
    # Use first 63 features (right hand) or 63-126 (left hand)
    # Compare consecutive frames
    motion = 0.0
    for i in range(1, len(recent)):
        diff = np.abs(recent[i][:126] - recent[i-1][:126])  # Both hands
        motion += np.mean(diff)
    return motion / (n_frames - 1)


def main():
    print("=" * 70)
    print("SIGNLINK VIRTUAL CAMERA - 7 Phrases + Alphabet")
    print("=" * 70)

    # ==========================================================
    # TIMING SETTINGS
    # ==========================================================
    T_PHRASES = 30              # Window size for phrases
    T_ALPHABET = 10             # Window size for alphabet (shorter)
    STRIDE = 2                  # Inference every 2 frames
    MIN_PREDICTIONS = 4         # Min predictions to commit
    HAND_EXIT_CONFIRM = 5       # Frames to confirm hands left
    HOLD_N = 50                 # How long to show caption
    MOTION_THRESHOLD = 0.015    # Threshold for static vs dynamic

    # Camera settings
    camera_index = 0
    width = 1280
    height = 720

    # ==========================================================
    # LOAD MODELS
    # ==========================================================

    # Phrases model paths
    phrases_model_path = "models_signlink_pose/pose_gru.pt"
    phrases_labels_path = "models_signlink_pose/pose_labels.json"

    # Alphabet model paths
    alphabet_model_path = "models_alphabet/alphabet_gru.pt"
    alphabet_labels_path = "models_alphabet/alphabet_labels.json"
    alphabet_config_path = "models_alphabet/alphabet_config.json"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Initialize feature extractor (shared)
    from src.features_hands_pose import HandsPoseFeatureExtractor
    extractor = HandsPoseFeatureExtractor()
    print("Feature extractor: HandsPoseFeatureExtractor (156-dim)")

    # Load phrases model
    phrases_model = None
    phrases_labels = []
    if os.path.exists(phrases_model_path) and os.path.exists(phrases_labels_path):
        with open(phrases_labels_path, "r") as f:
            phrases_labels = json.load(f)

        phrases_model = SignGRU(
            input_size=156,  # hands + pose
            hidden_size=128,
            num_layers=2,
            num_classes=len(phrases_labels),
            dropout=0.2
        )
        phrases_model.load_state_dict(torch.load(phrases_model_path, map_location=device))
        phrases_model.to(device)
        phrases_model.eval()
        print(f"Phrases model: {len(phrases_labels)} signs - {phrases_labels}")
    else:
        print(f"WARNING: Phrases model not found at {phrases_model_path}")

    # Load alphabet model
    alphabet_model = None
    alphabet_labels = []
    if os.path.exists(alphabet_model_path) and os.path.exists(alphabet_labels_path):
        with open(alphabet_labels_path, "r") as f:
            alphabet_labels = json.load(f)

        # Load config to get model architecture
        with open(alphabet_config_path, "r") as f:
            alphabet_config = json.load(f)

        alphabet_model = SignGRU(
            input_size=alphabet_config["input_size"],  # 63
            hidden_size=alphabet_config["hidden_size"],  # 256
            num_layers=alphabet_config["num_layers"],  # 2
            num_classes=len(alphabet_labels),
            dropout=0.3
        )
        alphabet_model.load_state_dict(torch.load(alphabet_model_path, map_location=device))
        alphabet_model.to(device)
        alphabet_model.eval()
        print(f"Alphabet model: {len(alphabet_labels)} letters")
    else:
        print(f"WARNING: Alphabet model not found at {alphabet_model_path}")

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
    # STATE VARIABLES
    # ==========================================================
    window = deque(maxlen=T_PHRASES)  # Use larger window
    alphabet_window = deque(maxlen=T_ALPHABET)  # Separate window for alphabet

    pred_hist = deque(maxlen=15)
    conf_hist = deque(maxlen=15)

    current_state = "WAITING"
    recognition_mode = "AUTO"  # AUTO, PHRASE, ALPHABET
    active_model = "PHRASE"    # Which model is currently being used

    hands_gone_frames = 0
    display_caption = ""
    display_conf = None
    hold_frames = 0
    gesture_count = 0
    debug_line = ""
    frozen = False
    frame_i = 0
    current_motion = 0.0

    def commit(label: str, conf: float, mode: str):
        nonlocal display_caption, display_conf, hold_frames, current_state
        display_caption = format_label(label, mode)
        display_conf = conf
        hold_frames = HOLD_N
        current_state = "SHOWING"
        print(f"SHOWING ({mode}): '{display_caption}' ({conf:.2f})")

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
        print("MODES:")
        print("  AUTO    - Automatic: static hands = alphabet, moving = phrases")
        print("  PHRASE  - Force phrase recognition only")
        print("  ALPHABET- Force alphabet recognition only")
        print()
        print("CONTROLS:")
        print("  M = Cycle modes (AUTO -> PHRASE -> ALPHABET -> AUTO)")
        print("  Q = Quit")
        print("  C = Clear/Reset")
        print("  SPACE = Freeze")
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

                # Extract features (156-dim for phrases)
                feat, info = extractor.extract(rgb)
                window.append(feat)

                # Also maintain alphabet window (63-dim right hand only)
                # Take right hand features (indices 63:126) from the full feature vector
                right_hand_feat = feat[63:126]  # Right hand (63 features)
                alphabet_window.append(right_hand_feat)

                left = int(info["left_present"])
                right = int(info["right_present"])
                pose = int(info.get("pose_present", 0))
                hands_detected = left > 0 or right > 0

                hands_state = f"L={left} R={right} P={pose}"

                if not hands_detected:
                    hands_gone_frames += 1
                else:
                    hands_gone_frames = 0

                # Calculate motion for AUTO mode
                current_motion = calculate_motion(window)

                # ==================================================
                # DETERMINE ACTIVE MODEL (in AUTO mode)
                # ==================================================
                if recognition_mode == "AUTO":
                    if current_motion < MOTION_THRESHOLD and right > 0:
                        active_model = "ALPHABET"
                    else:
                        active_model = "PHRASE"
                elif recognition_mode == "PHRASE":
                    active_model = "PHRASE"
                else:
                    active_model = "ALPHABET"

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
                        print(f"\nGesture {gesture_count} started (mode: {recognition_mode}, active: {active_model})")

                elif current_state == "COLLECTING":
                    # Commit prediction when hands leave frame
                    if not hands_detected and hands_gone_frames >= HAND_EXIT_CONFIRM:
                        if len(pred_hist) >= MIN_PREDICTIONS:
                            vote_counts = Counter(pred_hist)
                            best_idx, _ = vote_counts.most_common(1)[0]
                            confs = [c for p, c in zip(pred_hist, conf_hist) if p == best_idx]
                            avg_conf = sum(confs) / len(confs) if confs else 0

                            # Get label from appropriate model
                            if active_model == "ALPHABET" and alphabet_labels:
                                commit(alphabet_labels[best_idx], avg_conf, "ALPHABET")
                            elif phrases_labels:
                                commit(phrases_labels[best_idx], avg_conf, "PHRASE")
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
                        print(f"\nGesture {gesture_count} started (mode: {recognition_mode}, active: {active_model})")

                # ==================================================
                # INFERENCE
                # ==================================================
                if current_state == "COLLECTING" and not frozen and frame_i % STRIDE == 0:

                    if active_model == "ALPHABET" and alphabet_model is not None:
                        # Alphabet inference (63-dim, T=10)
                        if len(alphabet_window) >= T_ALPHABET:
                            x = np.stack(list(alphabet_window)[-T_ALPHABET:], axis=0).astype(np.float32)
                            xt = torch.from_numpy(x).unsqueeze(0).to(device)
                            with torch.no_grad():
                                logits = alphabet_model(xt).cpu().numpy()[0]
                            probs = softmax_np(logits)
                            pred = int(np.argmax(probs))
                            pred_conf = float(probs[pred])

                            # Filter out "nothing" predictions
                            if alphabet_labels[pred] != "nothing":
                                pred_hist.append(pred)
                                conf_hist.append(pred_conf)
                                debug_line = f"[A] {format_label(alphabet_labels[pred], 'ALPHABET')} ({pred_conf:.2f})"

                    elif phrases_model is not None:
                        # Phrases inference (156-dim, T=30)
                        if len(window) == T_PHRASES:
                            x = np.stack(window, axis=0).astype(np.float32)
                            xt = torch.from_numpy(x).unsqueeze(0).to(device)
                            with torch.no_grad():
                                logits = phrases_model(xt).cpu().numpy()[0]
                            probs = softmax_np(logits)
                            pred = int(np.argmax(probs))
                            pred_conf = float(probs[pred])

                            pred_hist.append(pred)
                            conf_hist.append(pred_conf)
                            debug_line = f"[P] {format_label(phrases_labels[pred], 'PHRASE')} ({pred_conf:.2f})"

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

                if phrases_model is None and alphabet_model is None:
                    caption = "No models loaded"
                    conf = None

                # Build status line
                status = hands_state
                status += f" | Mode:{recognition_mode}"
                status += f" | Active:{active_model[0]}"  # P or A
                status += f" | Motion:{current_motion:.3f}"
                if frozen:
                    status += " | FROZEN"
                if debug_line and current_state == "COLLECTING":
                    status += f" | {debug_line}"
                status += f" | {current_state}"

                draw_status(frame, status)
                draw_caption(frame, caption, conf)

                # Border color based on state and active model
                h, w = frame.shape[:2]
                if current_state == "WAITING":
                    color = (128, 128, 128)  # Gray
                elif current_state == "COLLECTING":
                    if active_model == "ALPHABET":
                        color = (255, 165, 0)  # Orange for alphabet
                    else:
                        color = (0, 255, 0)  # Green for phrases
                else:
                    color = (255, 0, 0)  # Blue for showing
                cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 6)

                # Mode indicator
                mode_text = f"Mode: {recognition_mode}"
                cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 255), 2)

                cam.send(frame)
                cv2.imshow("SignLink 7Phrases+Alphabet", frame)

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
                elif key == ord('m'):
                    # Cycle through modes
                    if recognition_mode == "AUTO":
                        recognition_mode = "PHRASE"
                    elif recognition_mode == "PHRASE":
                        recognition_mode = "ALPHABET"
                    else:
                        recognition_mode = "AUTO"
                    print(f"\nMode changed to: {recognition_mode}")
                    pred_hist.clear()
                    conf_hist.clear()

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
