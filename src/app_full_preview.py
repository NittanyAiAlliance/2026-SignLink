#!/usr/bin/env python3
"""
SignLink Preview - Full Features Model (Hands + Pose + Face)
Test the model with 193 features.
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

from src.features_full import FullFeatureExtractor
from src.overlay import draw_caption, draw_status
from src.model import SignGRU


def softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


def main():
    print("=" * 70)
    print("SIGNLINK - Full Features Model (Hands + Pose + Face)")
    print("=" * 70)

    camera_index = 0
    T = 60

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Initializing Full Feature Extractor (hands + pose + face)...")
    extractor = FullFeatureExtractor()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    labels_path = "models_full/full_labels.json"
    model_path = "models_full/full_gru.pt"
    config_path = "models_full/full_config.json"

    if os.path.exists(model_path):
        with open(labels_path) as f:
            labels = json.load(f)
        with open(config_path) as f:
            config = json.load(f)

        model = SignGRU(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_classes=config["num_classes"],
            dropout=0.2
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded: {config['input_size']} features, {len(labels)} signs")
    else:
        model = None
        labels = []
        print("Model not found!")

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

    HOLD_N = 60
    MIN_PREDICTIONS = 6
    HAND_EXIT_FRAMES = 10

    def commit(label, conf):
        nonlocal display_caption, display_conf, hold_frames
        display_caption = label
        display_conf = conf
        hold_frames = HOLD_N

    print("\nReady! Press Q to quit, C to clear, SPACE to freeze")
    print("=" * 70)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feat, info = extractor.extract(rgb)
            window.append(feat)

            hands_detected = info["left_present"] > 0 or info["right_present"] > 0
            hands_gone_frames = 0 if hands_detected else hands_gone_frames + 1

            # State machine
            if current_state == "WAITING" and hands_detected:
                current_state = "COLLECTING"
                gesture_count += 1
                pred_hist.clear()
                conf_hist.clear()
                display_caption = ""
                hold_frames = 0
                print(f"\nGesture {gesture_count} started")

            elif current_state == "COLLECTING":
                if not hands_detected and hands_gone_frames >= HAND_EXIT_FRAMES:
                    if len(pred_hist) >= MIN_PREDICTIONS:
                        votes = Counter(pred_hist)
                        best_idx, _ = votes.most_common(1)[0]
                        confs = [c for p, c in zip(pred_hist, conf_hist) if p == best_idx]
                        avg_conf = sum(confs) / len(confs)
                        commit(labels[best_idx], avg_conf)
                        current_state = "SHOWING"
                        print(f"-> {labels[best_idx]} ({avg_conf:.2f})")
                    else:
                        current_state = "WAITING"

            elif current_state == "SHOWING" and hands_detected:
                current_state = "COLLECTING"
                gesture_count += 1
                pred_hist.clear()
                conf_hist.clear()
                display_caption = ""
                hold_frames = 0

            # Inference
            if current_state == "COLLECTING" and not frozen and model and len(window) == T and frame_i % 2 == 0:
                x = np.stack(window, axis=0).astype(np.float32)
                xt = torch.from_numpy(x).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(xt).cpu().numpy()[0]
                probs = softmax_np(logits)
                pred = int(np.argmax(probs))
                pred_hist.append(pred)
                conf_hist.append(float(probs[pred]))
                debug_line = f"{labels[pred]} ({probs[pred]:.2f})"

            # Display
            if hold_frames > 0:
                hold_frames -= 1
                if hold_frames == 0:
                    current_state = "WAITING"
                    display_caption = ""

            status = f"L={int(info['left_present'])} R={int(info['right_present'])} F={int(info['face_present'])}"
            if debug_line and current_state == "COLLECTING":
                status += f" | {debug_line}"
            status += f" | {current_state}"

            draw_status(frame, status)
            draw_caption(frame, display_caption if hold_frames > 0 else "", display_conf if hold_frames > 0 else None)

            h, w = frame.shape[:2]
            color = (128, 128, 128) if current_state == "WAITING" else (0, 255, 0) if current_state == "COLLECTING" else (255, 0, 0)
            cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 6)

            cv2.imshow("SignLink Full Model (Q=quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                pred_hist.clear()
                conf_hist.clear()
                current_state = "WAITING"
                display_caption = ""
                hold_frames = 0
            elif key == ord(' '):
                frozen = not frozen

            frame_i += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
