#!/usr/bin/env python3
"""
SignLink Live App - Vision Transformer Recognition (NO PREVIEW)

Runs silently in the background — no cv2 window.
Sends overlaid frames directly to the virtual camera for video calls.

  - Subtitle shown after hands leave the frame
  - Virtual camera output: select in Zoom / Meet / WhatsApp / Teams
  - Press Ctrl+C to stop

Train first with:
    python scripts/train_transformer.py

Usage:
    python src/app_live_transformer_no_preview.py
"""

import importlib.util
import json
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyvirtualcam
import torch

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_module(name: str, path: Path):
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_feature_extractor():
    m = _load_module("features_hands_pose",
                     Path(__file__).parent / "features_hands_pose.py")
    return m.HandsPoseFeatureExtractor()


def load_transformer_class():
    m = _load_module("sign_transformer",
                     Path(__file__).parent / "sign_transformer.py")
    return m.SignTransformer


DISPLAY_LABELS = {
    "hello":              "Hello",
    "how_are_you":        "How are you?",
    "my_name_is":         "My name is",
    "kat":                "Kat",
    "nice_to_meet_you":   "Nice to meet you",
    "welcome_to_signlink":"Welcome to SignLink",
    "thank_you":          "Thank you",
    "background":         None,
}


def format_label(label: str):
    return DISPLAY_LABELS.get(label, label)


def draw_rounded_rect(img, pt1, pt2, color, radius=10, alpha=0.7):
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_caption(frame, caption):
    if not caption:
        return
    h, w = frame.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.4, 2
    (tw, th), _ = cv2.getTextSize(caption, font, scale, thick)
    x = (w - tw) // 2
    y = h - 60
    pad_x, pad_y = 25, 18
    draw_rounded_rect(frame,
                      (max(0, x - pad_x),     max(0, y - th - pad_y)),
                      (min(w, x + tw + pad_x), min(h, y + pad_y)),
                      (0, 0, 0), radius=8, alpha=0.7)
    cv2.putText(frame, caption, (x, y), font, scale, (255, 255, 255),
                thick, cv2.LINE_AA)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


def main():
    print("=" * 60)
    print("SIGNLINK - VISION TRANSFORMER (NO PREVIEW)")
    print("=" * 60)

    MODEL_PATH  = "models_transformer/transformer.pt"
    LABELS_PATH = "models_transformer/transformer_labels.json"
    CONFIG_PATH = "models_transformer/transformer_config.json"

    WINDOW_SIZE     = 30
    STRIDE          = 3
    CONF_THRESHOLD  = 0.91
    STABLE_NEEDED   = 3
    HOLD_FRAMES     = 60
    COOLDOWN_FRAMES = 20

    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Run 'python scripts/train_transformer.py' first.")
        return

    with open(LABELS_PATH) as f:
        labels = json.load(f)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    bg_idx = labels.index("background") if "background" in labels else -1

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Labels ({len(labels)}): {labels}")

    SignTransformer = load_transformer_class()
    model = SignTransformer(
        input_dim=cfg["input_dim"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        num_classes=cfg["num_classes"],
        max_seq_len=cfg["max_seq_len"],
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded.")

    print("Initialising feature extractor...")
    extractor = load_feature_extractor()

    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"Camera: {actual_w}x{actual_h} @ {fps}fps")

    print("Initialising virtual camera...")
    try:
        try:
            cam = pyvirtualcam.Camera(width=actual_w, height=actual_h, fps=fps,
                                      fmt=pyvirtualcam.PixelFormat.BGR,
                                      backend=pyvirtualcam.Backend.OBS)
        except (AttributeError, ValueError):
            cam = pyvirtualcam.Camera(width=actual_w, height=actual_h, fps=fps,
                                      fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"Virtual camera: {cam.device}")
    except Exception as e:
        print(f"ERROR: Virtual camera failed: {e}")
        cap.release()
        return

    # ── State ─────────────────────────────────────────────────────────────────
    frame_buffer    = deque(maxlen=WINDOW_SIZE)

    stable_label    = None
    stable_count    = 0
    committed_label = None
    cooldown        = 0

    subtitle_text = ""
    subtitle_hold = 0
    gesture_count = 0
    frame_i       = 0

    def commit(label_idx: int, conf: float):
        nonlocal subtitle_text, subtitle_hold, gesture_count
        nonlocal stable_label, stable_count, committed_label, cooldown
        label   = labels[label_idx]
        display = format_label(label)
        if display:
            subtitle_text = display
            subtitle_hold = HOLD_FRAMES
        gesture_count  += 1
        committed_label = label_idx
        cooldown        = COOLDOWN_FRAMES
        stable_label    = None
        stable_count    = 0
        print(f"[{gesture_count}] {display}  (conf={conf:.2f})")

    print()
    print("=" * 60)
    print("NO PREVIEW — running in background")
    print(f"Select '{cam.device}' in your video call app")
    print(f"  Threshold: {CONF_THRESHOLD:.0%} | Stable: {STABLE_NEEDED} calls")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        with cam:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                feat, info = extractor.extract(rgb)
                frame_buffer.append(feat)

                left  = int(info["left_present"])
                right = int(info["right_present"])
                hands_detected = left > 0 or right > 0

                # Reset stability when hands leave
                if not hands_detected:
                    stable_label = None
                    stable_count = 0

                if subtitle_hold > 0:
                    subtitle_hold -= 1
                    if subtitle_hold == 0:
                        subtitle_text = ""

                if cooldown > 0:
                    cooldown -= 1

                # ── Inference (runs while hands present) ───────────────────────
                if (hands_detected
                        and len(frame_buffer) == WINDOW_SIZE
                        and frame_i % STRIDE == 0):

                    window = np.stack(list(frame_buffer), axis=0).astype(np.float32)
                    xt = torch.from_numpy(window).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(xt).cpu().numpy()[0]

                    probs    = softmax(logits)
                    pred_idx = int(np.argmax(probs))
                    conf     = float(probs[pred_idx])

                    if conf >= CONF_THRESHOLD and pred_idx != bg_idx:
                        if pred_idx == stable_label:
                            stable_count += 1
                        else:
                            stable_label = pred_idx
                            stable_count = 1
                    else:
                        stable_label = None
                        stable_count = 0

                    # Commit when stable enough
                    if (stable_count >= STABLE_NEEDED
                            and pred_idx != bg_idx
                            and (pred_idx != committed_label or cooldown == 0)):
                        commit(pred_idx, conf)

                # ── Draw overlay + send to virtual camera ──────────────────────
                draw_caption(frame, subtitle_text)

                if subtitle_text and subtitle_hold > 0:
                    border = (0, 255, 0)
                elif hands_detected and stable_count > 0:
                    border = (0, 255, 255)
                elif hands_detected:
                    border = (150, 150, 150)
                else:
                    border = (80, 80, 80)
                cv2.rectangle(frame, (0, 0), (actual_w - 1, actual_h - 1), border, 4)

                cam.send(frame)   # no cv2.imshow
                frame_i += 1
                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        print(f"\nStopping... Recognised {gesture_count} signs.")
    finally:
        cap.release()
        extractor.close()
        print("Virtual camera stopped.")


if __name__ == "__main__":
    main()
