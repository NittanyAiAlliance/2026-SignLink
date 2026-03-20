#!/usr/bin/env python3
"""
SignLink Live App - Vision Transformer Recognition

Architecture: Temporal Vision Transformer (Vaswani et al. 2017)
  - Each video frame → 156-dim MediaPipe pose features (one "token")
  - 30 tokens pass through 4 self-attention encoder layers
  - Global average pool → classification head

Inference:
  - Sliding window of 30 frames, scored every 3 frames (stride=3)
  - Sign fires when: confidence >= 91%, stable for 3 consecutive calls,
    class is not background
  - Subtitle shown after hands leave the frame (~2 sec hold)
  - Virtual camera output for video calls (Zoom, Meet, WhatsApp, Teams)

Train first with:
    python scripts/train_transformer.py

Usage:
    python src/app_live_transformer.py
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


# ── Load modules without triggering src/__init__.py ──────────────────────────
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


# ── Display labels ────────────────────────────────────────────────────────────
DISPLAY_LABELS = {
    "hello":              "Hello",
    "how_are_you":        "How are you?",
    "my_name_is":         "My name is",
    "kat":                "Kat",
    "nice_to_meet_you":   "Nice to meet you",
    "welcome_to_signlink":"Welcome to SignLink",
    "thank_you":          "Thank you",
    "doctor":             "Doctor",
    "not_fine":           "I am not fine",
    "sick":               "I am sick",
    "have":               "I have",
    "fever":              "Fever",
    "my":                 "My",
    "temperature":        "Temperature",
    "high":               "High",
    "one_hundred_two":    "102",
    "yes":                "Yes",
    "cough":              "Cough",
    "throat_hurts":       "My throat hurts",
    "head_hurts":         "My head hurts",
    "two":                "Two",
    "days":               "Days",
    "ago":                "Ago",
    "understand":         "I understand",
    "background":         None,
}


def format_label(label: str):
    return DISPLAY_LABELS.get(label, label)


# ── Overlay helpers ───────────────────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SIGNLINK - VISION TRANSFORMER RECOGNITION")
    print("=" * 60)

    MODEL_PATH   = "models_transformer/transformer.pt"
    LABELS_PATH  = "models_transformer/transformer_labels.json"
    CONFIG_PATH  = "models_transformer/transformer_config.json"

    WINDOW_SIZE         = 30    # frames per observation window
    STRIDE              = 3     # run inference every N frames
    CONF_THRESHOLD      = 0.65  # minimum softmax confidence to commit
    STABLE_NEEDED       = 2     # consecutive high-conf hits required to fire
    HOLD_FRAMES         = 60    # subtitle hold (~2 sec at 30fps)
    COOLDOWN_FRAMES     = 20    # frames to wait before same sign can re-fire

    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Run 'python scripts/train_transformer.py' first.")
        return

    # ── Load config + model ───────────────────────────────────────────────────
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    bg_idx = labels.index("background") if "background" in labels else -1
    print(f"\nLabels ({len(labels)}): {labels}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

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

    # ── Benchmark inference speed ─────────────────────────────────────────────
    dummy = torch.zeros(1, WINDOW_SIZE, cfg["input_dim"]).to(device)
    import time
    with torch.no_grad():
        for _ in range(5):
            model(dummy)           # warm up
        t0 = time.perf_counter()
        for _ in range(50):
            model(dummy)
        ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"Inference: {ms:.1f}ms per window", end="")
    if ms > 10:
        print(f"  ⚠ WARNING: {ms:.1f}ms > 10ms target — may drop below 30fps")
    else:
        print(f"  ✓ within 10ms target")

    # ── Feature extractor ─────────────────────────────────────────────────────
    print("Initialising feature extractor...")
    extractor = load_feature_extractor()

    # ── Camera ────────────────────────────────────────────────────────────────
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"Camera: {actual_w}x{actual_h} @ {fps}fps")

    # ── Virtual camera ────────────────────────────────────────────────────────
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

    stable_label    = None    # label being tracked for stability
    stable_count    = 0       # consecutive high-conf hits for stable_label
    committed_label = None    # last committed label (for cooldown)
    cooldown        = 0       # frames remaining before same sign can re-fire

    subtitle_text   = ""
    subtitle_hold   = 0
    gesture_count   = 0
    frame_i         = 0
    debug_line      = ""
    last_conf       = 0.0     # most recent inference confidence (for UI bar)

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
    print("VISION TRANSFORMER RECOGNITION")
    print(f"  Window : {WINDOW_SIZE} frames | Stride: {STRIDE}")
    print(f"  Threshold: {CONF_THRESHOLD:.0%} | Stable: {STABLE_NEEDED} calls")
    print(f"  Commits when sign is stable — hands stay in frame")
    print()
    print("CONTROLS: Q=quit, C=clear")
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
                    debug_line   = ""

                # ── Subtitle hold countdown ────────────────────────────────────
                if subtitle_hold > 0:
                    subtitle_hold -= 1
                    if subtitle_hold == 0:
                        subtitle_text = ""

                # ── Cooldown countdown ─────────────────────────────────────────
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
                    last_conf = conf

                    # Track stability for high-confidence non-background preds
                    if conf >= CONF_THRESHOLD and pred_idx != bg_idx:
                        if pred_idx == stable_label:
                            stable_count += 1
                        else:
                            stable_label = pred_idx
                            stable_count = 1
                    else:
                        stable_label = None
                        stable_count = 0

                    label_name = format_label(labels[pred_idx]) or "background"
                    debug_line = f"{label_name} ({conf:.0%})"
                    if stable_count > 0:
                        debug_line += f" [stable:{stable_count}/{STABLE_NEEDED}]"

                    # ── Commit when stable enough ──────────────────────────────
                    if (stable_count >= STABLE_NEEDED
                            and pred_idx != bg_idx
                            and (pred_idx != committed_label or cooldown == 0)):
                        commit(pred_idx, conf)

                # ── Draw ───────────────────────────────────────────────────────
                display_frame = frame.copy()

                # Status bar
                status = f"L={left} R={right}"
                if debug_line and hands_detected:
                    status += f" | {debug_line}"
                cv2.putText(display_frame, status, (15, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1)

                # Architecture label
                cv2.putText(display_frame, "Vision Transformer", (15, 55),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (100, 200, 255), 1)

                # Subtitle
                if subtitle_text:
                    draw_caption(display_frame, subtitle_text)

                # Confidence bar
                if hands_detected and len(frame_buffer) == WINDOW_SIZE and last_conf > 0:
                    bar_x, bar_y = actual_w // 2 - 150, 70
                    bar_w, bar_h = 300, 18
                    fill = int(bar_w * last_conf)
                    cv2.rectangle(display_frame, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                    color = (0, 255, 0) if last_conf >= CONF_THRESHOLD else (0, 165, 255)
                    cv2.rectangle(display_frame, (bar_x, bar_y),
                                  (bar_x + fill, bar_y + bar_h), color, -1)
                    thresh_x = bar_x + int(bar_w * CONF_THRESHOLD)
                    cv2.line(display_frame, (thresh_x, bar_y - 4),
                             (thresh_x, bar_y + bar_h + 4), (255, 255, 0), 2)
                    cv2.rectangle(display_frame, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

                # Border colour
                if subtitle_text and subtitle_hold > 0:
                    border = (0, 255, 0)
                elif hands_detected and stable_count > 0:
                    border = (0, 255, 255)
                elif hands_detected:
                    border = (150, 150, 150)
                else:
                    border = (80, 80, 80)
                cv2.rectangle(display_frame, (0, 0),
                              (actual_w - 1, actual_h - 1), border, 4)

                cam.send(display_frame)
                cv2.imshow("SignLink Transformer", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    frame_buffer.clear()
                    subtitle_text   = ""
                    subtitle_hold   = 0
                    stable_label    = None
                    stable_count    = 0
                    committed_label = None
                    cooldown        = 0
                    last_conf       = 0.0
                    debug_line      = ""

                frame_i += 1
                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print(f"\nStopped. Recognised {gesture_count} signs.")


if __name__ == "__main__":
    main()
