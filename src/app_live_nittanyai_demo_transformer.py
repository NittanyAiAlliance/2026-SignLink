#!/usr/bin/env python3
"""
SignLink Live App - NittanyAI Demo (Vision Transformer + Phrase Assembler)

Architecture: Temporal Vision Transformer (Vaswani et al. 2017)
  - Each video frame → 156-dim MediaPipe pose features (one "token")
  - 30 tokens pass through 4 self-attention encoder layers
  - Global average pool → classification head

Phrase Assembler:
  - Tracks committed signs and accumulates them into growing sentence subtitles
  - Each word appears next to the previous as it is signed
  - Full sentence holds on screen for 2 seconds after the last word, then clears
  - Grammar-constrained: after each sign, only valid next signs are considered

Demo sentences:
  1. "Hello, Doctor, I am not good, I'm sick."
  2. "I have a fever, my temperature is high."
  3. "102."
  4. "Yes, I have a cough. My throat hurts and my head hurts."
  5. "Two days ago."
  6. "Thank you, I understand."
  7. "Thank you, doctor."

Train first with:
    python scripts/train_transformer.py

Usage:
    python src/app_live_nittanyai_demo_transformer.py
"""

import importlib.util
import json
import os
import sys
from collections import deque
from datetime import datetime
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


# ── Phrase Assembler ──────────────────────────────────────────────────────────
class PhraseAssembler:
    """
    Accumulates individual committed signs into growing sentence subtitles.

    Each sentence is a (sign_sequence, display_sequence) pair.
    As signs fire one by one, the subtitle text grows word-by-word on screen.

    on_sign() returns (display_text, is_complete):
      - display_text : the subtitle string to show right now (None = no change)
      - is_complete  : True when the last word of a sentence has just fired
                       → caller should hold the subtitle for 2 sec then clear
    """

    SENTENCES = [
        # ── Sentence 1 ────────────────────────────────────────────────────────
        (
            ["hello", "doctor", "not_fine", "sick"],
            [
                "Hello",
                "Hello, Doctor,",
                "Hello, Doctor, I am not fine,",
                "Hello, Doctor, I am not fine, I am sick.",
            ],
        ),
        # ── Sentence 2 ────────────────────────────────────────────────────────
        (
            ["i_have", "a_fever", "my_temperature", "is_high"],
            [
                "I have",
                "I have a fever,",
                "I have a fever, my temperature",
                "I have a fever, my temperature is high.",
            ],
        ),
        # ── Sentence 3 ────────────────────────────────────────────────────────
        (
            ["one_hundred_two"],
            ["102."],
        ),
        # ── Sentence 4 ────────────────────────────────────────────────────────
        (
            ["yes", "i_have", "a_headache", "and_a_sore_throat"],
            [
                "Yes,",
                "Yes, I have",
                "Yes, I have a headache",
                "Yes, I have a headache and a sore throat.",
            ],
        ),
        # ── Sentence 5 ────────────────────────────────────────────────────────
        (
            ["two", "days", "ago"],
            ["Two", "Two days", "Two days ago."],
        ),
        # ── Sentence 6 ────────────────────────────────────────────────────────
        (
            ["thank_you", "understand"],
            ["Thank you,", "Thank you, I understand."],
        ),
        # ── Sentence 7 ────────────────────────────────────────────────────────
        (
            ["thank_you", "doctor"],
            ["Thank you,", "Thank you, doctor."],
        ),
    ]

    def __init__(self):
        self.buffer = []        # signs committed so far in current sentence
        self.current_text = ""  # last displayed subtitle text

    def on_sign(self, label: str) -> tuple:
        """
        Process a newly committed sign label.
        Returns (display_text, is_sentence_complete).
        display_text is None if the sign doesn't fit any known sentence.
        """
        candidate = self.buffer + [label]

        # Find sentences where candidate is a valid prefix
        matches = [
            i for i, (signs, _) in enumerate(self.SENTENCES)
            if len(signs) >= len(candidate) and signs[:len(candidate)] == candidate
        ]

        if not matches:
            # No continuation — try starting a fresh sentence with this sign
            candidate = [label]
            matches = [
                i for i, (signs, _) in enumerate(self.SENTENCES)
                if signs[0] == label
            ]
            if not matches:
                # Sign doesn't belong to any known sentence — ignore
                return None, False

        self.buffer = candidate
        pos = len(self.buffer) - 1

        # All matching sentences share the same display text at this position
        # (e.g. both thank_you sentences show "Thank you," after the first sign)
        _, displays = self.SENTENCES[matches[0]]
        self.current_text = displays[pos]

        signs, _ = self.SENTENCES[matches[0]]
        is_complete = len(self.buffer) == len(signs)
        if is_complete:
            self.buffer = []

        return self.current_text, is_complete

    def reset(self):
        self.buffer = []
        self.current_text = ""

    def valid_next_signs(self) -> list:
        """Return the set of signs that are valid at the current buffer position."""
        if not self.buffer:
            return [signs[0] for signs, _ in self.SENTENCES]
        next_signs = []
        for signs, _ in self.SENTENCES:
            n = len(self.buffer)
            if len(signs) > n and signs[:n] == self.buffer:
                next_signs.append(signs[n])
        return next_signs


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
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2

    # Wrap long lines so they fit the frame width
    max_width = w - 80
    words = caption.split(" ")
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(test, font, scale, thick)
        if tw > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)

    line_h = cv2.getTextSize("A", font, scale, thick)[0][1] + 14
    total_h = line_h * len(lines)
    pad_x, pad_y = 25, 18
    box_y1 = h - 40 - total_h - pad_y * 2
    box_y2 = h - 40 + pad_y

    # Measure widest line for box width
    max_tw = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines)
    box_x1 = max(0, (w - max_tw) // 2 - pad_x)
    box_x2 = min(w, (w + max_tw) // 2 + pad_x)

    draw_rounded_rect(frame, (box_x1, box_y1), (box_x2, box_y2),
                      (0, 0, 0), radius=8, alpha=0.75)

    for i, line in enumerate(lines):
        (lw, lh), _ = cv2.getTextSize(line, font, scale, thick)
        x = (w - lw) // 2
        y = box_y1 + pad_y + lh + i * line_h
        cv2.putText(frame, line, (x, y), font, scale, (255, 255, 255),
                    thick, cv2.LINE_AA)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SIGNLINK - NITTANYAI DEMO (PHRASE ASSEMBLER)")
    print("=" * 60)

    MODEL_PATH   = "models_transformer/transformer.pt"
    LABELS_PATH  = "models_transformer/transformer_labels.json"
    CONFIG_PATH  = "models_transformer/transformer_config.json"

    WINDOW_SIZE        = 30     # frames per observation window
    STRIDE             = 3      # run inference every N frames
    CONF_THRESHOLD     = 0.50   # minimum softmax confidence to commit
    STABLE_NEEDED      = 2      # consecutive high-conf hits required to fire
    SENTENCE_HOLD_FRAMES = 60   # hold complete sentence (~2 sec) then clear
    COOLDOWN_FRAMES    = 20     # frames before same sign can re-fire

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
            model(dummy)
        t0 = time.perf_counter()
        for _ in range(50):
            model(dummy)
        ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"Inference: {ms:.1f}ms per window", end="")
    if ms > 10:
        print(f"  WARNING: {ms:.1f}ms > 10ms target — may drop below 30fps")
    else:
        print(f"  OK within 10ms target")

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

    stable_label    = None
    stable_count    = 0
    committed_label = None
    cooldown        = 0

    subtitle_text      = ""
    subtitle_hold      = 0
    sentence_complete  = False
    gesture_count      = 0
    frame_i            = 0
    debug_line         = ""
    last_conf          = 0.0

    # ── Transcript file ───────────────────────────────────────────────────────
    TRANSCRIPT_PATH = Path("transcript.txt")
    session_start   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TRANSCRIPT_PATH, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Session started: {session_start}\n")
        f.write(f"{'='*50}\n")
    print(f"Transcript: {TRANSCRIPT_PATH.resolve()}")

    assembler = PhraseAssembler()

    def commit(label_idx: int, conf: float):
        nonlocal subtitle_text, subtitle_hold, gesture_count
        nonlocal stable_label, stable_count, committed_label, cooldown
        nonlocal sentence_complete

        label = labels[label_idx]
        if label == "background":
            return

        text, is_complete = assembler.on_sign(label)

        if text is not None:
            subtitle_text     = text
            subtitle_hold     = SENTENCE_HOLD_FRAMES if is_complete else 0
            sentence_complete = is_complete
            if is_complete:
                timestamp = datetime.now().strftime("%H:%M:%S")
                with open(TRANSCRIPT_PATH, "a") as f:
                    f.write(f"[{timestamp}] {text}\n")

        gesture_count  += 1
        committed_label = label_idx
        cooldown        = COOLDOWN_FRAMES
        stable_label    = None
        stable_count    = 0
        print(f"[{gesture_count}] {label} → \"{subtitle_text}\"  "
              f"({'COMPLETE' if is_complete else 'building...'})  conf={conf:.2f}")


    print()
    print("=" * 60)
    print("NITTANYAI DEMO — PHRASE ASSEMBLER ACTIVE")
    print(f"  Window : {WINDOW_SIZE} frames | Stride: {STRIDE}")
    print(f"  Threshold: {CONF_THRESHOLD:.0%} | Stable: {STABLE_NEEDED} calls")
    print(f"  Sentence hold: {SENTENCE_HOLD_FRAMES}f (~2 sec) then clears")
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

                # ── Subtitle hold countdown (only fires after complete sentence) ──
                if sentence_complete and subtitle_hold > 0:
                    subtitle_hold -= 1
                    if subtitle_hold == 0:
                        subtitle_text     = ""
                        sentence_complete = False
                        assembler.reset()

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

                    # Track stability
                    if conf >= CONF_THRESHOLD and pred_idx != bg_idx:
                        if pred_idx == stable_label:
                            stable_count += 1
                        else:
                            stable_label = pred_idx
                            stable_count = 1
                    else:
                        stable_label = None
                        stable_count = 0

                    pred_name  = labels[pred_idx]
                    label_disp = pred_name if pred_name != "background" else "background"
                    debug_line = f"{label_disp} ({conf:.0%})"
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
                cv2.putText(display_frame, "Vision Transformer  |  Phrase Assembler",
                            (15, 55), cv2.FONT_HERSHEY_DUPLEX, 0.45, (100, 200, 255), 1)

                # Show valid next signs (debug aid — remove for final demo)
                next_signs = assembler.valid_next_signs()
                if next_signs:
                    next_str = "next: " + "  |  ".join(next_signs[:6])
                    cv2.putText(display_frame, next_str, (15, actual_h - 20),
                                cv2.FONT_HERSHEY_DUPLEX, 0.4, (180, 180, 60), 1)

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
                if sentence_complete and subtitle_hold > 0:
                    border = (0, 255, 0)      # green — sentence complete, holding
                elif subtitle_text:
                    border = (0, 200, 255)    # amber — sentence building
                elif hands_detected and stable_count > 0:
                    border = (0, 255, 255)    # cyan — sign stabilising
                elif hands_detected:
                    border = (150, 150, 150)  # grey — hands present
                else:
                    border = (80, 80, 80)     # dark — idle
                cv2.rectangle(display_frame, (0, 0),
                              (actual_w - 1, actual_h - 1), border, 4)

                cam.send(display_frame)
                cv2.imshow("SignLink - NittanyAI Demo", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    frame_buffer.clear()
                    subtitle_text      = ""
                    subtitle_hold      = 0
                    sentence_complete  = False
                    stable_label       = None
                    stable_count       = 0
                    committed_label    = None
                    cooldown           = 0
                    last_conf          = 0.0
                    debug_line         = ""
                    assembler.reset()

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
