#!/usr/bin/env python3
"""
SignLink Live App - 4-State HMM Recognition

Key difference from Zuo/GRU approach:
  - Each sign has its own HMM trained on its temporal structure
  - Background is also an HMM — used as a null hypothesis
  - A sign fires only when its log-likelihood beats background
    by a margin (LR_THRESHOLD). No arbitrary confidence cutoff.
  - Left-to-right topology means the model must pass through
    all sign phases — so resting hands can't accidentally complete a sign.

Usage:
    python src/app_live_hmm.py

Train first with:
    python scripts/train_hmm.py
"""

import json
import os
import sys
import pickle
import importlib.util
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import pyvirtualcam


# ── Load feature extractor (avoids src/__init__.py) ──────────────────────────
def load_feature_extractor():
    module_path = Path(__file__).parent / "features_hands_pose.py"
    spec   = importlib.util.spec_from_file_location("features_hands_pose", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HandsPoseFeatureExtractor()


# ── Display labels ────────────────────────────────────────────────────────────
DISPLAY_LABELS = {
    "hello":             "Hello",
    "how_are_you":       "How are you?",
    "my_name_is":        "My name is",
    "kat":               "Kat",
    "nice_to_meet_you":  "Nice to meet you",
    "welcome_to_signlink": "Welcome to SignLink",
    "thank_you":         "Thank you",
    "background":        None,
}

def format_label(label: str):
    return DISPLAY_LABELS.get(label, label)


# ── Overlay helpers ───────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, radius=15, alpha=0.85):
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
    cv2.putText(frame, caption, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


# ── HMM scorer ───────────────────────────────────────────────────────────────
class HMMRecognizer:
    """
    Wraps the trained HMM package and scores a window of frames.

    Scoring:
        For each sign class, compute log P(obs | HMM_sign).
        Compare against log P(obs | HMM_background).
        Accept the sign with the highest likelihood IF it beats
        background by at least LR_THRESHOLD nats/frame.
    """

    def __init__(self, model_path: str, lr_threshold: float = 2.0):
        with open(model_path, "rb") as f:
            pkg = pickle.load(f)

        self.models    = pkg["models"]    # dict: label → GaussianHMM
        self.scaler    = pkg["scaler"]
        self.pca       = pkg["pca"]
        self.labels    = pkg["labels"]
        self.lr_thresh = lr_threshold     # log-likelihood ratio threshold per frame

        self.sign_labels = [l for l in self.labels if l != "background"]
        print(f"  Loaded HMMs for: {self.sign_labels}")

    def score_window(self, window: np.ndarray):
        """
        Score a (T, 156) window.

        Returns:
            best_label  — predicted sign, or 'background'
            best_lr     — log-likelihood ratio vs background (per frame)
            all_scores  — dict of label → score for display
        """
        T = len(window)
        window_scaled = self.scaler.transform(window)
        window_pca    = self.pca.transform(window_scaled)

        scores = {}
        for label, model in self.models.items():
            try:
                scores[label] = model.score(window_pca) / T   # per-frame LL
            except Exception:
                scores[label] = -np.inf

        bg_score = scores.get("background", -np.inf)

        # Find best sign (excluding background)
        best_label, best_score = "background", -np.inf
        for label in self.sign_labels:
            if scores[label] > best_score:
                best_score = scores[label]
                best_label = label

        best_lr = best_score - bg_score   # log-likelihood ratio vs background

        if best_lr < self.lr_thresh:
            return "background", best_lr, scores

        return best_label, best_lr, scores


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SIGNLINK - 4-STATE HMM RECOGNITION")
    print("=" * 60)

    MODEL_PATH  = "models_hmm/hmm_models.pkl"
    WINDOW_SIZE = 30     # frames per observation window
    HOLD_FRAMES = 60     # frames to hold subtitle (~2 sec at 30fps)
    LR_THRESHOLD = 2.0   # log-likelihood ratio threshold (nats/frame)
                         # raise if too many false fires; lower if signs missed

    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Run 'python scripts/train_hmm.py' first.")
        return

    # Load HMM models
    print(f"\nLoading HMM models: {MODEL_PATH}")
    recognizer = HMMRecognizer(MODEL_PATH, lr_threshold=LR_THRESHOLD)

    # Feature extractor
    print("Initialising feature extractor...")
    extractor = load_feature_extractor()

    # Camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"Camera: {actual_w}x{actual_h} @ {fps}fps")

    # Virtual camera
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

    # State
    frame_buffer  = deque(maxlen=WINDOW_SIZE)
    subtitle_text = ""
    subtitle_hold = 0
    last_output   = None
    last_was_blank = True
    gesture_count  = 0
    current_label  = "ready"
    current_lr     = 0.0

    print()
    print("=" * 60)
    print("HMM RECOGNITION:")
    print(f"  Window : {WINDOW_SIZE} frames")
    print(f"  LR threshold: {LR_THRESHOLD:.1f} nats/frame")
    print()
    print("CONTROLS: Q=quit, C=clear, +/- adjust LR threshold")
    print("=" * 60 + "\n")

    try:
        with cam:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Extract features
                feat, info = extractor.extract(rgb)
                frame_buffer.append(feat)

                left  = int(info["left_present"])
                right = int(info["right_present"])
                hands_detected = left > 0 or right > 0

                # Subtitle hold countdown
                if subtitle_hold > 0:
                    subtitle_hold -= 1
                    if subtitle_hold == 0:
                        subtitle_text = ""

                # Run HMM scoring when buffer is full and hands are visible
                if hands_detected and len(frame_buffer) == WINDOW_SIZE:
                    window = np.array(list(frame_buffer), dtype=np.float32)
                    pred_label, lr, all_scores = recognizer.score_window(window)

                    current_label = pred_label
                    current_lr    = lr

                    if pred_label == "background":
                        last_was_blank = True
                    else:
                        is_new = (pred_label != last_output) or last_was_blank
                        if is_new:
                            gesture_count += 1
                            display = format_label(pred_label)
                            if display:
                                subtitle_text = display
                                subtitle_hold = HOLD_FRAMES
                                print(f"[{gesture_count}] {display}  (LR={lr:.2f})")
                            last_output    = pred_label
                            last_was_blank = False
                elif not hands_detected:
                    last_was_blank = True
                    current_label  = "no hands"
                    current_lr     = 0.0

                # ── Draw ──────────────────────────────────────────────────────
                display_frame = frame.copy()

                # Status bar
                status = f"L={left} R={right}"
                if current_label not in ("ready", "no hands", "background"):
                    status += f" | {format_label(current_label)} (LR={current_lr:.1f})"
                elif current_label == "background":
                    status += f" | background (LR={current_lr:.1f})"
                else:
                    status += f" | {current_label}"

                cv2.putText(display_frame, status, (15, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)

                # Threshold hint
                cv2.putText(display_frame, f"LR thresh: {recognizer.lr_thresh:.1f}  (+/- to adjust)",
                            (15, 55), cv2.FONT_HERSHEY_DUPLEX, 0.45, (150, 150, 150), 1)

                # Subtitle
                if subtitle_text:
                    draw_caption(display_frame, subtitle_text)

                # Border colour
                if subtitle_text and subtitle_hold > 0:
                    border = (0, 255, 0)       # green — showing
                elif hands_detected and current_label not in ("background", "no hands", "ready"):
                    border = (0, 255, 255)     # yellow — detecting
                elif hands_detected:
                    border = (150, 150, 150)   # grey — ready
                else:
                    border = (80, 80, 80)      # dark — no hands

                cv2.rectangle(display_frame, (0, 0), (actual_w-1, actual_h-1), border, 4)

                # LR bar (replaces confidence bar)
                if hands_detected and len(frame_buffer) == WINDOW_SIZE:
                    bar_x, bar_y = actual_w // 2 - 150, 70
                    bar_w, bar_h = 300, 18
                    max_lr = 10.0
                    fill = int(bar_w * min(max(current_lr, 0), max_lr) / max_lr)

                    cv2.rectangle(display_frame, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                    bar_color = (0, 255, 0) if current_lr >= recognizer.lr_thresh else (0, 100, 255)
                    cv2.rectangle(display_frame, (bar_x, bar_y),
                                  (bar_x + fill, bar_y + bar_h), bar_color, -1)
                    cv2.rectangle(display_frame, (bar_x, bar_y),
                                  (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
                    # threshold marker
                    thresh_x = bar_x + int(bar_w * recognizer.lr_thresh / max_lr)
                    cv2.line(display_frame, (thresh_x, bar_y), (thresh_x, bar_y + bar_h),
                             (255, 255, 0), 2)
                    cv2.putText(display_frame,
                                f"LR={current_lr:.1f} vs bg",
                                (bar_x, bar_y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                cam.send(display_frame)
                cv2.imshow("SignLink HMM", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    frame_buffer.clear()
                    subtitle_text  = ""
                    subtitle_hold  = 0
                    last_output    = None
                    last_was_blank = True
                elif key == ord('+'):
                    recognizer.lr_thresh += 0.5
                    print(f"LR threshold → {recognizer.lr_thresh:.1f}")
                elif key == ord('-'):
                    recognizer.lr_thresh = max(0.5, recognizer.lr_thresh - 0.5)
                    print(f"LR threshold → {recognizer.lr_thresh:.1f}")

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
