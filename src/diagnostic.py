# src/app_live.py - DIAGNOSTIC VERSION with ON-SCREEN DISPLAY
import json
import os
import sys
from pathlib import Path
from datetime import datetime

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from collections import deque, Counter

import torch

from src.config import AppConfig
from src.features_hands import HandsFeatureExtractor
from src.overlay import draw_caption, draw_status
from src.model import SignGRU

def softmax_np(x: np.ndarray):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)

def main():
    cfg = AppConfig()
    cap = cv2.VideoCapture(cfg.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    extractor = HandsFeatureExtractor()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    labels_path = os.path.join("models", "labels.json")
    ckpt_path = os.path.join("models", "signlink_gru.pt")

    if os.path.exists(labels_path) and os.path.exists(ckpt_path):
        with open(labels_path, "r") as f:
            labels = json.load(f)
        cfg.labels = labels

        model = SignGRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            num_classes=len(cfg.labels),
            dropout=0.2
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        print("=" * 60)
        print("✅ MODEL LOADED - DIAGNOSTIC MODE ACTIVE")
        print("=" * 60)
        print("\nWatch the ON-SCREEN display for:")
        print("  - Motion values (should be < 0.01 when still)")
        print("  - Still counter (should reach 12+ before commit)")
        print("  - Commit count (should be 1 per gesture)")
        print("\nPerform a gesture and watch what happens!")
        print("=" * 60)
    else:
        model = None
        print("⚠️ No model found")

    window = deque(maxlen=cfg.T)
    pred_hist = deque(maxlen=15)
    conf_hist = deque(maxlen=15)

    selected_label_idx = 0
    frozen = False
    frame_i = 0

    # Parameters
    MOTION_START_THR = 0.05
    MOTION_END_THR = 0.015
    STILL_N = 12
    HOLD_N = 25
    
    MIN_PREDICTIONS = 10
    commit_min_conf = cfg.min_confidence
    
    stride_live = 2
    
    prev_feat = None
    still_count = 0
    prev_hands_detected = False
    
    current_state = "IDLE"
    gesture_active = False
    gesture_committed = False
    
    best_candidate_label = None
    best_candidate_conf = 0.0
    best_candidate_stable_count = 0
    MIN_STABLE_FRAMES = 5
    
    display_caption = ""
    display_conf = None
    hold_frames = 0
    
    # DIAGNOSTIC COUNTERS
    total_commits = 0
    commits_this_gesture = 0
    gesture_start_frame = 0
    
    def commit(label: str, conf: float | None):
        nonlocal display_caption, display_conf, hold_frames, total_commits, commits_this_gesture
        display_caption = label
        display_conf = conf
        hold_frames = HOLD_N
        total_commits += 1
        commits_this_gesture += 1
        
        print(f"\n{'='*60}")
        print(f"🎯 COMMIT #{total_commits} (Gesture #{commits_this_gesture})")
        print(f"   Label: {label}")
        print(f"   Confidence: {conf:.2f if conf else 0:.2f}")
        print(f"   Frames since gesture start: {frame_i - gesture_start_frame}")
        print(f"{'='*60}\n")
    
    debug_line = ""
    motion = 0.0

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
        hands_detected = (left > 0 or right > 0)
        hands_state = f"hands: L={left} R={right}"

        # Motion computation
        if prev_feat is not None:
            motion = np.mean(np.abs(feat - prev_feat))
        
        if motion > MOTION_START_THR:
            gesture_active = True
            still_count = 0
        elif motion < MOTION_END_THR:
            if gesture_active:
                still_count += 1
        else:
            still_count = 0
        
        prev_feat = feat.copy()

        # Detect new gesture
        new_gesture_started = False
        if not prev_hands_detected and hands_detected:
            new_gesture_started = True
            gesture_start_frame = frame_i
            commits_this_gesture = 0
            print(f"\n>>> NEW GESTURE STARTED (frame {frame_i})")
        elif hands_detected and not gesture_active and motion > MOTION_START_THR:
            new_gesture_started = True
            gesture_start_frame = frame_i
            commits_this_gesture = 0
            print(f"\n>>> NEW GESTURE STARTED by motion (frame {frame_i})")
        
        # State transitions
        if not hands_detected:
            if prev_hands_detected:
                print(f"<<< HANDS DISAPPEARED (frame {frame_i})")
                if commits_this_gesture == 0:
                    print("    ⚠️ WARNING: Gesture ended with NO commits!")
                elif commits_this_gesture > 1:
                    print(f"    ⚠️ WARNING: Multiple commits ({commits_this_gesture})!")
            current_state = "IDLE"
            gesture_active = False
            gesture_committed = False
            still_count = 0
            if hold_frames == 0:
                pred_hist.clear()
                conf_hist.clear()
                best_candidate_label = None
                best_candidate_conf = 0.0
                best_candidate_stable_count = 0
        elif hold_frames > 0:
            current_state = "HOLD"
        elif hands_detected:
            current_state = "SIGNING"
            if new_gesture_started:
                gesture_committed = False
                pred_hist.clear()
                conf_hist.clear()
                best_candidate_label = None
                best_candidate_conf = 0.0
                best_candidate_stable_count = 0
                still_count = 0
        
        # SIGNING: Inference
        if current_state == "SIGNING" and (not frozen) and (model is not None):
            if (len(window) == cfg.T) and (frame_i % stride_live == 0):
                x = np.stack(window, axis=0).astype(np.float32)
                xt = torch.from_numpy(x).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(xt).cpu().numpy()[0]
                probs = softmax_np(logits)
                pred = int(np.argmax(probs))
                pred_conf = float(probs[pred])
                pred_hist.append(pred)
                conf_hist.append(pred_conf)

                debug_line = f"pred={cfg.labels[pred]} conf={pred_conf:.2f}"

            # Update best candidate
            if len(pred_hist) >= MIN_PREDICTIONS:
                vote_counts = Counter(pred_hist)
                if len(vote_counts) > 0:
                    most_common = vote_counts.most_common(1)[0]
                    vote_label, vote_count = most_common
                    
                    label_confs = [c for p, c in zip(pred_hist, conf_hist) if p == vote_label]
                    avg_conf = sum(label_confs) / len(label_confs) if label_confs else 0.0
                    
                    if vote_label == best_candidate_label:
                        best_candidate_stable_count += 1
                        if avg_conf > best_candidate_conf:
                            best_candidate_conf = avg_conf
                    else:
                        if avg_conf > best_candidate_conf:
                            best_candidate_label = vote_label
                            best_candidate_conf = avg_conf
                            best_candidate_stable_count = 1

            # COMMIT CHECK
            if not gesture_committed and len(pred_hist) >= MIN_PREDICTIONS:
                vote_counts = Counter(pred_hist)
                if len(vote_counts) > 0:
                    most_common = vote_counts.most_common(1)[0]
                    vote_label, vote_count = most_common
                    
                    label_confs = [c for p, c in zip(pred_hist, conf_hist) if p == vote_label]
                    avg_conf = sum(label_confs) / len(label_confs) if label_confs else 0.0
                    
                    motion_still = (still_count >= STILL_N)
                    
                    if motion_still:
                        candidate_is_stable = (best_candidate_stable_count >= MIN_STABLE_FRAMES)
                        
                        if (best_candidate_label is not None and 
                            best_candidate_conf >= commit_min_conf and 
                            candidate_is_stable):
                            commit(cfg.labels[best_candidate_label], best_candidate_conf)
                            current_state = "HOLD"
                            gesture_committed = True
                        elif avg_conf >= commit_min_conf:
                            commit(cfg.labels[vote_label], avg_conf)
                            current_state = "HOLD"
                            gesture_committed = True
                        
                        if gesture_committed:
                            pred_hist.clear()
                            conf_hist.clear()
                            still_count = 0
                            best_candidate_label = None
                            best_candidate_conf = 0.0
                            best_candidate_stable_count = 0
        
        prev_hands_detected = hands_detected

        # Display
        if hold_frames > 0:
            hold_frames -= 1
            if hold_frames == 0:
                current_state = "IDLE" if not hands_detected else "SIGNING"
                gesture_committed = False
        
        if hold_frames > 0:
            caption_to_display = display_caption
            conf_to_display = display_conf
        else:
            caption_to_display = ""
            conf_to_display = None
            if current_state == "IDLE":
                debug_line = ""
        
        if model is None and current_state == "IDLE":
            caption_to_display = f"Selected label: {cfg.labels[selected_label_idx]} (press 1..0)"
            conf_to_display = None

        # ===== ENHANCED ON-SCREEN DIAGNOSTIC DISPLAY =====
        # Draw standard overlay
        draw_status(
            frame,
            hands_state
            + (" | FROZEN" if frozen else "")
            + (f" | {debug_line}" if debug_line else "")
        )

        # Draw LARGE diagnostic info overlay (top-left corner)
        y_offset = 80
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Background box for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 60), (500, 280), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Diagnostic info
        info_lines = [
            f"STATE: {current_state}",
            f"MOTION: {motion:.4f} (start>{MOTION_START_THR:.3f}, end<{MOTION_END_THR:.3f})",
            f"STILL COUNT: {still_count}/{STILL_N}",
            f"PREDICTIONS: {len(pred_hist)}/{MIN_PREDICTIONS}",
            f"STABLE COUNT: {best_candidate_stable_count}/{MIN_STABLE_FRAMES}",
            f"BEST: {cfg.labels[best_candidate_label] if best_candidate_label is not None else 'None'} ({best_candidate_conf:.2f})",
            f"COMMITS THIS GESTURE: {commits_this_gesture}",
            f"TOTAL COMMITS: {total_commits}"
        ]
        
        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if i < 6 else (0, 255, 255)  # Green for status, cyan for counts
            if "COMMITS THIS GESTURE" in line and commits_this_gesture > 1:
                color = (0, 0, 255)  # Red if multiple commits
            cv2.putText(frame, line, (20, y_offset + i*25), font, font_scale, color, thickness)

        draw_caption(frame, caption_to_display, conf_to_display)

        cv2.imshow(cfg.window_title, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print(f"\n{'='*60}")
            print(f"SESSION SUMMARY:")
            print(f"  Total commits: {total_commits}")
            print(f"{'='*60}")
            break
        elif key == ord('c'):
            pred_hist.clear()
            conf_hist.clear()
            commit("", None)
            hold_frames = 0
            still_count = 0
            current_state = "IDLE"
            gesture_active = False
            gesture_committed = False
            best_candidate_label = None
            best_candidate_conf = 0.0
            best_candidate_stable_count = 0
            debug_line = ""
            print("🧹 State cleared")
        elif key == ord(' '):
            frozen = not frozen
        elif key in [ord(str(i)) for i in range(1, 10)]:
            selected_label_idx = int(chr(key)) - 1
        elif key == ord('0'):
            selected_label_idx = 9

        frame_i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()