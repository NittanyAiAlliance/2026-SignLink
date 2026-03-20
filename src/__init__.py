# src/app_live.py - ULTRA-STRICT NO-FLICKER VERSION
# ABSOLUTE GUARANTEE: No subtitles until gesture completely finished
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
        print("✅ ULTRA-STRICT MODE - ZERO FLICKER GUARANTEED")
        print("=" * 60)
        print("Rules:")
        print("  - Subtitles ONLY appear in HOLD state")
        print("  - Must be still for 20 frames (0.67 seconds)")
        print("  - 1.5 second cooldown between commits")
        print("  - Requires high confidence (0.65+)")
        print("=" * 60)
    else:
        model = None
        print("⚠️ No model found")

    window = deque(maxlen=cfg.T)
    pred_hist = deque(maxlen=20)  # Larger for more evidence
    conf_hist = deque(maxlen=20)

    selected_label_idx = 0
    frozen = False
    frame_i = 0

    # ====================================================================
    # ULTRA-STRICT PARAMETERS - NO FLICKER GUARANTEED
    # ====================================================================
    
    # Very conservative motion thresholds
    MOTION_START_THR = 0.03   # Easy to start
    MOTION_END_THR = 0.008    # EXTREMELY strict - almost frozen
    STILL_N = 20              # Must be still for 20 frames (0.67 seconds)
    HOLD_N = 45               # Display for 1.5 seconds
    
    # High evidence requirements
    MIN_PREDICTIONS = 12      # Need lots of predictions
    commit_min_conf = 0.65    # Higher confidence threshold (instead of 0.6)
    
    # AGGRESSIVE COOLDOWN - Prevents any possible double-commit
    frames_since_last_commit = 999
    COMMIT_COOLDOWN = 45      # 1.5 seconds between commits
    
    # Very high stability requirement
    MIN_STABLE_FRAMES = 10    # Prediction must be stable for 10 frames
    
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
    
    # ====================================================================
    # ABSOLUTE UI FIREWALL
    # ====================================================================
    display_caption = ""
    display_conf = None
    hold_frames = 0
    
    # DEBUGGING: Track commits
    total_commits = 0
    
    def commit(label: str, conf: float | None):
        """ONLY way to update display. Called ONLY when gesture definitively ends."""
        nonlocal display_caption, display_conf, hold_frames, frames_since_last_commit, total_commits
        
        # ABSOLUTE FIREWALL: Only commit if we're NOT already in HOLD
        if hold_frames > 0:
            print(f"⚠️ BLOCKED: Already displaying a result")
            return
        
        display_caption = label
        display_conf = conf
        hold_frames = HOLD_N
        frames_since_last_commit = 0
        total_commits += 1
        print(f"\n{'='*60}")
        print(f"✅ COMMIT #{total_commits}: '{label}' (conf={conf:.2f if conf else 0})")
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

        # State transitions
        new_gesture_started = False
        if not prev_hands_detected and hands_detected:
            new_gesture_started = True
            print(f"\n>>> NEW GESTURE STARTED")
        elif hands_detected and not gesture_active and motion > MOTION_START_THR:
            new_gesture_started = True
            print(f"\n>>> NEW GESTURE STARTED (motion)")
        
        if not hands_detected:
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
            # CRITICAL: Stay in HOLD state while displaying
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
        
        # ====================================================================
        # SIGNING STATE: Silent background tracking (NO DISPLAY)
        # ====================================================================
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

                debug_line = f"tracking: {cfg.labels[pred]} ({pred_conf:.2f}) | motion={motion:.4f} still={still_count}/{STILL_N}"

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

            # ====================================================================
            # ULTRA-STRICT COMMIT CONDITIONS
            # ====================================================================
            # ALL of these must be true:
            # 1. Not already committed
            # 2. Have enough predictions
            # 3. Cooldown expired
            # 4. NOT already in HOLD state (extra safety)
            # 5. EITHER hands gone OR motion extremely still
            # 6. High confidence
            # 7. High stability
            # ====================================================================
            if (not gesture_committed and 
                len(pred_hist) >= MIN_PREDICTIONS and
                frames_since_last_commit >= COMMIT_COOLDOWN and
                hold_frames == 0):  # EXTRA SAFETY: Don't commit if already showing
                
                vote_counts = Counter(pred_hist)
                if len(vote_counts) > 0:
                    most_common = vote_counts.most_common(1)[0]
                    vote_label, vote_count = most_common
                    
                    label_confs = [c for p, c in zip(pred_hist, conf_hist) if p == vote_label]
                    avg_conf = sum(label_confs) / len(label_confs) if label_confs else 0.0
                    
                    # HYBRID TRIGGERS
                    hands_disappeared = prev_hands_detected and not hands_detected
                    motion_very_still = (still_count >= STILL_N and motion < MOTION_END_THR)
                    
                    # Only commit if gesture DEFINITIVELY ended
                    if (hands_disappeared or motion_very_still):
                        candidate_is_stable = (best_candidate_stable_count >= MIN_STABLE_FRAMES)
                        
                        # PREFER best candidate if stable and confident
                        if (best_candidate_label is not None and 
                            best_candidate_conf >= commit_min_conf and 
                            candidate_is_stable):
                            
                            print(f"🎯 Committing best: {cfg.labels[best_candidate_label]} ({best_candidate_conf:.2f}) [stable={best_candidate_stable_count}]")
                            commit(cfg.labels[best_candidate_label], best_candidate_conf)
                            current_state = "HOLD"
                            gesture_committed = True
                            
                        # Fallback to vote
                        elif avg_conf >= commit_min_conf:
                            print(f"🎯 Committing vote: {cfg.labels[vote_label]} ({avg_conf:.2f})")
                            commit(cfg.labels[vote_label], avg_conf)
                            current_state = "HOLD"
                            gesture_committed = True
                        else:
                            print(f"❌ Motion still but confidence too low: {avg_conf:.2f} < {commit_min_conf}")
                        
                        if gesture_committed:
                            pred_hist.clear()
                            conf_hist.clear()
                            still_count = 0
                            best_candidate_label = None
                            best_candidate_conf = 0.0
                            best_candidate_stable_count = 0
        
        prev_hands_detected = hands_detected

        # ====================================================================
        # ABSOLUTE UI FIREWALL - NO EXCEPTIONS
        # ====================================================================
        # RULE: Subtitles can ONLY appear if hold_frames > 0
        # ====================================================================
        
        if hold_frames > 0:
            hold_frames -= 1
            if hold_frames == 0:
                print(f"⏰ Hold expired - clearing display")
                current_state = "IDLE" if not hands_detected else "SIGNING"
                gesture_committed = False
        
        # ABSOLUTE DISPLAY RULE: Only show during HOLD
        if hold_frames > 0 and current_state == "HOLD":
            caption_to_display = display_caption
            conf_to_display = display_conf
        else:
            # EVERYTHING ELSE: Blank
            caption_to_display = ""
            conf_to_display = None
            if current_state == "IDLE":
                debug_line = ""
        
        if model is None and current_state == "IDLE":
            caption_to_display = f"Selected label: {cfg.labels[selected_label_idx]} (press 1..0)"
            conf_to_display = None

        # Draw
        draw_status(
            frame,
            hands_state
            + (" | FROZEN" if frozen else "")
            + (f" | {debug_line}" if debug_line else "")
            + (f" | STATE={current_state}" if current_state == "HOLD" else "")
        )

        draw_caption(frame, caption_to_display, conf_to_display)

        cv2.imshow(cfg.window_title, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print(f"\n{'='*60}")
            print(f"SESSION ENDED - Total commits: {total_commits}")
            print(f"{'='*60}")
            break
        elif key == ord('c'):
            pred_hist.clear()
            conf_hist.clear()
            display_caption = ""
            display_conf = None
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
        frames_since_last_commit += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()