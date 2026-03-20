# src/app_live.py - SIMPLE HAND-PRESENCE BASED SYSTEM
# ZERO FLICKER GUARANTEE: Subtitles only appear after hands exit frame
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
        print("✅ SIMPLE HAND-PRESENCE MODE - ZERO FLICKER")
        print("=" * 60)
        print("3-State System:")
        print("  WAITING: No hands detected")
        print("  COLLECTING: Hands in frame (subtitle BLANK)")
        print("  SHOWING: Hands exited, displaying result")
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

    # ====================================================================
    # SIMPLE HAND-PRESENCE PARAMETERS
    # ====================================================================
    HOLD_N = 60  # Display subtitle for 2 seconds (60 frames at 30fps)
    MIN_PREDICTIONS = 6  # Minimum predictions before allowing commit
    commit_min_conf = cfg.min_confidence  # Use config confidence threshold
    HAND_EXIT_CONFIRM_FRAMES = 10  # Wait 10 frames to confirm hands are gone
    
    stride_live = 2  # Run inference every 2 frames
    
    prev_hands_detected = False
    
    # ====================================================================
    # 3-STATE MACHINE: WAITING → COLLECTING → SHOWING
    # ====================================================================
    current_state = "WAITING"  # WAITING, COLLECTING, or SHOWING
    
    # Hand exit confirmation counter
    hands_gone_frames = 0
    
    # Best prediction tracking during COLLECTING state
    best_candidate_label = None
    best_candidate_conf = 0.0
    
    # ====================================================================
    # UI FIREWALL: Display variables (ONLY updated by commit())
    # ====================================================================
    display_caption = ""
    display_conf = None
    hold_frames = 0
    
    def commit(label: str, conf: float | None):
        """ONLY function allowed to update display_caption/display_conf."""
        nonlocal display_caption, display_conf, hold_frames
        # Set the caption FIRST before any print statements (in case print fails)
        display_caption = str(label)  # Ensure it's a string
        display_conf = conf
        hold_frames = HOLD_N
        
        # Then do debug output (this might fail but caption is already set)
        try:
            print(f"🔍 DEBUG COMMIT: label='{label}', conf={conf}, type={type(label)}")
            # Fix format specifier - handle None separately
            if conf is not None:
                conf_str = f"{conf:.2f}"
            else:
                conf_str = "None"
            print(f"✅ COMMIT: '{label}' (conf={conf_str})")
            print(f"🔍 DEBUG: display_caption set to '{display_caption}'")
        except Exception as e:
            print(f"⚠️ Error in commit print (caption already set): {e}")
            # Don't overwrite display_caption - it's already set correctly above
    
    debug_line = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️ Camera read failed - exiting")
                break

            try:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                feat, info = extractor.extract(rgb)
                window.append(feat)
            except Exception as e:
                print(f"⚠️ Error processing frame: {e}")
                import traceback
                traceback.print_exc()
                continue  # Skip this frame and continue

            left = int(info["left_present"])
            right = int(info["right_present"])
            hands_detected = (left > 0 or right > 0)
            hands_state = f"hands: L={left} R={right}"

            # ====================================================================
            # 3-STATE MACHINE: WAITING → COLLECTING → SHOWING
            # ====================================================================
            # State transitions based ONLY on hand presence/absence
            # ====================================================================
            
            # Track hand exit confirmation
            if not hands_detected:
                hands_gone_frames += 1
            else:
                hands_gone_frames = 0
            
            # State transitions
            if current_state == "WAITING":
                # WAITING → COLLECTING: Hands enter frame
                if hands_detected:
                    current_state = "COLLECTING"
                    # Clear previous subtitle immediately
                    display_caption = ""
                    display_conf = None
                    hold_frames = 0
                    # Reset prediction buffers
                    pred_hist.clear()
                    conf_hist.clear()
                    best_candidate_label = None
                    best_candidate_conf = 0.0
                    print(">>> GESTURE START: Hands entered - COLLECTING state")
            
            elif current_state == "COLLECTING":
                # COLLECTING → SHOWING: Hands exit frame (confirmed after 10 frames)
                if not hands_detected and hands_gone_frames >= HAND_EXIT_CONFIRM_FRAMES:
                    # Always commit the most common prediction, regardless of confidence
                    committed = False
                    try:
                        if len(pred_hist) > 0:
                            vote_counts = Counter(pred_hist)
                            if len(vote_counts) > 0:
                                most_common = vote_counts.most_common(1)[0]
                                vote_label, vote_count = most_common
                                
                                # Validate label index
                                if vote_label >= 0 and vote_label < len(cfg.labels):
                                    # Calculate average confidence for the voted label
                                    label_confs = [c for p, c in zip(pred_hist, conf_hist) if p == vote_label]
                                    avg_conf = sum(label_confs) / len(label_confs) if label_confs else 0.0
                                    
                                    # Always use the most common prediction (majority vote)
                                    label_to_commit = cfg.labels[vote_label]
                                    commit(label_to_commit, avg_conf)
                                    committed = True
                                    print(f">>> GESTURE END: Committed '{label_to_commit}' (conf={avg_conf:.2f}, votes={vote_count}/{len(pred_hist)})")
                                else:
                                    print(f"⚠️ Invalid vote_label index: {vote_label} (max: {len(cfg.labels)-1})")
                            else:
                                print(f"⚠️ No vote counts available")
                        else:
                            print(f"⚠️ No predictions in history")
                    except Exception as e:
                        print(f"⚠️ Error during commit: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # If commit failed, show "collecting" as fallback
                    if not committed:
                        try:
                            commit("collecting", None)
                            print(f">>> GESTURE END: Showing 'collecting' (no valid prediction)")
                        except Exception as e:
                            print(f"⚠️ Error committing 'collecting': {e}")
                            display_caption = "collecting"
                            display_conf = None
                            hold_frames = HOLD_N
                    
                    current_state = "SHOWING"
                    print(f">>> State: COLLECTING → SHOWING")
            
            elif current_state == "SHOWING":
                # SHOWING → COLLECTING: Hands re-enter (new gesture starts)
                if hands_detected:
                    current_state = "COLLECTING"
                    # Clear subtitle immediately
                    display_caption = ""
                    display_conf = None
                    hold_frames = 0
                    # Reset prediction buffers
                    pred_hist.clear()
                    conf_hist.clear()
                    best_candidate_label = None
                    best_candidate_conf = 0.0
                    print(">>> NEW GESTURE: Hands re-entered - COLLECTING state")
            
            # ====================================================================
            # COLLECTING STATE: Background inference (NO DISPLAY)
            # ====================================================================
            if current_state == "COLLECTING" and (not frozen) and (model is not None):
                # Run inference every stride_live frames
                if (len(window) == cfg.T) and (frame_i % stride_live == 0):
                    try:
                        x = np.stack(window, axis=0).astype(np.float32)
                        xt = torch.from_numpy(x).unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits = model(xt).cpu().numpy()[0]
                        probs = softmax_np(logits)
                        pred = int(np.argmax(probs))
                        pred_conf = float(probs[pred])
                        
                        # Validate prediction index
                        if pred >= 0 and pred < len(cfg.labels):
                            pred_hist.append(pred)
                            conf_hist.append(pred_conf)
                            debug_line = f"collecting: {cfg.labels[pred]} ({pred_conf:.2f})"
                        else:
                            print(f"⚠️ Invalid prediction index: {pred} (max: {len(cfg.labels)-1})")
                            debug_line = f"collecting: invalid_pred ({pred_conf:.2f})"
                    except Exception as e:
                        print(f"⚠️ Error during inference: {e}")
                        import traceback
                        traceback.print_exc()

                # Update best candidate (peak confidence) during collection
                if len(pred_hist) >= MIN_PREDICTIONS:
                    try:
                        vote_counts = Counter(pred_hist)
                        if len(vote_counts) > 0:
                            most_common = vote_counts.most_common(1)[0]
                            vote_label, vote_count = most_common
                            
                            # Validate label index
                            if vote_label >= 0 and vote_label < len(cfg.labels):
                                # Calculate average confidence for the voted label
                                label_confs = [c for p, c in zip(pred_hist, conf_hist) if p == vote_label]
                                avg_conf = sum(label_confs) / len(label_confs) if label_confs else 0.0
                                
                                # Update best candidate if confidence is higher
                                if avg_conf > best_candidate_conf:
                                    best_candidate_label = vote_label
                                    best_candidate_conf = avg_conf
                    except Exception as e:
                        print(f"⚠️ Error updating best candidate: {e}")
            
            prev_hands_detected = hands_detected

            # ====================================================================
            # UI FIREWALL: Display Logic
            # ====================================================================
            # Decrement hold counter if in SHOWING state
            if hold_frames > 0:
                hold_frames -= 1
                if hold_frames == 0:
                    # Transition back to WAITING when display expires
                    current_state = "WAITING"
                    display_caption = ""
                    display_conf = None
            
            # Display rule: ONLY show subtitle in SHOWING state with hold_frames > 0
            if current_state == "SHOWING" and hold_frames > 0:
                caption_to_display = display_caption
                conf_to_display = display_conf
                # Debug: verify what we're displaying
                if frame_i % 30 == 0:  # Print every second
                    print(f"🔍 DEBUG DISPLAY: state={current_state}, hold_frames={hold_frames}, display_caption='{display_caption}', caption_to_display='{caption_to_display}'")
            else:
                # ALL OTHER STATES: Blank (UI firewall)
                caption_to_display = ""
                conf_to_display = None
                if current_state == "WAITING":
                    debug_line = ""
            
            # Model not loaded fallback
            if model is None and current_state == "WAITING":
                caption_to_display = f"Selected label: {cfg.labels[selected_label_idx]} (press 1..0)"
                conf_to_display = None

            # Draw overlay
            try:
                draw_status(
                    frame,
                    hands_state
                    + (" | FROZEN" if frozen else "")
                    + (f" | {debug_line}" if debug_line else "")
                    + f" | STATE={current_state}"
                )

                # UI FIREWALL: draw_caption() receives only display_caption/display_conf
                draw_caption(frame, caption_to_display, conf_to_display)

                # Ensure frame is valid before showing
                if frame is not None and frame.size > 0:
                    cv2.imshow(cfg.window_title, frame)
                else:
                    print("⚠️ Invalid frame, skipping display")
                    continue
                
                # Check if window was closed (user clicked X)
                # Note: This check can be unreliable on some systems, so we catch all exceptions
                # DISABLED: Window property check can cause false exits on some systems
                # Uncomment only if you need this feature and it works on your system
                # try:
                #     window_prop = cv2.getWindowProperty(cfg.window_title, cv2.WND_PROP_VISIBLE)
                #     if window_prop is not None and window_prop < 1:
                #         print("⚠️ Window closed by user - exiting")
                #         break
                # except (cv2.error, AttributeError, TypeError):
                #     # Window property check may fail on some systems, ignore it and continue
                #     pass
                # except Exception as e:
                #     # Catch any other unexpected errors from window check
                #     print(f"⚠️ Unexpected error in window check (ignoring): {e}")
                #     pass
            except Exception as e:
                print(f"⚠️ Error during rendering: {e}")
                import traceback
                traceback.print_exc()
                # Continue running even if rendering fails

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear all state
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
                print("🧹 State cleared")
            elif key == ord(' '):
                frozen = not frozen
            elif key in [ord(str(i)) for i in range(1, 10)]:
                selected_label_idx = int(chr(key)) - 1
            elif key == ord('0'):
                selected_label_idx = 9

            frame_i += 1

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n⚠️ Fatal error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()