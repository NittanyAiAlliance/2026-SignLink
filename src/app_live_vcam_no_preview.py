# src/app_live_vcam_no_preview.py - SignLink with Virtual Camera Output (NO PREVIEW)
# Works with WhatsApp, Zoom, Google Meet, Microsoft Teams, Discord, etc.
# No preview window - runs in background

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

from src.config import AppConfig
from src.features_hands import HandsFeatureExtractor
from src.overlay import draw_caption, draw_status
from src.model import SignGRU

def softmax_np(x: np.ndarray):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)

def main():
    cfg = AppConfig()

    print("=" * 70)
    print("🚀 SIGNLINK VIRTUAL CAMERA (NO PREVIEW MODE) - Starting...")
    print("=" * 70)

    # Camera setup
    print(f"📷 Opening camera {cfg.camera_index}...")
    cap = cv2.VideoCapture(cfg.camera_index)

    if not cap.isOpened():
        print(f"❌ ERROR: Could not open camera {cfg.camera_index}")
        print(f"   Make sure your camera is connected and not being used by another app")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    # Get actual camera dimensions
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    print(f"✅ Camera opened: {actual_width}x{actual_height} @ {fps}fps")

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
        print("\n" + "=" * 70)
        print("✅ SIGNLINK VIRTUAL CAMERA MODE (NO PREVIEW)")
        print("=" * 70)
        print("Virtual Camera: 'SignLink Camera'")
        print("Resolution: {}x{} @ {}fps".format(actual_width, actual_height, fps))
        print("\n📹 Select 'SignLink Camera' in your video call app:")
        print("   • WhatsApp Desktop")
        print("   • Zoom")
        print("   • Google Meet")
        print("   • Microsoft Teams")
        print("   • Discord")
        print("   • Any other video conferencing app")
        print("\n🎯 Do your gestures - subtitles will appear in the call!")
        print("⌨️  Press Ctrl+C to stop")
        print("=" * 70 + "\n")
    else:
        model = None
        print("⚠️ No model found")

    window = deque(maxlen=cfg.T)
    pred_hist = deque(maxlen=15)
    conf_hist = deque(maxlen=15)

    selected_label_idx = 0
    frozen = False
    frame_i = 0

    # Configuration
    HOLD_N = 60
    MIN_PREDICTIONS = 6
    commit_min_conf = 0.50
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
        print(f"✅ SHOWING: '{label}' (confidence: {conf:.2f})")

    debug_line = ""
    gesture_count = 0

    print("🎬 Starting virtual camera...")
    print("   Press Ctrl+C to stop\n")
    print(f"📹 Camera dimensions: {actual_width}x{actual_height} @ {fps}fps")

    # ====================================================================
    # VIRTUAL CAMERA INITIALIZATION
    # ====================================================================
    print("\n" + "=" * 70)
    print("🔄 Initializing virtual camera...")
    print("=" * 70)
    print(f"   Resolution: {actual_width}x{actual_height} @ {fps}fps")
    print(f"   Backend: OBS Virtual Camera (macOS)")
    print()

    try:
        # Try to create virtual camera with OBS backend
        print("   Creating virtual camera device...")
        print("   Trying OBS Virtual Camera backend...")

        # Try with OBS backend first (most common on macOS)
        try:
            cam = pyvirtualcam.Camera(
                width=actual_width,
                height=actual_height,
                fps=fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
                backend=pyvirtualcam.Backend.OBS
            )
        except (AttributeError, ValueError):
            # If backend parameter doesn't work, try without it
            print("   Backend parameter not supported, trying default...")
            cam = pyvirtualcam.Camera(
                width=actual_width,
                height=actual_height,
                fps=fps,
                fmt=pyvirtualcam.PixelFormat.BGR
            )

        print(f"✅ Virtual camera created successfully!")
        print(f"   Device: {cam.device}")
        print(f"   Camera name: 'SignLink Camera' or '{cam.device}'")
        print(f"   Resolution: {actual_width}x{actual_height}")
        print(f"\n📹 Next steps:")
        print(f"   1. Open your video call app (WhatsApp, Zoom, Meet, Teams, etc.)")
        print(f"   2. Go to camera settings")
        print(f"   3. Select 'SignLink Camera' or '{cam.device}'")
        print(f"   4. Start signing - subtitles will appear in the call!")
        print(f"\n⚠️  NO PREVIEW MODE - Running in background")
        print(f"   Check your video call app to see the output")
        print("=" * 70 + "\n")
    except Exception as cam_error:
        print(f"\n❌ ERROR: Failed to create virtual camera")
        print(f"   Error: {cam_error}")
        print(f"\n💡 SOLUTION:")
        print(f"   On macOS, pyvirtualcam requires OBS Virtual Camera to be STARTED.")
        print(f"   OBS is installed, but you need to start the Virtual Camera:")
        print(f"\n   Steps:")
        print(f"   1. Open OBS Studio (it's already installed)")
        print(f"   2. In OBS, go to: Tools → Start Virtual Camera")
        print(f"   3. Wait for 'Virtual Camera Active' message")
        print(f"   4. Then run this script again: python src/app_live_vcam_no_preview.py")
        print(f"\n   OR use the regular app_live.py for local testing:")
        print(f"     python src/app_live.py")
        print("=" * 70 + "\n")
        cap.release()
        return

    # Main loop - runs when virtual camera is successfully created
    try:
        with cam:
            print("🎥 Virtual camera is running...")
            print("   Gesture recognition is active")
            print("   Waiting for hands...\n")

            while True:
                ok, frame = cap.read()
                if not ok:
                    print("⚠️ Camera read failed")
                    break

                # Mirror for natural signing
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                feat, info = extractor.extract(rgb)
                window.append(feat)

                left = int(info["left_present"])
                right = int(info["right_present"])
                hands_detected = (left > 0 or right > 0)
                hands_state = f"hands: L={left} R={right}"

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
                        print(f"\n🟢 GESTURE {gesture_count} STARTED")

                elif current_state == "COLLECTING":
                    if not hands_detected and hands_gone_frames >= HAND_EXIT_CONFIRM_FRAMES:
                        if len(pred_hist) >= MIN_PREDICTIONS:
                            vote_counts = Counter(pred_hist)
                            most_common = vote_counts.most_common(1)[0]
                            vote_label_idx, vote_count = most_common

                            label_confs = [c for p, c in zip(pred_hist, conf_hist) if p == vote_label_idx]
                            avg_conf = sum(label_confs) / len(label_confs) if label_confs else 0.0

                            predicted_label = cfg.labels[vote_label_idx]

                            commit(predicted_label, avg_conf)
                            current_state = "SHOWING"
                            print(f"🔴 GESTURE {gesture_count} ENDED")
                            print(f"   → Prediction: '{predicted_label}' ({avg_conf:.2f})")
                        else:
                            print(f"⚠️  Not enough predictions")
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
                        print(f"\n🟢 GESTURE {gesture_count} STARTED")

                # ============================================================
                # INFERENCE
                # ============================================================
                if current_state == "COLLECTING" and (not frozen) and (model is not None):
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

                        debug_line = f"collecting: {cfg.labels[pred]} ({pred_conf:.2f})"

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
                        print("⏰ Subtitle cleared\n")

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

                # ============================================================
                # SEND TO VIRTUAL CAMERA (No preview window!)
                # ============================================================
                cam.send(frame)

                # NO PREVIEW WINDOW - That's the difference!
                # The video goes directly to virtual camera for video calls

                frame_i += 1

                # Wait to maintain fps
                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        print("\n\n⚠️ Stopping virtual camera...")
        print(f"   Total gestures performed: {gesture_count}")
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        print("✅ Virtual camera stopped\n")

if __name__ == "__main__":
    main()
