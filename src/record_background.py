#!/usr/bin/env python3
"""
Record BACKGROUND samples (hands in frame, NOT signing).

This creates training data for the "background" class - when hands are
visible but the person is NOT performing a sign.

What to record as background:
  - Hands resting in frame
  - Hands moving randomly (not signing)
  - Transitioning between signs
  - Adjusting position
  - Any hand movement that is NOT one of your 7 signs

Controls:
  SPACE: Start/stop continuous recording
  R: Record single sample (30 frames)
  C: Cancel current recording
  Q: Quit
"""

import os
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2

from src.features_hands_pose import HandsPoseFeatureExtractor
from src.overlay import draw_status


# Settings
T = 30  # Frames per sample (same as your signs)
DATA_DIR = "data_pose_personal"
LABEL = "background"

# Target: Record at least 50-100 background samples
TARGET_SAMPLES = 70


def main():
    print("=" * 60)
    print("RECORD BACKGROUND SAMPLES")
    print("=" * 60)
    print()
    print("WHAT TO DO:")
    print("  - Keep hands IN FRAME")
    print("  - Move hands around randomly")
    print("  - DO NOT perform any of your 7 signs")
    print("  - Wave, gesture, scratch your head, etc.")
    print()
    print(f"Target: {TARGET_SAMPLES} samples")
    print(f"Saving to: {DATA_DIR}/{LABEL}/")
    print()
    print("CONTROLS:")
    print("  SPACE = Start/stop CONTINUOUS recording")
    print("  R     = Record single sample (30 frames)")
    print("  C     = Cancel")
    print("  Q     = Quit")
    print("=" * 60 + "\n")

    # Create directory
    save_dir = Path(DATA_DIR) / LABEL
    save_dir.mkdir(parents=True, exist_ok=True)

    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Initializing pose extractor...")
    extractor = HandsPoseFeatureExtractor()
    print("Ready!\n")

    # State
    continuous_mode = False
    recording = False
    window = []
    sample_count = len(list(save_dir.glob("*.npy")))
    countdown_until = 0.0

    def save_sample(arr):
        nonlocal sample_count
        fname = f"background_pose_{int(time.time()*1000)}.npy"
        path = save_dir / fname
        np.save(path, arr)
        sample_count += 1
        return path

    print(f"Existing samples: {sample_count}")
    print(f"Need: {max(0, TARGET_SAMPLES - sample_count)} more\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        feat, info = extractor.extract(rgb)

        left = int(info["left_present"])
        right = int(info["right_present"])
        pose = int(info.get("pose_present", 0))
        hands_detected = left > 0 or right > 0

        now = time.time()

        # Recording logic
        if continuous_mode and hands_detected:
            window.append(feat)
            if len(window) >= T:
                arr = np.stack(window[:T], axis=0).astype(np.float32)
                path = save_sample(arr)
                print(f"[{sample_count}/{TARGET_SAMPLES}] Saved: {path.name}")
                window = []  # Reset for next sample

        elif recording and hands_detected:
            window.append(feat)
            if len(window) >= T:
                arr = np.stack(window[:T], axis=0).astype(np.float32)
                path = save_sample(arr)
                print(f"[{sample_count}/{TARGET_SAMPLES}] Saved: {path.name}")
                recording = False
                window = []
                countdown_until = now + 0.5

        # Build status
        status = f"L={left} R={right} P={pose} | Samples: {sample_count}/{TARGET_SAMPLES}"

        # Build caption
        if now < countdown_until:
            caption = "Saved!"
            color = (0, 255, 0)
        elif continuous_mode:
            caption = f"CONTINUOUS RECORDING ({len(window)}/{T})"
            color = (0, 0, 255)
        elif recording:
            caption = f"Recording... ({len(window)}/{T})"
            color = (0, 165, 255)
        else:
            remaining = max(0, TARGET_SAMPLES - sample_count)
            if remaining == 0:
                caption = "TARGET REACHED! Press Q to finish"
                color = (0, 255, 0)
            else:
                caption = f"Press SPACE for continuous, R for single ({remaining} more needed)"
                color = (255, 255, 255)

        # Draw UI
        draw_status(frame, status)

        # Caption
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(caption, font, font_scale, thickness)
        x = (frame.shape[1] - text_w) // 2
        y = frame.shape[0] - 80
        cv2.rectangle(frame, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10), (0, 0, 0), -1)
        cv2.putText(frame, caption, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Title
        cv2.putText(frame, "BACKGROUND RECORDING", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Move hands randomly - DO NOT sign!", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Progress bar
        progress = min(sample_count / TARGET_SAMPLES, 1.0)
        bar_x, bar_y = 10, 80
        bar_w, bar_h = 300, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 255, 0), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        cv2.putText(frame, f"{int(progress*100)}%", (bar_x + bar_w + 10, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Recording indicator
        if continuous_mode or recording:
            cv2.circle(frame, (frame.shape[1] - 40, 40), 20, (0, 0, 255), -1)

        # Warning if no hands
        if not hands_detected and (continuous_mode or recording):
            cv2.putText(frame, "NO HANDS DETECTED!", (frame.shape[1]//2 - 150, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("Record Background", frame)
        key = cv2.waitKey(1) & 0xFF

        # Controls
        if key == ord('q'):
            break
        elif key == ord(' '):  # SPACE
            continuous_mode = not continuous_mode
            recording = False
            window = []
            if continuous_mode:
                print("Continuous recording STARTED - move hands randomly!")
            else:
                print("Continuous recording STOPPED")
        elif key == ord('r'):
            if not continuous_mode and not recording and now >= countdown_until:
                recording = True
                window = []
                print("Recording single sample...")
        elif key == ord('c'):
            recording = False
            window = []
            print("Cancelled")

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    print("\n" + "=" * 60)
    print("BACKGROUND RECORDING COMPLETE")
    print("=" * 60)
    print(f"Total background samples: {sample_count}")
    if sample_count >= TARGET_SAMPLES:
        print("Target reached! Ready for training.")
    else:
        print(f"Need {TARGET_SAMPLES - sample_count} more samples.")
    print("=" * 60)


if __name__ == "__main__":
    main()
