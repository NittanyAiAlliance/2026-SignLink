#!/usr/bin/env python3
"""
Record dataset with POSE features (hands + body) - 156 dimensions.

This creates training data with body pose features for better accuracy.
Data is saved to data_pose_personal/ directory.

Controls:
  1-7: Select phrase
  R: Start recording
  C: Cancel recording
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
from src.overlay import draw_caption, draw_status


# Your 7 personal phrases
LABELS = [
    "hello",
    "how_are_you",
    "my_name_is",
    "kat",
    "nice_to_meet_you",
    "welcome_to_signlink",
    "thank_you"
]

DISPLAY_LABELS = {
    "hello": "Hello",
    "how_are_you": "How are you?",
    "my_name_is": "My name is",
    "kat": "Kat",
    "nice_to_meet_you": "Nice to meet you",
    "welcome_to_signlink": "Welcome to SignLink",
    "thank_you": "Thank you",
}

# Recording settings
T = 30  # Frames per sample (1 second at 30fps)
DATA_DIR = "data_pose_personal"


def main():
    print("=" * 60)
    print("RECORD DATASET WITH POSE FEATURES (156-dim)")
    print("=" * 60)
    print(f"\nSaving to: {DATA_DIR}/")
    print(f"Frames per sample: {T}")
    print("\nPhrases:")
    for i, label in enumerate(LABELS):
        print(f"  {i+1}. {DISPLAY_LABELS.get(label, label)}")
    print("\nControls: 1-7=select, R=record, C=cancel, Q=quit")
    print("=" * 60 + "\n")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Use POSE feature extractor (156 dimensions)
    print("Initializing pose extractor...")
    extractor = HandsPoseFeatureExtractor()
    
    selected = 0
    recording = False
    window = []
    countdown_until = 0.0
    last_saved = ""

    def save_sample(label, arr):
        folder = os.path.join(DATA_DIR, label)
        os.makedirs(folder, exist_ok=True)
        fname = f"{label}_pose_{int(time.time()*1000)}.npy"
        path = os.path.join(folder, fname)
        np.save(path, arr)
        return path

    def get_sample_count(label):
        folder = Path(DATA_DIR) / label
        if folder.exists():
            return len(list(folder.glob("*.npy")))
        return 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        feat, info = extractor.extract(rgb)

        # Status text with pose info
        left = int(info["left_present"])
        right = int(info["right_present"])
        pose = int(info.get("pose_present", 0))
        status = f"L={left} R={right} P={pose}"

        label = LABELS[selected]
        display_label = DISPLAY_LABELS.get(label, label)
        sample_count = get_sample_count(label)

        # Recording logic
        now = time.time()
        if recording:
            window.append(feat)
            if len(window) >= T:
                arr = np.stack(window[:T], axis=0).astype(np.float32)
                path = save_sample(label, arr)
                last_saved = path
                recording = False
                window = []
                countdown_until = now + 0.6
                print(f"Saved: {path} (shape: {arr.shape})")

        # UI caption
        if now < countdown_until:
            caption = "Saved!"
            color = (0, 255, 0)
        elif recording:
            caption = f"REC: {display_label} ({len(window)}/{T})"
            color = (0, 0, 255)
        else:
            caption = f"[{selected+1}] {display_label} ({sample_count} samples)"
            color = (255, 255, 255)

        # Draw UI
        draw_status(frame, status)

        # Draw caption with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(caption, font, font_scale, thickness)

        x = (frame.shape[1] - text_w) // 2
        y = frame.shape[0] - 80

        cv2.rectangle(frame, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10), (0, 0, 0), -1)
        cv2.putText(frame, caption, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Show feature dimension
        cv2.putText(frame, f"Features: 156-dim (pose)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Recording indicator
        if recording:
            cv2.circle(frame, (frame.shape[1] - 40, 40), 20, (0, 0, 255), -1)

        cv2.imshow("Record Pose Dataset", frame)
        key = cv2.waitKey(1) & 0xFF

        # Controls
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not recording and now >= countdown_until:
                recording = True
                window = []
                print(f"Recording: {display_label}...")
        elif key == ord('c'):
            window = []
            recording = False
            print("Cancelled")
        elif key in [ord(str(i)) for i in range(1, 8)]:
            idx = int(chr(key)) - 1
            if idx < len(LABELS):
                selected = idx
                print(f"Selected: {DISPLAY_LABELS.get(LABELS[selected], LABELS[selected])}")

    cap.release()
    cv2.destroyAllWindows()

    # Print summary
    print("\n" + "=" * 60)
    print("RECORDING SUMMARY")
    print("=" * 60)
    for label in LABELS:
        count = get_sample_count(label)
        print(f"  {DISPLAY_LABELS.get(label, label)}: {count} samples")
    print("=" * 60)


if __name__ == "__main__":
    main()
