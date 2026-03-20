import os
import sys
import time
from pathlib import Path

# Add parent directory to path to allow imports when running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2

from src.config import AppConfig
from src.features_hands import HandsFeatureExtractor
from src.overlay import draw_caption, draw_status

def main():
    cfg = AppConfig()
    os.makedirs("data", exist_ok=True)

    cap = cv2.VideoCapture(cfg.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    extractor = HandsFeatureExtractor()

    selected = 0
    recording = False
    window = []
    countdown_until = 0.0
    last_saved = ""

    def save_sample(label, arr):
        folder = os.path.join("data", label)
        os.makedirs(folder, exist_ok=True)
        fname = f"{label}_{int(time.time()*1000)}.npy"
        path = os.path.join(folder, fname)
        np.save(path, arr)
        return path

    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        feat, info = extractor.extract(rgb)

        # status text
        left = int(info["left_present"])
        right = int(info["right_present"])
        status = f"hands: L={left} R={right}"

        label = cfg.labels[selected]

        # recording logic
        now = time.time()
        if recording:
            window.append(feat)
            if len(window) >= cfg.T:
                arr = np.stack(window[:cfg.T], axis=0).astype(np.float32)  # (T,F)
                path = save_sample(label, arr)
                last_saved = path
                recording = False
                window = []
                countdown_until = now + 0.6  # tiny cool-down

        # UI caption
        if now < countdown_until:
            caption = "Saved ✅"
        elif recording:
            caption = f"REC ● {label}  ({len(window)}/{cfg.T})"
        else:
            caption = f"Selected: {label} | press R to record"

        draw_status(frame, status)
        draw_caption(frame, caption, None)

        if last_saved:
            cv2.putText(
                frame, os.path.basename(last_saved),
                (15, cfg.height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA
            )

        cv2.imshow("SignLink - Record Dataset", frame)
        key = cv2.waitKey(1) & 0xFF

        # controls
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not recording and now >= countdown_until:
                recording = True
                window = []
        elif key == ord('c'):
            window = []
            recording = False
        elif key in [ord(str(i)) for i in range(1, 10)]:
            selected = int(chr(key)) - 1
        elif key == ord('0'):
            selected = 9

        frame_i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
