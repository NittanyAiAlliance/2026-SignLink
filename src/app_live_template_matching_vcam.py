#!/usr/bin/env python3
"""
SignLink Virtual Camera - Template Matching (Keyframe-based)

Instead of neural network classification, this compares specific frame segments
against stored reference templates from your training data.

Compares 3 segments:
- Frames 1-5 (start of sign)
- Frames 13-18 (middle of sign)
- Frames 25-30 (end of sign)

If ALL segments match a sign with 90%+ similarity → show subtitle.
"""

import json
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from collections import deque

import pyvirtualcam

from src.overlay import draw_caption, draw_status


# Display labels
DISPLAY_LABELS = {
    "hello": "Hello",
    "how_are_you": "How are you?",
    "my_name_is": "My name is",
    "kat": "Kat",
    "nice_to_meet_you": "Nice to meet you",
    "welcome_to_signlink": "Welcome to SignLink",
    "thank_you": "Thank you",
}

LABELS = [
    "hello",
    "how_are_you",
    "my_name_is",
    "kat",
    "nice_to_meet_you",
    "welcome_to_signlink",
    "thank_you"
]


def format_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a_flat, b_flat) / (norm_a * norm_b)


def load_templates(data_dir, labels):
    """
    Load all training samples and create templates for each sign.

    Returns dict: {label: list of (start_segment, middle_segment, end_segment)}
    """
    templates = {label: [] for label in labels}

    for label in labels:
        label_dir = Path(data_dir) / label
        if not label_dir.exists():
            print(f"  {label}: directory not found")
            continue

        files = list(label_dir.glob("*.npy"))
        count = 0

        for f in files:
            data = np.load(f)
            if data.shape[0] >= 30 and data.shape[1] == 156:
                # Extract segments: start (0-5), middle (12-18), end (24-30)
                start_seg = data[0:5]      # Frames 1-5
                middle_seg = data[12:18]   # Frames 13-18
                end_seg = data[24:30]      # Frames 25-30

                templates[label].append({
                    'start': start_seg,
                    'middle': middle_seg,
                    'end': end_seg,
                    'full': data
                })
                count += 1

        print(f"  {label}: {count} templates loaded")

    return templates


def match_against_templates(window, templates, threshold=0.90):
    """
    Compare current window against all templates.

    Returns: (best_label, best_similarity, segment_scores) or (None, 0, None)
    """
    if len(window) < 30:
        return None, 0, None

    # Extract segments from current window
    window_arr = np.array(window)
    current_start = window_arr[0:5]
    current_middle = window_arr[12:18]
    current_end = window_arr[24:30]

    best_label = None
    best_avg_sim = 0
    best_scores = None

    for label, label_templates in templates.items():
        for template in label_templates:
            # Compare each segment
            start_sim = cosine_similarity(current_start, template['start'])
            middle_sim = cosine_similarity(current_middle, template['middle'])
            end_sim = cosine_similarity(current_end, template['end'])

            # All segments must exceed threshold
            if start_sim >= threshold and middle_sim >= threshold and end_sim >= threshold:
                avg_sim = (start_sim + middle_sim + end_sim) / 3

                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
                    best_label = label
                    best_scores = {
                        'start': start_sim,
                        'middle': middle_sim,
                        'end': end_sim,
                        'avg': avg_sim
                    }

    return best_label, best_avg_sim, best_scores


def compute_hand_velocity(feat_curr, feat_prev, left_present, right_present):
    """Compute velocity of hand landmarks between frames."""
    if feat_prev is None:
        return 0.0

    velocity = 0.0
    n_hands = 0

    if left_present > 0:
        left_vel = np.linalg.norm(feat_curr[0:63] - feat_prev[0:63])
        velocity += left_vel
        n_hands += 1

    if right_present > 0:
        right_vel = np.linalg.norm(feat_curr[63:126] - feat_prev[63:126])
        velocity += right_vel
        n_hands += 1

    if n_hands > 0:
        velocity /= n_hands

    return velocity


def main():
    print("=" * 70)
    print("SIGNLINK - TEMPLATE MATCHING (Keyframe-based)")
    print("=" * 70)

    # ==========================================================
    # SETTINGS
    # ==========================================================
    T = 30                          # Window size
    SIMILARITY_THRESHOLD = 0.90     # 90% similarity required
    MIN_VELOCITY = 0.01             # Minimum movement to consider
    HOLD_FRAMES = 60                # Show subtitle for 2 seconds

    camera_index = 0
    width, height = 1280, 720

    DATA_DIR = "data_pose_personal"

    # ==========================================================
    # LOAD TEMPLATES
    # ==========================================================
    print("\nLoading templates from training data...")
    templates = load_templates(DATA_DIR, LABELS)

    total_templates = sum(len(t) for t in templates.values())
    if total_templates == 0:
        print("\nERROR: No templates loaded!")
        return

    print(f"\nTotal: {total_templates} templates across {len(LABELS)} signs")

    # ==========================================================
    # FEATURE EXTRACTOR
    # ==========================================================
    print("\nInitializing feature extractor...")
    from src.features_hands_pose import HandsPoseFeatureExtractor
    extractor = HandsPoseFeatureExtractor()

    # ==========================================================
    # CAMERA
    # ==========================================================
    print(f"Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"Camera: {actual_w}x{actual_h} @ {fps}fps")

    # ==========================================================
    # STATE
    # ==========================================================
    window = deque(maxlen=T)
    velocity_history = deque(maxlen=10)

    prev_feat = None
    had_movement = False

    # Display state
    committed_caption = ""
    committed_conf = None
    committed_label = ""
    hold_counter = 0

    # Current match info
    current_match = None
    current_scores = None

    # Stats
    gesture_count = 0

    # ==========================================================
    # VIRTUAL CAMERA
    # ==========================================================
    print("\nInitializing virtual camera...")
    try:
        try:
            cam = pyvirtualcam.Camera(
                width=actual_w, height=actual_h, fps=fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
                backend=pyvirtualcam.Backend.OBS
            )
        except (AttributeError, ValueError):
            cam = pyvirtualcam.Camera(
                width=actual_w, height=actual_h, fps=fps,
                fmt=pyvirtualcam.PixelFormat.BGR
            )
        print(f"Virtual camera: {cam.device}")
    except Exception as e:
        print(f"ERROR: Virtual camera failed: {e}")
        cap.release()
        return

    print()
    print("=" * 70)
    print("TEMPLATE MATCHING:")
    print(f"  Similarity threshold: {SIMILARITY_THRESHOLD*100:.0f}%")
    print(f"  Segments compared:    Start (1-5), Middle (13-18), End (25-30)")
    print(f"  Templates loaded:     {total_templates}")
    print()
    print("HOW IT WORKS:")
    print("  1. Compares your live frames against stored training samples")
    print("  2. Checks START, MIDDLE, and END segments separately")
    print("  3. If ALL segments match 90%+ → shows subtitle")
    print()
    print("CONTROLS: Q=quit, C=clear")
    print("=" * 70 + "\n")

    # ==========================================================
    # MAIN LOOP
    # ==========================================================
    try:
        with cam:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Extract features
                feat, info = extractor.extract(rgb)
                window.append(feat)

                left = int(info["left_present"])
                right = int(info["right_present"])
                hands_detected = left > 0 or right > 0

                # ==================================================
                # COMPUTE VELOCITY
                # ==================================================
                velocity = compute_hand_velocity(
                    feat, prev_feat,
                    info["left_present"],
                    info["right_present"]
                )
                velocity_history.append(velocity)
                avg_velocity = np.mean(velocity_history) if len(velocity_history) > 0 else 0

                prev_feat = feat.copy() if hands_detected else None

                if avg_velocity > MIN_VELOCITY:
                    had_movement = True

                # ==================================================
                # HOLD COUNTER
                # ==================================================
                if hold_counter > 0:
                    hold_counter -= 1
                    if hold_counter == 0:
                        committed_caption = ""
                        committed_conf = None

                # ==================================================
                # TEMPLATE MATCHING
                # ==================================================
                current_match = None
                current_scores = None

                if hands_detected and len(window) >= T and had_movement:
                    match_label, match_sim, scores = match_against_templates(
                        list(window), templates, SIMILARITY_THRESHOLD
                    )

                    if match_label is not None:
                        current_match = match_label
                        current_scores = scores

                        # Check if it's a new sign
                        is_new_sign = match_label != committed_label

                        if is_new_sign:
                            gesture_count += 1
                            committed_caption = format_label(match_label)
                            committed_conf = match_sim
                            committed_label = match_label
                            hold_counter = HOLD_FRAMES
                            had_movement = False

                            print(f"\n[{gesture_count}] {committed_caption}")
                            print(f"    Start: {scores['start']:.1%}, Middle: {scores['middle']:.1%}, End: {scores['end']:.1%}")

                # ==================================================
                # DISPLAY
                # ==================================================
                # Status line
                status = f"L={left} R={right} | vel={avg_velocity:.3f} | frames={len(window)}/{T}"

                draw_status(frame, status)

                # Caption
                if committed_caption:
                    draw_caption(frame, committed_caption, committed_conf)

                # Border color
                if committed_caption and hold_counter > 0:
                    color = (0, 255, 0)      # Green - showing result
                elif current_match is not None:
                    color = (0, 255, 255)    # Yellow - match found
                elif hands_detected and had_movement:
                    color = (255, 200, 0)    # Blue - detecting
                elif hands_detected:
                    color = (150, 150, 150)  # Gray - hands still
                else:
                    color = (80, 80, 80)     # Dark gray - no hands

                cv2.rectangle(frame, (0, 0), (actual_w-1, actual_h-1), color, 6)

                # Segment similarity bars (when matching)
                if hands_detected and len(window) >= T:
                    bar_x = 20
                    bar_y = actual_h - 120
                    bar_width = 150
                    bar_height = 20

                    # Get current similarities even if not matching
                    window_arr = np.array(list(window))
                    current_start = window_arr[0:5]
                    current_middle = window_arr[12:18]
                    current_end = window_arr[24:30]

                    # Find best match for display
                    best_start, best_middle, best_end = 0, 0, 0
                    for label, label_templates in templates.items():
                        for template in label_templates:
                            s = cosine_similarity(current_start, template['start'])
                            m = cosine_similarity(current_middle, template['middle'])
                            e = cosine_similarity(current_end, template['end'])
                            if (s + m + e) / 3 > (best_start + best_middle + best_end) / 3:
                                best_start, best_middle, best_end = s, m, e

                    segments = [
                        ("Start (1-5)", best_start),
                        ("Middle (13-18)", best_middle),
                        ("End (25-30)", best_end),
                    ]

                    for i, (seg_name, sim) in enumerate(segments):
                        y = bar_y + i * 30

                        # Background
                        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height), (50, 50, 50), -1)

                        # Fill
                        fill_width = int(bar_width * sim)
                        if sim >= SIMILARITY_THRESHOLD:
                            bar_color = (0, 255, 0)  # Green - above threshold
                        else:
                            bar_color = (0, 165, 255)  # Orange - below threshold

                        cv2.rectangle(frame, (bar_x, y), (bar_x + fill_width, y + bar_height), bar_color, -1)

                        # Threshold marker
                        thresh_x = bar_x + int(bar_width * SIMILARITY_THRESHOLD)
                        cv2.line(frame, (thresh_x, y), (thresh_x, y + bar_height), (255, 255, 255), 2)

                        # Border
                        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height), (255, 255, 255), 1)

                        # Label
                        cv2.putText(frame, f"{seg_name}: {sim:.0%}",
                                   (bar_x + bar_width + 10, y + 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Movement indicator
                    if avg_velocity < MIN_VELOCITY:
                        cv2.putText(frame, "(waiting for movement)",
                                   (bar_x, bar_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

                cam.send(frame)
                cv2.imshow("SignLink Template Matching", frame)

                # ==================================================
                # KEY HANDLING
                # ==================================================
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('c'):
                    window.clear()
                    velocity_history.clear()
                    committed_caption = ""
                    committed_conf = None
                    committed_label = ""
                    hold_counter = 0
                    had_movement = False
                    prev_feat = None

                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nStopped. Recognized {gesture_count} signs.")


if __name__ == "__main__":
    main()
