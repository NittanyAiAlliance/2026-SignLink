"""
Enhanced Feature Extractor with Hand + Pose Landmarks.

This extractor combines:
- Hand landmarks (63 features per hand = 126 total)
- Upper body pose landmarks (27 features)
- Presence flags (3 total)

Total: 156 features per frame

The pose features capture arm position and body orientation,
which helps distinguish similar signs like family/brother/father/mother.
"""

import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


# Upper body landmarks we care about (indices in MediaPipe Pose)
POSE_LANDMARKS = [
    0,   # Nose (face reference)
    11,  # Left shoulder
    12,  # Right shoulder
    13,  # Left elbow
    14,  # Right elbow
    15,  # Left wrist
    16,  # Right wrist
    23,  # Left hip (torso reference)
    24,  # Right hip (torso reference)
]

# Total features breakdown:
# - Left hand: 21 landmarks * 3 coords = 63
# - Right hand: 21 landmarks * 3 coords = 63
# - Pose: 9 landmarks * 3 coords = 27
# - Presence flags: 3 (left_hand, right_hand, pose)
# Total: 63 + 63 + 27 + 3 = 156 features


class HandsPoseFeatureExtractor:
    """
    Extracts hand landmarks + upper body pose for sign language recognition.

    Features:
    - Hand landmarks capture finger positions and hand shapes
    - Pose landmarks capture arm position relative to body
    - This combination helps distinguish signs that differ in arm movement
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_det_conf: float = 0.5,
        min_track_conf: float = 0.5,
        pose_min_det_conf: float = 0.5,
        pose_min_track_conf: float = 0.5,
    ):
        # Initialize MediaPipe Hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

        # Initialize MediaPipe Pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=pose_min_det_conf,
            min_tracking_confidence=pose_min_track_conf,
        )

    @staticmethod
    def _hand_to_vec(hand_landmarks) -> np.ndarray:
        """Convert hand landmarks to flat vector (63 features)."""
        vec = []
        for lm in hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
        return np.array(vec, dtype=np.float32)

    @staticmethod
    def _pose_to_vec(pose_landmarks) -> np.ndarray:
        """Extract upper body pose landmarks (27 features)."""
        vec = []
        for idx in POSE_LANDMARKS:
            lm = pose_landmarks.landmark[idx]
            vec.extend([lm.x, lm.y, lm.z])
        return np.array(vec, dtype=np.float32)

    def extract(self, rgb_frame: np.ndarray):
        """
        Extract combined hand + pose features from an RGB frame.

        Args:
            rgb_frame: RGB image as numpy array (H, W, 3)

        Returns:
            feat: Feature vector of shape (156,)
            info: Dictionary with detection metadata
        """
        # Process hands
        hands_result = self.hands.process(rgb_frame)

        # Process pose
        pose_result = self.pose.process(rgb_frame)

        # Initialize feature vectors
        left_hand_vec = np.zeros((63,), dtype=np.float32)
        right_hand_vec = np.zeros((63,), dtype=np.float32)
        pose_vec = np.zeros((27,), dtype=np.float32)

        left_present = 0.0
        right_present = 0.0
        pose_present = 0.0

        # Extract hand features
        if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
            for lm, handed in zip(
                hands_result.multi_hand_landmarks,
                hands_result.multi_handedness
            ):
                label = handed.classification[0].label
                if label == "Left":
                    left_hand_vec = self._hand_to_vec(lm)
                    left_present = 1.0
                elif label == "Right":
                    right_hand_vec = self._hand_to_vec(lm)
                    right_present = 1.0

        # Extract pose features
        if pose_result.pose_landmarks:
            pose_vec = self._pose_to_vec(pose_result.pose_landmarks)
            pose_present = 1.0

        # Concatenate all features
        feat = np.concatenate([
            left_hand_vec,      # 63 features
            right_hand_vec,     # 63 features
            pose_vec,           # 27 features
            np.array([left_present, right_present, pose_present], dtype=np.float32)  # 3 flags
        ])

        info = {
            "left_present": left_present,
            "right_present": right_present,
            "pose_present": pose_present,
            "hands_raw": hands_result,
            "pose_raw": pose_result,
        }

        return feat, info

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
        self.pose.close()


# For backward compatibility, also provide the old interface
class HandsFeatureExtractor:
    """
    Original hand-only extractor (128 features).
    Kept for backward compatibility with existing models.
    """

    def __init__(self, max_num_hands=2, min_det_conf=0.5, min_track_conf=0.5):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

    @staticmethod
    def _hand_to_vec(hand_landmarks) -> np.ndarray:
        vec = []
        for lm in hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
        return np.array(vec, dtype=np.float32)

    def extract(self, rgb_frame: np.ndarray):
        res = self.hands.process(rgb_frame)

        left_vec = np.zeros((63,), dtype=np.float32)
        right_vec = np.zeros((63,), dtype=np.float32)
        left_present = 0.0
        right_present = 0.0

        if res.multi_hand_landmarks and res.multi_handedness:
            for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handed.classification[0].label
                if label == "Left":
                    left_vec = self._hand_to_vec(lm)
                    left_present = 1.0
                elif label == "Right":
                    right_vec = self._hand_to_vec(lm)
                    right_present = 1.0

        feat = np.concatenate([
            left_vec,
            right_vec,
            np.array([left_present, right_present], dtype=np.float32)
        ])
        info = {
            "left_present": left_present,
            "right_present": right_present,
            "raw": res,
        }
        return feat, info
