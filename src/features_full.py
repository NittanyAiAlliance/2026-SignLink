"""
Full Feature Extractor: Hands + Pose + Face Landmarks.

This extractor combines:
- Hand landmarks (63 features per hand = 126 total)
- Upper body pose landmarks (27 features)
- Face landmarks for touch detection (36 features)
- Presence flags (4 total)

Total: 193 features per frame

The face features help distinguish signs that touch different
parts of the face (e.g., mother=chin vs father=forehead).
"""

import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

# Upper body pose landmarks
POSE_LANDMARKS = [
    0,   # Nose
    11,  # Left shoulder
    12,  # Right shoulder
    13,  # Left elbow
    14,  # Right elbow
    15,  # Left wrist
    16,  # Right wrist
    23,  # Left hip
    24,  # Right hip
]

# Key face landmarks for sign language (touch detection)
# These cover forehead, chin, cheeks, nose, mouth - areas commonly touched in ASL
FACE_LANDMARKS = [
    10,   # Forehead top center
    151,  # Forehead center (for "father")
    9,    # Nose bridge
    4,    # Nose tip
    152,  # Chin center (for "mother")
    175,  # Chin bottom
    234,  # Left cheek outer
    454,  # Right cheek outer
    116,  # Left cheek inner
    345,  # Right cheek inner
    0,    # Upper lip center
    17,   # Lower lip center
]

# Feature dimensions:
# - Left hand: 21 * 3 = 63
# - Right hand: 21 * 3 = 63
# - Pose: 9 * 3 = 27
# - Face: 12 * 3 = 36
# - Flags: 4 (left_hand, right_hand, pose, face)
# Total: 63 + 63 + 27 + 36 + 4 = 193


class FullFeatureExtractor:
    """
    Extracts hand + pose + face landmarks for sign language recognition.

    Face landmarks help distinguish:
    - mother (touches chin) vs father (touches forehead)
    - Signs that touch cheeks, nose, mouth
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_det_conf: float = 0.5,
        min_track_conf: float = 0.5,
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
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

        # Initialize MediaPipe Face Mesh
        self.face = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
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

    @staticmethod
    def _face_to_vec(face_landmarks) -> np.ndarray:
        """Extract key face landmarks (36 features)."""
        vec = []
        for idx in FACE_LANDMARKS:
            lm = face_landmarks.landmark[idx]
            vec.extend([lm.x, lm.y, lm.z])
        return np.array(vec, dtype=np.float32)

    def extract(self, rgb_frame: np.ndarray):
        """
        Extract combined hand + pose + face features from an RGB frame.

        Args:
            rgb_frame: RGB image as numpy array (H, W, 3)

        Returns:
            feat: Feature vector of shape (193,)
            info: Dictionary with detection metadata
        """
        # Process all three
        hands_result = self.hands.process(rgb_frame)
        pose_result = self.pose.process(rgb_frame)
        face_result = self.face.process(rgb_frame)

        # Initialize feature vectors
        left_hand_vec = np.zeros((63,), dtype=np.float32)
        right_hand_vec = np.zeros((63,), dtype=np.float32)
        pose_vec = np.zeros((27,), dtype=np.float32)
        face_vec = np.zeros((36,), dtype=np.float32)

        left_present = 0.0
        right_present = 0.0
        pose_present = 0.0
        face_present = 0.0

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

        # Extract face features
        if face_result.multi_face_landmarks:
            face_vec = self._face_to_vec(face_result.multi_face_landmarks[0])
            face_present = 1.0

        # Concatenate all features
        feat = np.concatenate([
            left_hand_vec,      # 63 features
            right_hand_vec,     # 63 features
            pose_vec,           # 27 features
            face_vec,           # 36 features
            np.array([left_present, right_present, pose_present, face_present], dtype=np.float32)
        ])

        info = {
            "left_present": left_present,
            "right_present": right_present,
            "pose_present": pose_present,
            "face_present": face_present,
        }

        return feat, info

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
        self.pose.close()
        self.face.close()
