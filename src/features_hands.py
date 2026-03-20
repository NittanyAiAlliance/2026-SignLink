import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

class HandsFeatureExtractor:
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
