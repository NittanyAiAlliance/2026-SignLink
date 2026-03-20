# src/config_msasl.py
# Configuration for MS-ASL dataset (top 10 signs with most samples)
# This is SEPARATE from your personal demo config!

from dataclasses import dataclass

@dataclass
class MSASLConfig:
    """Configuration for MS-ASL word recognition model"""

    # Camera settings (same as personal demo)
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    window_title: str = "SignLink - MS-ASL Word Recognition"

    # Model inference settings
    T: int = 60  # Window size for predictions
    stride: int = 3
    min_confidence: float = 0.60

    # MS-ASL vocabulary (top 10 signs with most samples)
    labels = [
        "brother",
        "family",
        "father",
        "friend",
        "hello",
        "mother",
        "school",
        "teacher",
        "tired",
        "yes",
    ]

