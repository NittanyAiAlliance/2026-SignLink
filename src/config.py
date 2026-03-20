from dataclasses import dataclass

@dataclass
class AppConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    window_title: str = "SignLink - Live"

    T: int = 60
    stride: int = 3  # More frequent inference for faster response
    min_confidence: float = 0.60  # Lower threshold with better smoothing

    labels = [
        "Hello",
        "How are you",
        "My name is",
        "Kat",
        "Nice to meet you",
        "Welcome to signlink",
        "Thank you",
    ]
