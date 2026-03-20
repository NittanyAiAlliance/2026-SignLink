import cv2
import numpy as np

# Modern Blue/Purple Color Scheme
COLOR_PRIMARY = (180, 100, 255)      # Purple (BGR)
COLOR_SECONDARY = (255, 140, 50)     # Blue (BGR)
COLOR_SUCCESS = (100, 200, 100)      # Green (BGR)
COLOR_WARNING = (80, 160, 255)       # Orange (BGR)
COLOR_TEXT = (255, 255, 255)         # White (BGR)
COLOR_TEXT_DIM = (200, 200, 200)     # Light Gray (BGR)

def draw_rounded_rect(img: np.ndarray, pt1: tuple, pt2: tuple, color: tuple,
                     thickness: int = -1, radius: int = 15, alpha: float = 0.85):
    """Draw a rounded rectangle with optional transparency."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Create overlay for transparency
    overlay = img.copy()

    # Draw the rounded rectangle on overlay
    # Draw rectangles for the main body
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    # Draw circles at corners
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)

    # Blend with original image for transparency
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_gradient_rect(img: np.ndarray, pt1: tuple, pt2: tuple,
                       color1: tuple, color2: tuple, alpha: float = 0.85):
    """Draw a rectangle with gradient fill and transparency."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Create gradient
    height = y2 - y1
    overlay = img.copy()

    for i in range(height):
        ratio = i / height
        # Interpolate between color1 and color2
        color = tuple(int(c1 * (1 - ratio) + c2 * ratio)
                     for c1, c2 in zip(color1, color2))
        cv2.line(overlay, (x1, y1 + i), (x2, y1 + i), color, 1)

    # Blend with original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_caption(frame_bgr: np.ndarray, caption: str, conf: float | None = None):
    """Draw caption with Netflix/YouTube professional styling at bottom center.

    Args:
        frame_bgr: Video frame in BGR format
        caption: Text to display
        conf: Confidence score (currently unused for clean Netflix/YouTube look)
    """
    # Note: conf parameter kept for API compatibility but hidden for professional look
    _ = conf  # Acknowledge parameter (unused for Netflix/YouTube style)

    # Don't draw anything if caption is empty
    if not caption or caption.strip() == "":
        return

    h, w = frame_bgr.shape[:2]

    text = f"{caption}"
    # Use sans-serif font for clean, professional look
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = h - 60  # Position from bottom

    # Netflix-style padding - subtle and balanced
    pad_x, pad_y = 25, 18
    box_x1 = max(0, x - pad_x)
    box_y1 = max(0, y - th - pad_y)
    box_x2 = min(w, x + tw + pad_x)
    box_y2 = min(h, y + pad_y)

    # Create overlay for drop shadow effect
    shadow_overlay = frame_bgr.copy()
    shadow_offset = 4
    draw_rounded_rect(shadow_overlay,
                     (box_x1 + shadow_offset, box_y1 + shadow_offset),
                     (box_x2 + shadow_offset, box_y2 + shadow_offset),
                     (0, 0, 0), -1, radius=8, alpha=0.3)

    # Professional semi-transparent black background (70% opacity)
    # Netflix/YouTube style - simple black, no gradient
    draw_rounded_rect(frame_bgr, (box_x1, box_y1), (box_x2, box_y2),
                     (0, 0, 0), -1, radius=8, alpha=0.7)

    # White sans-serif text - clean and readable
    # No border/outline for cleaner Netflix look
    cv2.putText(frame_bgr, text, (x, y),
               font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Optional: Subtle confidence indicator (Netflix/YouTube don't show this)
    # Uncomment if you want to display confidence, but it's hidden by default for clean look
    # if conf is not None and conf < 0.8:  # Only show if confidence is lower
    #     conf_text = f"{conf:.0%}"
    #     conf_font_scale = 0.6
    #     (ctw, cth), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX,
    #                                    conf_font_scale, 1)
    #
    #     # Small, subtle badge in top-right
    #     conf_x = w - ctw - 30
    #     conf_y = 30
    #
    #     # Very subtle, semi-transparent
    #     cv2.putText(frame_bgr, conf_text, (conf_x, conf_y),
    #                cv2.FONT_HERSHEY_SIMPLEX, conf_font_scale,
    #                (200, 200, 200), 1, cv2.LINE_AA)
    pass  # Confidence hidden for professional Netflix/YouTube look

def draw_status(frame_bgr: np.ndarray, status: str):
    """Draw status with modern styling and state-based coloring."""
    # Parse state from status string
    state = "WAITING"
    if "STATE=" in status:
        state_part = status.split("STATE=")[1].split("|")[0].split()[0]
        state = state_part

    # Color-code based on state
    state_colors = {
        "WAITING": (150, 150, 150),      # Gray
        "COLLECTING": COLOR_WARNING,      # Orange
        "SHOWING": COLOR_SUCCESS          # Green
    }
    state_color = state_colors.get(state, COLOR_TEXT_DIM)

    # Split status into components for better layout
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    thickness = 2

    # Draw state badge (top-left)
    state_text = f" {state} "
    (stw, sth), _ = cv2.getTextSize(state_text, font, font_scale, thickness)

    badge_x1 = 15
    badge_y1 = 15
    badge_x2 = badge_x1 + stw + 20
    badge_y2 = badge_y1 + sth + 15

    # Draw rounded state badge
    draw_rounded_rect(frame_bgr, (badge_x1, badge_y1), (badge_x2, badge_y2),
                     state_color, -1, radius=10, alpha=0.85)

    # State text
    cv2.putText(frame_bgr, state_text, (badge_x1 + 11, badge_y1 + sth + 6),
               font, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA)

    # Additional status info (if present)
    # Draw debug info in smaller text below state badge
    if "|" in status:
        debug_parts = status.split("|")
        debug_text = " | ".join(p.strip() for p in debug_parts[:-1])  # Exclude STATE part

        if debug_text.strip():
            debug_y = badge_y2 + 25
            # Semi-transparent background for debug text
            (dtw, dth), _ = cv2.getTextSize(debug_text, font, 0.6, 1)
            debug_bg_x1 = 15
            debug_bg_y1 = debug_y - dth - 5
            debug_bg_x2 = 15 + dtw + 20
            debug_bg_y2 = debug_y + 5

            draw_rounded_rect(frame_bgr, (debug_bg_x1, debug_bg_y1),
                            (debug_bg_x2, debug_bg_y2),
                            (50, 50, 50), -1, radius=8, alpha=0.7)

            cv2.putText(frame_bgr, debug_text, (20, debug_y),
                       font, 0.6, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
