"""Rendering helpers for the tracking UI."""

import cv2
import numpy as np

WINDOW_NAME = "Original | Warped"
_WINDOW_INITIALIZED = False


def _ensure_window():
    global _WINDOW_INITIALIZED
    if not _WINDOW_INITIALIZED:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        _WINDOW_INITIALIZED = True


def _resize_to_window(image):
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
    except cv2.error:
        return image

    if win_w <= 0 or win_h <= 0:
        return image

    h, w = image.shape[:2]
    scale = min(win_w / w, win_h / h)
    if scale <= 0:
        return image

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    if new_w == w and new_h == h:
        return image
    return cv2.resize(image, (new_w, new_h))


def display_combined(original, warped, text_info="", is_landscape=False):
    """
    Display original and warped images side by side or stacked.
    - is_landscape=False: side by side (horizontal)
    - is_landscape=True: stacked vertically (landscape)
    """
    # Scale down - smaller for landscape
    scale = 0.35 if is_landscape else 0.4
    original_scaled = cv2.resize(
        original, (int(original.shape[1] * scale), int(original.shape[0] * scale))
    )

    h, _ = original_scaled.shape[:2]
    wh, ww = warped.shape[:2]

    if is_landscape:
        # Stack vertically - make widths match
        aspect_ratio = ww / wh
        new_w = int(h * aspect_ratio)
        warped_resized = cv2.resize(warped, (new_w, h))

        # Resize original to match warped width
        original_scaled = cv2.resize(original_scaled, (new_w, h))
        combined = np.vstack([original_scaled, warped_resized])
    else:
        # Stack horizontally - make heights match
        aspect_ratio = ww / wh
        new_w = int(h * aspect_ratio)
        warped_resized = cv2.resize(warped, (new_w, h))
        combined = np.hstack([original_scaled, warped_resized])

    # Add info text
    if text_info:
        cv2.putText(
            combined, text_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
        )

    _ensure_window()
    combined = _resize_to_window(combined)
    cv2.imshow(WINDOW_NAME, combined)

    return combined


def draw_tracked_balls(warped, tracked):
    """Draw circles and labels on warped image for tracked balls."""
    color_map = {
        "cue": (255, 255, 255),
        "yellow": (0, 255, 255),
        "green": (0, 200, 0),
        "brown": (42, 42, 165),
        "blue": (255, 0, 0),
        "pink": (203, 192, 255),
        "black": (0, 0, 0),
        "unknown": (200, 200, 200),
    }

    for label, pos in tracked.items():
        if len(pos) == 3:
            x, y, r = pos
        else:
            x, y = pos
            r = 8

        color = color_map.get(label, color_map["unknown"])
        outline = (255, 255, 255) if label == "black" else color
        if label == "cue":
            outline = (0, 0, 0)

        cv2.circle(warped, (x, y), r, outline, 2)
        cv2.putText(
            warped,
            f"{label}",
            (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
    return warped
