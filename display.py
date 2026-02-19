"""Rendering helpers for the tracking UI."""

import cv2
import numpy as np

WINDOW_NAME = "Original | Warped"
_WINDOW_STATE = {"initialized": False, "last_size": (0, 0)}


def _ensure_window():
    if not _WINDOW_STATE["initialized"]:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        _WINDOW_STATE["initialized"] = True


def _resize_to_window(image):
    result = image
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
    except Exception:  # pylint: disable=broad-exception-caught
        win_w, win_h = 0, 0

    if win_w > 0 and win_h > 0:
        h, w = image.shape[:2]
        if abs(win_w - w) >= 2 or abs(win_h - h) >= 2:
            scale = min(1.0, win_w / w, win_h / h)
            if scale > 0:
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                if new_w == w and new_h == h:
                    if win_w > w + 2 or win_h > h + 2:
                        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                        x0 = max(0, (win_w - w) // 2)
                        y0 = max(0, (win_h - h) // 2)
                        canvas[y0 : y0 + h, x0 : x0 + w] = image
                        result = canvas
                else:
                    result = cv2.resize(
                        image, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )

    return result


def _window_image_size():
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
        return int(win_w), int(win_h)
    except Exception:  # pylint: disable=broad-exception-caught
        return 0, 0


def _adaptive_scale(original, warped, is_landscape, has_pocket_panel):
    default_scale = 0.35 if is_landscape else 0.4
    win_w, win_h = _window_image_size()
    if win_w <= 0 or win_h <= 0:
        return default_scale

    oh, ow = original.shape[:2]
    wh, ww = warped.shape[:2]
    if wh <= 0:
        return default_scale

    warped_ratio = ww / wh
    panel_w = 260 if has_pocket_panel else 0

    if is_landscape:
        # Vertically stacked; both panels share the same width.
        base_w = oh * warped_ratio
        base_h = 2.0 * oh
    else:
        # Side-by-side; same height.
        base_w = ow + (oh * warped_ratio)
        base_h = float(oh)

    usable_w = max(1.0, float(win_w - panel_w))
    scale_w = usable_w / max(1.0, base_w)
    scale_h = float(win_h) / max(1.0, base_h)
    scale = min(1.0, scale_w, scale_h)
    return max(0.1, scale)


def is_window_open():
    """Return True if the display window is still open."""
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _append_pocket_panel(image, pocket_info):
    """Append a right-side panel with per-pocket detected colors."""
    if not pocket_info:
        return image

    panel_w = 260
    panel = np.full((image.shape[0], panel_w, 3), 28, dtype=np.uint8)

    cv2.putText(
        panel,
        "Pocket Colors",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        2,
    )
    cv2.line(panel, (10, 38), (panel_w - 10, 38), (90, 90, 90), 1)

    color_map = {
        "cue": (255, 255, 255),
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 200, 0),
        "brown": (42, 42, 165),
        "blue": (255, 0, 0),
        "pink": (203, 192, 255),
        "black": (0, 0, 0),
        "none": (140, 140, 140),
        "unknown": (180, 180, 180),
    }

    lines = len(pocket_info)
    if lines == 0:
        return np.hstack([image, panel])

    step = max(24, min(40, (panel.shape[0] - 60) // lines))
    y = 62
    for idx, label in enumerate(pocket_info, 1):
        key = str(label).lower()
        draw_color = color_map.get(key, color_map["unknown"])
        text_label = "None" if key == "none" else str(label).capitalize()
        line_text = f"Pocket {idx}: {text_label}"

        cv2.putText(
            panel,
            line_text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            draw_color,
            2 if key == "black" else 1,
        )
        y += step

    return np.hstack([image, panel])


def display_combined(
    original, warped, text_info="", is_landscape=False, pocket_info=None
):
    """
    Display original and warped images side by side or stacked.
    - is_landscape=False: side by side (horizontal)
    - is_landscape=True: stacked vertically (landscape)
    """
    # Adapt scale to current window size to avoid fullscreen blur from heavy upscaling.
    scale = _adaptive_scale(
        original, warped, is_landscape, has_pocket_panel=bool(pocket_info)
    )
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

    combined = _append_pocket_panel(combined, pocket_info)

    _ensure_window()
    if _WINDOW_STATE["last_size"] == (0, 0):
        _WINDOW_STATE["last_size"] = combined.shape[:2][::-1]
        cv2.resizeWindow(
            WINDOW_NAME, _WINDOW_STATE["last_size"][0], _WINDOW_STATE["last_size"][1]
        )
    combined = _resize_to_window(combined)
    cv2.imshow(WINDOW_NAME, combined)

    return combined


def draw_tracked_balls(warped, tracked):
    """Draw circles and labels on warped image for tracked balls."""
    color_map = {
        "cue": (255, 255, 255),
        "red": (0, 0, 255),
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


def draw_pockets(warped, pockets):
    """Draw square markers for detected pockets."""
    h, w = warped.shape[:2]
    for idx, pocket in enumerate(pockets, 1):
        x = int(pocket["x"])
        y = int(pocket["y"])
        size = max(8, int(pocket["size"]))

        x1 = int(np.clip(x, 0, w - 1))
        y1 = int(np.clip(y, 0, h - 1))
        x2 = int(np.clip(x + size, 0, w - 1))
        y2 = int(np.clip(y + size, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            warped,
            f"P{idx}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 165, 255),
            1,
        )

    return warped
