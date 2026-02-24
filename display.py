"""Rendering helpers for the tracking UI."""

import cv2
import numpy as np

WINDOW_NAME = "Original | Warped"
_WINDOW_STATE = {"initialized": False, "last_size": (0, 0)}
INITIAL_WINDOW_SCALE = 1.5
POCKET_PANEL_WIDTH = 320
POCKET_ROW_STEP = 24
SCORE_VISIBLE_ENTRIES = 15
SCORE_ROW_H = 18


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
            # Keep aspect ratio in both windowed and fullscreen modes.
            scale = min(win_w / w, win_h / h)
            if scale > 0:
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
                resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

                if new_w != win_w or new_h != win_h:
                    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    x0 = max(0, (win_w - new_w) // 2)
                    y0 = max(0, (win_h - new_h) // 2)
                    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
                    result = canvas
                else:
                    result = resized

    return result


def _fixed_scale(is_landscape):
    """Keep display scale stable regardless of window/fullscreen size."""
    return 0.35 if is_landscape else 0.4


def is_window_open():
    """Return True if the display window is still open."""
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _append_pocket_panel(image, pocket_info, pocket_log):
    """Append a right-side panel with per-pocket state."""
    if not pocket_info and not pocket_log:
        return image

    panel = np.full((image.shape[0], POCKET_PANEL_WIDTH, 3), 28, dtype=np.uint8)

    cv2.putText(
        panel,
        "Pocket State",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        2,
    )
    cv2.line(panel, (10, 38), (POCKET_PANEL_WIDTH - 10, 38), (90, 90, 90), 1)

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
        "ball": (0, 200, 255),
    }

    step = POCKET_ROW_STEP
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
            0.5,
            draw_color,
            2 if key == "black" else 1,
        )
        y += step

    return np.hstack([image, panel])


def _ensure_min_panel_height(image, pocket_info):
    """Pad image height so pocket panel can show all pocket rows."""
    pocket_count = len(pocket_info or [])
    if pocket_count <= 0:
        return image

    required_h = 70 + (pocket_count * POCKET_ROW_STEP) + 12
    current_h, current_w = image.shape[:2]
    if current_h >= required_h:
        return image

    padded = np.zeros((required_h, current_w, 3), dtype=np.uint8)
    padded[:current_h, :current_w] = image
    return padded


def _append_scoring_record_bottom(image, pocket_log):
    """Append a bottom strip for scoring record history."""
    logs = list(pocket_log or [])
    visible_logs = logs[:SCORE_VISIBLE_ENTRIES]
    score_panel_h = 32 + (SCORE_VISIBLE_ENTRIES * SCORE_ROW_H)

    score_panel = np.full((score_panel_h, image.shape[1], 3), 18, dtype=np.uint8)
    cv2.line(score_panel, (0, 0), (image.shape[1] - 1, 0), (85, 85, 85), 1)

    cv2.putText(
        score_panel,
        "Scoring Record",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
    )

    if visible_logs:
        y = 40
        for entry in visible_logs:
            cv2.putText(
                score_panel,
                str(entry),
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (170, 170, 170),
                1,
            )
            y += SCORE_ROW_H
    else:
        cv2.putText(
            score_panel,
            "No scores yet",
            (12, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (170, 170, 170),
            1,
        )

    return np.vstack([image, score_panel])


def display_combined(
    _original, warped, text_info="", is_landscape=False, panel_data=None
):
    """
    Display only the warped image.
    - _original is kept for call-site compatibility.
    """
    panel_data = panel_data or {}
    pocket_info = panel_data.get("pocket_info")
    pocket_log = panel_data.get("pocket_log")

    # Keep render scale fixed so fullscreen does not change effective resolution.
    scale = _fixed_scale(is_landscape)
    combined = cv2.resize(
        warped, (int(warped.shape[1] * scale), int(warped.shape[0] * scale))
    )

    combined = _ensure_min_panel_height(combined, pocket_info)
    combined = _append_pocket_panel(combined, pocket_info, pocket_log)
    combined = _append_scoring_record_bottom(combined, pocket_log)

    # Place status text at top-right of the pocket panel area.
    if text_info:
        parts = [part.strip() for part in str(text_info).split("|") if part.strip()]
        if not parts:
            parts = [str(text_info)]

        has_panel = bool(pocket_info or pocket_log)
        if has_panel:
            panel_x0 = combined.shape[1] - POCKET_PANEL_WIDTH
            right_edge = combined.shape[1] - 12
            min_x = panel_x0 + 12
        else:
            right_edge = combined.shape[1] - 12
            min_x = 10

        y = 25
        for part in parts:
            (text_w, _), _ = cv2.getTextSize(part, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            x = max(min_x, right_edge - text_w)
            cv2.putText(
                combined,
                part,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
            )
            y += 22

    _ensure_window()
    if _WINDOW_STATE["last_size"] == (0, 0):
        base_w, base_h = combined.shape[:2][::-1]
        _WINDOW_STATE["last_size"] = (
            int(base_w * INITIAL_WINDOW_SCALE),
            int(base_h * INITIAL_WINDOW_SCALE),
        )
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

        base_label = label.split("_", 1)[0]
        color = color_map.get(base_label, color_map["unknown"])
        outline = (255, 255, 255) if base_label == "black" else color
        if base_label == "cue":
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
