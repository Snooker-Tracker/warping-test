import cv2
import numpy as np


def display_combined(original, warped, text_info="", is_landscape=False):
    """
    Display original and warped images side by side or stacked.
    - is_landscape=False: side by side (horizontal)
    - is_landscape=True: stacked vertically (landscape)
    """
    # Scale down - smaller for landscape
    scale = 0.35 if is_landscape else 0.4
    original_scaled = cv2.resize(original, (int(original.shape[1] * scale), int(original.shape[0] * scale)))
    
    h, w = original_scaled.shape[:2]
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
        cv2.putText(combined, text_info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Display
    cv2.imshow("Original | Warped", combined)
    
    return combined


def draw_tracked_balls(warped, tracked):
    """Draw circles and IDs on warped image for tracked balls."""
    for obj_id, (x, y) in tracked.items():
        cv2.circle(warped, (x, y), 8, (0, 255, 255), 2)
        cv2.putText(
            warped,
            f"{obj_id}",
            (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )
    return warped
