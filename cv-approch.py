import cv2
import numpy as np
import math
from collections import OrderedDict
import os

TABLE_W = 280
TABLE_H = 560

# Global flag for abort
abort_flag = False

# ============================
# UTILS
# ============================


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


# ============================
# AUTOMATIC TABLE DETECTION
# ============================


def detect_table_corners(frame):
    """
    Detect table corners using contour detection.
    Works with rotated/skewed tables by finding the quadrilateral that best fits the table.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    table_contour = max(contours, key=cv2.contourArea)

    # Use Ramer-Douglas-Peucker algorithm to find 4 corners
    epsilon = 0.02 * cv2.arcLength(table_contour, True)
    approx = cv2.approxPolyDP(table_contour, epsilon, True)

    if len(approx) != 4:
        return None

    return order_points(approx.reshape(4, 2))


# ============================
# TABLE WARPER
# ============================


class TableWarper:
    def __init__(self, dst_size):
        self.dst_pts = np.float32(
            [[0, 0], [dst_size[0], 0], [dst_size[0], dst_size[1]], [0, dst_size[1]]]
        )
        self.matrix = None
        self.size = dst_size

    def compute(self, src_pts):
        self.matrix = cv2.getPerspectiveTransform(src_pts, self.dst_pts)

    def warp(self, frame):
        return cv2.warpPerspective(frame, self.matrix, self.size)


# ============================
# CENTROID TRACKER
# ============================


class CentroidTracker:
    def __init__(self, max_distance=60):
        self.objects = OrderedDict()
        self.next_id = 0
        self.max_distance = max_distance

    def update(self, detections):
        updated = OrderedDict()

        for cx, cy in detections:
            assigned = False
            for obj_id, (ox, oy) in self.objects.items():
                if math.hypot(cx - ox, cy - oy) < self.max_distance:
                    updated[obj_id] = (cx, cy)
                    assigned = True
                    break

            if not assigned:
                updated[self.next_id] = (cx, cy)
                self.next_id += 1

        self.objects = updated
        return self.objects


# ============================
# IMPROVED BALL DETECTION
# ============================


def setup_blob_detector():
    """Setup SimpleBlobDetector with parameters optimized for ball detection."""
    params = cv2.SimpleBlobDetector_Params()

    # Filter by area - very lenient
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 20000

    # Filter by circularity - very lenient
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by color (dark blobs)
    params.filterByColor = False

    # Filter by convexity - very lenient
    params.filterByConvexity = True
    params.minConvexity = 0.3

    # Filter by inertia - very lenient
    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    return cv2.SimpleBlobDetector_create(params)


def detect_balls(frame):
    """Detect balls using SimpleBlobDetector."""
    # Create a binary mask by inverting the green table
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Table mask (green)
    table_lower = np.array([35, 40, 40])
    table_upper = np.array([85, 255, 255])
    table_mask = cv2.inRange(hsv, table_lower, table_upper)

    # Invert to get non-table objects (balls, shadows, etc.)
    mask = cv2.bitwise_not(table_mask)

    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Detect blobs
    detector = setup_blob_detector()
    keypoints = detector.detect(mask)

    # Extract centroids from keypoints
    detections = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

    return detections


# ============================
# MAIN
# ============================


def mouse_callback(event, x, y, flags, param):
    """Callback for mouse events on the combined display window."""
    global abort_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is on the abort button area (top-right corner)
        img = param
        h, w = img.shape[:2]
        # Abort button is in top-right: approximately last 100 pixels width, first 40 pixels height
        if w - 100 < x < w and 0 < y < 40:
            abort_flag = True


def display_combined(original, warped, text_info=""):
    """
    Display original and warped images side by side.
    """
    # Scale down by 50% for smaller window
    scale = 0.5
    original_scaled = cv2.resize(
        original, (int(original.shape[1] * scale), int(original.shape[0] * scale))
    )

    h, w = original_scaled.shape[:2]
    wh, ww = warped.shape[:2]

    # Resize warped to match original height for better side-by-side view
    aspect_ratio = ww / wh
    new_w = int(h * aspect_ratio)
    warped_resized = cv2.resize(warped, (new_w, h))

    # Create combined image
    combined = np.hstack([original_scaled, warped_resized])

    # Add info text
    if text_info:
        cv2.putText(
            combined, text_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
        )

    # Display and set mouse callback
    cv2.imshow("Original | Warped", combined)
    cv2.setMouseCallback("Original | Warped", mouse_callback, combined)

    return combined


def main():
    global abort_flag

    # Ask user for video path
    print("\n=== Ball Tracking System ===")
    print("Available videos:")
    video_dir = "videos"
    if os.path.exists(video_dir):
        videos = [
            f
            for f in os.listdir(video_dir)
            if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        for i, v in enumerate(videos, 1):
            print(f"  {i}. {v}")

    video_path = input("\nEnter video path (or video number): ").strip()

    # If user entered a number, select from available videos
    if video_path.isdigit():
        video_num = int(video_path) - 1
        if 0 <= video_num < len(videos):
            video_path = os.path.join(video_dir, videos[video_num])
        else:
            print("ERROR: Invalid video number")
            return

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Empty video")
        return

    table_pts = detect_table_corners(first_frame)
    if table_pts is None:
        print("ERROR: Table not detected")
        return

    print(f"Detected table corners: {table_pts}")
    print("Press SPACE to pause/resume, ESC to quit")

    warper = TableWarper((TABLE_W, TABLE_H))
    warper.compute(table_pts)

    tracker = CentroidTracker()
    frame_count = 0
    paused = False
    current_frame = None
    current_warped = None
    current_tracked = None

    while True:
        abort_flag = False

        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            current_frame = frame
            current_warped = warper.warp(frame)
            detections = detect_balls(current_warped)
            current_tracked = tracker.update(detections)

        frame = current_frame
        warped = current_warped
        tracked = current_tracked

        for obj_id, (x, y) in tracked.items():
            cv2.circle(warped, (x, y), 8, (0, 255, 255), 2)
            cv2.putText(
                warped,
                f"{obj_id}",
                (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        # Display combined view with pause status
        pause_text = " [PAUSED]" if paused else ""
        display_combined(
            frame, warped, f"Frame: {frame_count} | Balls: {len(tracked)}{pause_text}"
        )

        # Handle key presses and abort flag
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or abort_flag:  # ESC to quit
            break
        elif key == 32:  # SPACE to pause/resume
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
