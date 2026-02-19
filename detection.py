"""Table and ball detection utilities."""

import cv2
import numpy as np

TABLE_HSV_LOWER = np.array([35, 40, 40])
TABLE_HSV_UPPER = np.array([85, 255, 255])

ALLOWED_BALL_LABELS = {"cue", "yellow", "green", "brown", "blue", "pink", "black"}


def order_points(pts):
    """Order corner points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def detect_table_corners(frame):
    """
    Detect table corners using contour detection.
    Works with rotated/skewed tables by finding the quadrilateral that best fits the table.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, TABLE_HSV_LOWER, TABLE_HSV_UPPER)

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


def detect_table_orientation(corners):
    """
    Detect if table is filmed from short side or long side.
    Returns True if from long side (landscape), False if from short side (portrait).
    """
    # Calculate distances between corners
    # corners order: top-left, top-right, bottom-right, bottom-left
    top_edge = np.linalg.norm(corners[1] - corners[0])  # top side length
    left_edge = np.linalg.norm(corners[3] - corners[0])  # left side length

    # If top edge is longer than left edge, it's landscape (long side view)
    return top_edge > left_edge


_BLOB_DETECTOR = None


def setup_blob_detector():
    """Setup SimpleBlobDetector with parameters optimized for ball detection."""
    params = cv2.SimpleBlobDetector_Params()

    # Filter by area - lenient range
    params.filterByArea = True
    params.minArea = 8
    params.maxArea = 80000

    # Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = 0.05

    # Filter by color (white blobs on mask)
    params.filterByColor = True
    params.blobColor = 255

    # Thresholding for binary/near-binary masks
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep = 5

    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.2

    # Filter by inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.05

    # Minimum distance between blobs
    params.minDistBetweenBlobs = 2

    return cv2.SimpleBlobDetector_create(params)


def _get_blob_detector():
    return setup_blob_detector()


def detect_balls_blob(frame):
    """Detect balls using SimpleBlobDetector and classify by color."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Table mask (green)
    table_mask = cv2.inRange(hsv, TABLE_HSV_LOWER, TABLE_HSV_UPPER)

    # Invert to get non-table objects (balls, shadows, etc.)
    mask = cv2.bitwise_not(table_mask)

    # Apply a light close to connect ball blobs without erasing small balls
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Detect blobs
    detector = _get_blob_detector()
    keypoints = detector.detect(mask)

    detections = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = max(2, int(kp.size / 2))
        label = classify_ball_color(hsv, x, y, r)
        if label in ALLOWED_BALL_LABELS:
            detections.append({"x": x, "y": y, "r": r, "label": label})

    # Add contour-based circle candidates to catch green/brown/pink balls
    min_r, max_r = _estimate_radius_bounds(frame.shape[:2])
    contour_circles = _detect_circles_contour(gray, min_r, max_r)
    detections = _merge_circle_detections(detections, contour_circles, hsv)

    return detections


def detect_balls_hough(_frame):
    """Deprecated: kept for reference, prefer detect_balls_blob."""
    return []


def detect_pockets_warp(frame):
    """Return fixed pocket boxes from warp geometry (4 corners + 2 middle long-side)."""
    h, w = frame.shape[:2]
    max_size = max(8, min(h, w) - 2)
    size = min(max(12, int(min(h, w) * 0.07)), max_size)
    half = size // 2
    x_min = half
    y_min = half
    x_max = max(x_min, w - 1 - half)
    y_max = max(y_min, h - 1 - half)
    x_mid = int(np.clip(w // 2, x_min, x_max))
    y_mid = int(np.clip(h // 2, y_min, y_max))

    # Four corners of the warped frame.
    centers = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]

    # Two middle pockets on the long sides.
    if w >= h:
        centers.extend([(x_mid, y_min), (x_mid, y_max)])
    else:
        centers.extend([(x_min, y_mid), (x_max, y_mid)])

    pockets = []
    for cx, cy in centers:
        x = int(np.clip(cx - half, 0, max(0, w - size)))
        y = int(np.clip(cy - half, 0, max(0, h - size)))
        pockets.append({"x": x, "y": y, "size": int(size)})

    return pockets


def classify_ball_color(hsv_frame, x, y, r):
    """Classify a ball color based on mean HSV in a circular mask."""
    h, s, v = _median_hsv_in_circle(hsv_frame, x, y, r)
    label = "unknown"

    if v < 50:
        label = "black"
    elif s < 40 and v > 150:
        label = "cue"
    elif h <= 12 or h >= 170:
        label = "red"
    elif 12 < h < 25 and v < 160:
        label = "brown"
    elif 15 <= h <= 40:
        label = "yellow"
    elif 40 < h <= 85:
        label = "green"
    elif 90 <= h <= 135:
        label = "blue"
    elif 145 <= h < 170:
        # Only accept pink when it's very bright and less saturated
        if v >= 170 and s < 140:
            label = "pink"
        else:
            label = "red"

    return label


def _median_hsv_in_circle(hsv_frame, x, y, r):
    h, w = hsv_frame.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    r = max(2, int(r * 0.6))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)

    pixels = hsv_frame[mask == 255]
    if pixels.size == 0:
        return 0.0, 0.0, 0.0

    # Filter out very dark pixels to avoid shadow bias.
    val = pixels[:, 2]
    keep = val > 30
    if keep.any():
        pixels = pixels[keep]

    h_med = float(np.median(pixels[:, 0]))
    s_med = float(np.median(pixels[:, 1]))
    v_med = float(np.median(pixels[:, 2]))
    return h_med, s_med, v_med


def _detect_circles_contour(gray, min_radius, max_radius):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.5:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        if min_radius <= radius <= max_radius:
            circles.append((int(x), int(y), radius))

    return circles


def _estimate_radius_bounds(shape):
    h, w = shape[:2]
    min_dim = min(h, w)
    min_radius = max(2, int(min_dim * 0.01))
    max_radius = max(min_radius + 2, int(min_dim * 0.045))
    return min_radius, max_radius


def _merge_circle_detections(detections, circles, hsv):
    merged = list(detections)
    for x, y, r in circles:
        if _is_near_existing(merged, x, y, r):
            continue
        label = classify_ball_color(hsv, x, y, r)
        if label in ALLOWED_BALL_LABELS:
            merged.append({"x": int(x), "y": int(y), "r": int(r), "label": label})
    return merged


def _is_near_existing(detections, x, y, r):
    for det in detections:
        dx = det["x"] - x
        dy = det["y"] - y
        if dx * dx + dy * dy <= (r * r):
            return True
    return False
