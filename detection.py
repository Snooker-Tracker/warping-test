import cv2
import numpy as np


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


def setup_blob_detector():
    """Setup SimpleBlobDetector with parameters optimized for ball detection."""
    params = cv2.SimpleBlobDetector_Params()

    # Filter by area - very lenient
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 20000

    # Filter by circularity - lenient but stable
    params.filterByCircularity = True
    params.minCircularity = 0.2

    # Filter by color (dark blobs)
    params.filterByColor = False

    # Filter by convexity - more stable
    params.filterByConvexity = True
    params.minConvexity = 0.4

    # Filter by inertia - more stable
    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    # Minimum distance between blobs - increased to prevent flickering
    params.minDistBetweenBlobs = 5

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

    # Apply morphological operations to clean up and stabilize
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Detect blobs
    detector = setup_blob_detector()
    keypoints = detector.detect(mask)

    # Extract centroids from keypoints
    detections = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

    return detections
