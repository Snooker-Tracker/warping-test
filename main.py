"""Main entry point for the ball tracking system."""

import os
import time

import cv2

from detection import detect_balls_blob, detect_table_corners, detect_table_orientation
from display import display_combined, draw_tracked_balls, is_window_open
from tracking import KalmanBallTracker
from warping import TableWarper

SHORT_SIDE_W = 280
SHORT_SIDE_H = 560


def select_video():
    """Ask user to select a video file."""
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
    else:
        videos = []

    video_path = input("\nEnter video path (or video number): ").strip()

    # If user entered a number, select from available videos
    if video_path.isdigit():
        video_num = int(video_path) - 1
        if 0 <= video_num < len(videos):
            video_path = os.path.join(video_dir, videos[video_num])
        else:
            print("ERROR: Invalid video number")
            return None

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return None

    return video_path


def main():
    """Run the ball tracking pipeline for a selected video."""
    video_path = select_video()
    if video_path is None:
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

    # Detect table orientation
    is_long_side = detect_table_orientation(table_pts)
    if is_long_side:
        table_w, table_h = SHORT_SIDE_H, SHORT_SIDE_W  # Swap for landscape
        print("Detected: Filming from LONG side")
    else:
        table_w, table_h = SHORT_SIDE_W, SHORT_SIDE_H
        print("Detected: Filming from SHORT side")

    print(f"Detected table corners: {table_pts}")
    print(f"Warp dimensions: {table_w}x{table_h}")
    print("Press SPACE to pause/resume, ESC to quit")

    warper = TableWarper((table_w, table_h))
    warper.compute(table_pts)

    tracker = KalmanBallTracker(max_missing=12, gating_distance=60)
    frame_count = 0
    paused = False
    current_frame = None
    current_warped = None
    current_tracked = {}

    # FPS tracking
    fps = 0
    prev_time = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            current_frame = frame
            current_warped = warper.warp(frame)
            detections = detect_balls_blob(current_warped)
            current_tracked = tracker.update(detections)

        frame = current_frame
        warped = current_warped.copy()
        tracked = current_tracked

        # Draw tracked balls
        warped = draw_tracked_balls(warped, tracked)

        # Calculate FPS
        elapsed = time.time() - prev_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
        prev_time = time.time()

        # Display combined view with pause status
        pause_text = " [PAUSED]" if paused else ""
        display_combined(
            frame,
            warped,
            f"Frame: {frame_count} | FPS: {fps:.1f} | Balls: {len(tracked)}{pause_text}",
            is_landscape=is_long_side,
        )

        # Handle key presses and window close events
        key = cv2.waitKey(30) & 0xFF
        if not is_window_open():
            break
        if key == 27:  # ESC to quit
            break
        if key == 32:  # SPACE to pause/resume
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
