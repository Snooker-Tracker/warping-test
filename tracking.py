"""Tracking utilities for detected balls."""

import math
from collections import OrderedDict

import cv2
import numpy as np


class CentroidTracker:
    """Tracks ball positions using centroid matching."""

    def __init__(self, max_distance=60):
        self.objects = OrderedDict()
        self.next_id = 0
        self.max_distance = max_distance

    def update(self, detections):
        """Update tracked objects with new detections."""
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

    def reset(self):
        """Reset tracker state."""
        self.objects.clear()
        self.next_id = 0


class KalmanBallTracker:
    """Lightweight multi-object tracker using per-ball Kalman filters."""

    def __init__(self, max_missing=10, gating_distance=50):
        self.max_missing = max_missing
        self.gating_distance = gating_distance
        self.tracks = {}

    def update(self, detections):
        """
        Update tracks with current detections.

        Args:
            detections: list of dicts with keys x, y, r, label

        Returns:
            dict[label] -> (x, y, r)
        """
        grouped = {}
        for det in detections:
            grouped.setdefault(det["label"], []).append(det)

        # Update existing tracks
        for label, track in list(self.tracks.items()):
            pred_x, pred_y = _predict(track["kf"])

            match = _select_best_detection(
                grouped.get(label, []), pred_x, pred_y, self.gating_distance
            )
            if match is not None:
                _correct(track["kf"], match["x"], match["y"])
                track["missing"] = 0
                track["radius"] = match["r"]
                track["last"] = (match["x"], match["y"])
            else:
                track["missing"] += 1
                track["last"] = (pred_x, pred_y)

            if track["missing"] > self.max_missing:
                del self.tracks[label]

        # Add new tracks for unseen labels
        for label, candidates in grouped.items():
            if label in self.tracks:
                continue
            best = _select_best_detection(candidates, None, None, None)
            if best is None:
                continue
            self.tracks[label] = _create_track(best["x"], best["y"], best["r"])

        tracked = {}
        for label, track in self.tracks.items():
            x, y = track["last"]
            tracked[label] = (int(x), int(y), int(track["radius"]))

        return tracked

    def reset(self):
        """Clear all tracked balls."""
        self.tracks.clear()


def _create_track(x, y, r):
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

    return {"kf": kf, "missing": 0, "radius": r, "last": (x, y)}


def _predict(kf):
    pred = kf.predict()
    return float(pred[0, 0]), float(pred[1, 0])


def _correct(kf, x, y):
    measurement = np.array([[np.float32(x)], [np.float32(y)]])
    kf.correct(measurement)


def _select_best_detection(candidates, pred_x, pred_y, gating_distance):
    if not candidates:
        return None
    if pred_x is None or pred_y is None:
        return candidates[0]

    best = None
    best_dist = None
    for det in candidates:
        dist = math.hypot(det["x"] - pred_x, det["y"] - pred_y)
        if best is None or dist < best_dist:
            best = det
            best_dist = dist

    if (
        best_dist is not None
        and gating_distance is not None
        and best_dist > gating_distance
    ):
        return None

    return best
