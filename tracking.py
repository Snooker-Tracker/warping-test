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

    def __init__(self, max_missing=10, gating_distance=50, max_red_tracks=15):
        self.max_missing = max_missing
        self.gating_distance = gating_distance
        self.max_red_tracks = max(0, int(max_red_tracks))
        self.single_tracks = {}
        self.red_tracks = {}
        self._next_red_track_id = 0

    def update(self, detections):
        """
        Update tracks with current detections.

        Args:
            detections: list of dicts with keys x, y, r, label

        Returns:
            dict[track_label] -> (x, y, r)
        """
        grouped = {}
        for det in detections:
            grouped.setdefault(det["label"], []).append(det)

        # Update existing non-red tracks (single instance per label).
        for label, track in list(self.single_tracks.items()):
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
                del self.single_tracks[label]

        # Add non-red tracks for unseen labels.
        for label, candidates in grouped.items():
            if label == "red" or label in self.single_tracks:
                continue
            best = _select_best_detection(candidates, None, None, None)
            if best is None:
                continue
            self.single_tracks[label] = _create_track(best["x"], best["y"], best["r"])

        # Update multi-instance red tracks.
        self._update_red_tracks(grouped.get("red", []))

        tracked = {}
        for label, track in self.single_tracks.items():
            x, y = track["last"]
            tracked[label] = (int(x), int(y), int(track["radius"]))

        for idx, track_id in enumerate(sorted(self.red_tracks), start=1):
            track = self.red_tracks[track_id]
            x, y = track["last"]
            tracked[f"red_{idx}"] = (int(x), int(y), int(track["radius"]))

        return tracked

    def _update_red_tracks(self, red_candidates):
        predictions = {}
        for track_id, track in self.red_tracks.items():
            pred_x, pred_y = _predict(track["kf"])
            track["last"] = (pred_x, pred_y)
            predictions[track_id] = (pred_x, pred_y)

        matches, unmatched_tracks, unmatched_dets = _match_tracks_to_detections(
            predictions, red_candidates, self.gating_distance
        )

        for track_id, det_idx in matches:
            det = red_candidates[det_idx]
            track = self.red_tracks[track_id]
            _correct(track["kf"], det["x"], det["y"])
            track["missing"] = 0
            track["radius"] = det["r"]
            track["last"] = (det["x"], det["y"])

        for track_id in unmatched_tracks:
            track = self.red_tracks.get(track_id)
            if track is None:
                continue
            track["missing"] += 1
            if track["missing"] > self.max_missing:
                del self.red_tracks[track_id]

        for det_idx in unmatched_dets:
            if len(self.red_tracks) >= self.max_red_tracks:
                break
            det = red_candidates[det_idx]
            self.red_tracks[self._next_red_track_id] = _create_track(
                det["x"], det["y"], det["r"]
            )
            self._next_red_track_id += 1

    def reset(self):
        """Clear all tracked balls."""
        self.single_tracks.clear()
        self.red_tracks.clear()
        self._next_red_track_id = 0


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


def _match_tracks_to_detections(predictions, detections, gating_distance):
    if not predictions:
        return [], [], list(range(len(detections)))
    if not detections:
        return [], list(predictions.keys()), []

    pairs = []
    for track_id, (pred_x, pred_y) in predictions.items():
        for det_idx, det in enumerate(detections):
            dist = math.hypot(det["x"] - pred_x, det["y"] - pred_y)
            pairs.append((dist, track_id, det_idx))
    pairs.sort(key=lambda item: item[0])

    used_tracks = set()
    used_dets = set()
    matches = []
    for dist, track_id, det_idx in pairs:
        if track_id in used_tracks or det_idx in used_dets:
            continue
        if gating_distance is not None and dist > gating_distance:
            continue
        used_tracks.add(track_id)
        used_dets.add(det_idx)
        matches.append((track_id, det_idx))

    unmatched_tracks = [
        track_id for track_id in predictions if track_id not in used_tracks
    ]
    unmatched_dets = [
        det_idx for det_idx in range(len(detections)) if det_idx not in used_dets
    ]
    return matches, unmatched_tracks, unmatched_dets
