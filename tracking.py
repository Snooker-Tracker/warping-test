"""Centroid-based tracking for detected balls."""

import math
from collections import OrderedDict


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
