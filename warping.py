"""Perspective warping utilities for the table view."""

import cv2
import numpy as np


class TableWarper:
    """Handles perspective transformation to view table from above."""

    def __init__(self, dst_size, edge_padding=10):
        """
        Initialize warper with destination size.

        Args:
            dst_size: tuple of (width, height) for the warped output
            edge_padding: pixels to add on each edge to capture holes (default 10)
        """
        self.original_size = dst_size
        self.edge_padding = edge_padding

        # Add padding to size
        padded_w = dst_size[0] + 2 * edge_padding
        padded_h = dst_size[1] + 2 * edge_padding

        # Destination points with padding offset (centered)
        self.dst_pts = np.float32(
            [
                [edge_padding, edge_padding],
                [dst_size[0] + edge_padding, edge_padding],
                [dst_size[0] + edge_padding, dst_size[1] + edge_padding],
                [edge_padding, dst_size[1] + edge_padding],
            ]
        )
        self.matrix = None
        self.size = (padded_w, padded_h)

    def compute(self, src_pts):
        """Compute the perspective transform matrix from source points."""
        self.matrix = cv2.getPerspectiveTransform(src_pts, self.dst_pts)

    def warp(self, frame):
        """Apply perspective transform to frame."""
        return cv2.warpPerspective(frame, self.matrix, self.size)
