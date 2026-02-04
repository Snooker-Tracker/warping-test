import cv2
import numpy as np


class TableWarper:
    """Handles perspective transformation to view table from above."""
    
    def __init__(self, dst_size):
        self.dst_pts = np.float32([
            [0, 0],
            [dst_size[0], 0],
            [dst_size[0], dst_size[1]],
            [0, dst_size[1]]
        ])
        self.matrix = None
        self.size = dst_size

    def compute(self, src_pts):
        """Compute the perspective transform matrix from source points."""
        self.matrix = cv2.getPerspectiveTransform(src_pts, self.dst_pts)

    def warp(self, frame):
        """Apply perspective transform to frame."""
        return cv2.warpPerspective(frame, self.matrix, self.size)
