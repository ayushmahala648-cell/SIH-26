"""
Boundary detection and structural edge affinity module for cadastral parcels.
Combines multi-scale morphological gradients, Canny edge detection with Otsu thresholding,
and directional line segment detection to delineate crisp parcel boundaries.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class CadastralBoundaryDetector:
    def __init__(
        self,
        blur_ksize: int = 5,
        morph_ksize: int = 3,
        canny_low_ratio: float = 0.5,
        canny_high_ratio: float = 1.2
    ):
        self.blur_ksize = blur_ksize
        self.morph_ksize = morph_ksize
        self.canny_low_ratio = canny_low_ratio
        self.canny_high_ratio = canny_high_ratio

    def extract_boundary_map(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Extracts a normalized continuous boundary probability/affinity map [0.0, 1.0]
        from an aerial RGB image.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Bilateral filtering to preserve sharp cadastral edges while smoothing roof/foliage noise
        filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

        # 2. Multi-scale Morphological Gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_ksize, self.morph_ksize))
        morph_grad = cv2.morphologyEx(filtered, cv2.MORPH_GRADIENT, kernel)

        # 3. Adaptive Otsu-based Canny edge detection
        high_thresh, _ = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = high_thresh * self.canny_low_ratio
        high_thresh = high_thresh * self.canny_high_ratio
        canny = cv2.Canny(filtered, low_thresh, high_thresh)

        # 4. Scharr / Sobel gradient magnitude
        grad_x = cv2.Scharr(filtered, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(filtered, cv2.CV_32F, 0, 1)
        magnitude = cv2.magnitude(grad_x, grad_y)
        cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
        magnitude = magnitude.astype(np.uint8)

        # 5. Fusion of edge features
        fused = cv2.addWeighted(morph_grad, 0.4, magnitude, 0.35, 0)
        fused = cv2.addWeighted(fused, 1.0, canny, 0.25, 0)

        # Normalize to [0.0, 1.0]
        boundary_map = fused.astype(np.float32) / 255.0
        return boundary_map

    def extract_straight_segments(self, binary_edges: np.ndarray, min_line_len: int = 30) -> list:
        """
        Detects prominent straight line segments (walls, compound boundaries, roads)
        using probabilistic Hough Transform.
        """
        lines = cv2.HoughLinesP(
            binary_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=min_line_len,
            maxLineGap=10
        )
        if lines is None:
            return []
        return [l[0].tolist() for l in lines]

