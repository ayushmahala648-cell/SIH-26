"""
Raster-to-Vector conversion engine for cadastral features.
Converts segmented raster masks into structured Shapely geometric polygons and linestrings.
"""

import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, shape
from shapely.validation import make_valid
from typing import List, Dict, Any, Optional, Tuple


class CadastralVectorizer:
    def __init__(self, min_area_px: float = 80.0):
        self.min_area_px = min_area_px

    def mask_to_polygons(self, binary_mask: np.ndarray, scale_back: float = 1.0) -> List[Polygon]:
        """
        Converts a binary mask (uint8) into a list of valid Shapely Polygons.
        scale_back allows scaling coordinates back to original high-res image dimensions.
        """
        contours, hierarchy = cv2.findContours(
            binary_mask.astype(np.uint8),
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours or hierarchy is None:
            return []

        hierarchy = hierarchy[0]
        polygons = []

        # Find external contours and their interior holes
        for idx, h in enumerate(hierarchy):
            # h: [next, prev, first_child, parent]
            # External contour (parent == -1)
            if h[3] == -1:
                ext_contour = contours[idx]
                if len(ext_contour) < 3:
                    continue

                ext_pts = ext_contour.squeeze(axis=1) * (1.0 / scale_back)
                if len(ext_pts) < 3:
                    continue

                # Find any direct children holes
                holes = []
                child_idx = h[2]
                while child_idx != -1:
                    hole_contour = contours[child_idx]
                    if len(hole_contour) >= 3:
                        hole_pts = hole_contour.squeeze(axis=1) * (1.0 / scale_back)
                        if len(hole_pts) >= 3:
                            holes.append(hole_pts.tolist())
                    child_idx = hierarchy[child_idx][0]

                try:
                    poly = Polygon(shell=ext_pts.tolist(), holes=holes)
                    if not poly.is_valid:
                        poly = make_valid(poly)

                    if isinstance(poly, Polygon) and poly.area >= self.min_area_px:
                        polygons.append(poly)
                    elif isinstance(poly, MultiPolygon):
                        for p in poly.geoms:
                            if p.area >= self.min_area_px:
                                polygons.append(p)
                except Exception:
                    continue

        return polygons

    def extract_parcel_polygons(self, labeled_mask: np.ndarray, scale_back: float = 1.0) -> Dict[int, Polygon]:
        """
        Extracts polygons for each discrete parcel label (1..N).
        Returns a dict mapping parcel_id -> Shapely Polygon.
        """
        parcel_dict = {}
        unique_ids = np.unique(labeled_mask)

        for pid in unique_ids:
            if pid <= 0:
                continue

            single_mask = (labeled_mask == pid).astype(np.uint8) * 255
            polys = self.mask_to_polygons(single_mask, scale_back=scale_back)
            if polys:
                # Select the largest polygon if multiple are returned
                largest_poly = max(polys, key=lambda p: p.area)
                parcel_dict[int(pid)] = largest_poly

        return parcel_dict

    def extract_road_centerlines(self, road_mask: np.ndarray, scale_back: float = 1.0) -> List[LineString]:
        """
        Extracts road centerlines using contour skeletonization.
        """
        contours, _ = cv2.findContours(
            road_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_TC89_KCOS
        )
        lines = []
        for c in contours:
            if len(c) >= 2:
                pts = c.squeeze(axis=1) * (1.0 / scale_back)
                if len(pts) >= 2:
                    try:
                        line = LineString(pts.tolist())
                        if line.length > 20:
                            lines.append(line)
                    except Exception:
                        pass
        return lines

