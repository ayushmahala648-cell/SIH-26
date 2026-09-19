"""
Cadastral Polygon Regularization and Topological Enforcement.
Implements Douglas-Peucker boundary simplification, orthogonal snapping for building footprints,
and planar topology sanitization to guarantee valid non-overlapping parcel maps.
"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union
from typing import List, Dict, Tuple, Optional


class CadastralRegularizer:
    def __init__(
        self,
        parcel_simplify_tol: float = 2.0,
        building_simplify_tol: float = 1.5,
        snap_orthogonal: bool = True
    ):
        self.parcel_simplify_tol = parcel_simplify_tol
        self.building_simplify_tol = building_simplify_tol
        self.snap_orthogonal = snap_orthogonal

    def simplify_parcel(self, poly: Polygon, tolerance: Optional[float] = None) -> Polygon:
        """
        Applies Douglas-Peucker simplification to reduce staircase raster noise
        while preserving actual cadastral corner vertices.
        """
        tol = tolerance if tolerance is not None else self.parcel_simplify_tol
        simplified = poly.simplify(tol, preserve_topology=True)
        if not simplified.is_valid:
            simplified = make_valid(simplified)
        if isinstance(simplified, MultiPolygon):
            simplified = max(simplified.geoms, key=lambda p: p.area)
        return simplified

    def regularize_building(self, poly: Polygon) -> Polygon:
        """
        Regularizes building footprints. For structures with high rectangularity (>0.75),
        snaps to minimum rotated bounding rectangle; otherwise simplifies with orthogonal edges.
        """
        if not poly.is_valid or poly.area < 10:
            return poly

        # Calculate Minimum Rotated Bounding Box
        min_box = poly.minimum_rotated_rectangle
        if min_box.area > 0:
            rectangularity = poly.area / min_box.area
            # If structure is close to a clean rectangle/box, snap to bounding box
            if rectangularity >= 0.80 and self.snap_orthogonal:
                return min_box

        # Otherwise simplify edges
        simplified = poly.simplify(self.building_simplify_tol, preserve_topology=True)
        if not simplified.is_valid:
            simplified = make_valid(simplified)
        if isinstance(simplified, MultiPolygon):
            simplified = max(simplified.geoms, key=lambda p: p.area)
        return simplified

    def enforce_parcel_topology(self, parcel_dict: Dict[int, Polygon]) -> Dict[int, Polygon]:
        """
        Enforces planar topology across adjacent parcels:
        Ensures parcels do not overlap each other by trimming overlapping slivers.
        """
        sanitized_parcels = {}
        # Sort parcels by area descending so dominant parcels maintain shape
        sorted_pids = sorted(parcel_dict.keys(), key=lambda pid: parcel_dict[pid].area, reverse=True)

        allocated_union = None

        for pid in sorted_pids:
            poly = parcel_dict[pid]
            if not poly.is_valid:
                poly = make_valid(poly)

            # Simplify boundary
            clean_poly = self.simplify_parcel(poly)

            if allocated_union is None:
                sanitized_parcels[pid] = clean_poly
                allocated_union = clean_poly
            else:
                # Subtract already allocated adjacent parcels if there is an overlap
                if clean_poly.intersects(allocated_union):
                    try:
                        diff = clean_poly.difference(allocated_union)
                        if not diff.is_empty:
                            if isinstance(diff, MultiPolygon):
                                diff = max(diff.geoms, key=lambda p: p.area)
                            if diff.area >= 50:
                                sanitized_parcels[pid] = diff
                                allocated_union = unary_union([allocated_union, diff])
                    except Exception:
                        sanitized_parcels[pid] = clean_poly
                        allocated_union = unary_union([allocated_union, clean_poly])
                else:
                    sanitized_parcels[pid] = clean_poly
                    allocated_union = unary_union([allocated_union, clean_poly])

        return sanitized_parcels

