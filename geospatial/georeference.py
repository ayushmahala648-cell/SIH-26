"""
Georeferencing, Spatial Metrics, and Cadastral Attribute Computation.
Transforms pixel coordinates to metric surveying units and geographic WGS84 coordinates.
Calculates parcel area (sq meters, hectares, acres, gaj), perimeter, compactness, and land-use classification.
"""

import math
from typing import Dict, Any, List, Optional, Tuple
from shapely.geometry import Polygon, Point


class CadastralGeoreferencer:
    def __init__(
        self,
        gsd_meters: float = 0.08,
        origin_lat: float = 28.6139,
        origin_lon: float = 77.2090
    ):
        """
        gsd_meters: Ground Sample Distance (meters per pixel)
        origin_lat, origin_lon: Reference geographic coordinates (WGS84) for anchor
        """
        self.gsd = gsd_meters
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon

    def pixel_to_geo(self, px_x: float, px_y: float) -> Tuple[float, float]:
        """
        Converts image pixel coordinates (x, y) to approximate WGS84 (lon, lat).
        Uses spherical Earth approximation locally.
        """
        # 1 deg latitude ~ 111,320 meters
        # 1 deg longitude ~ 111,320 * cos(lat) meters
        lat_rad = math.radians(self.origin_lat)
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(lat_rad)

        offset_x_meters = px_x * self.gsd
        offset_y_meters = -px_y * self.gsd  # y increases downward in image

        lon = self.origin_lon + (offset_x_meters / meters_per_deg_lon)
        lat = self.origin_lat + (offset_y_meters / meters_per_deg_lat)
        return lon, lat

    def compute_parcel_metrics(
        self,
        parcel_id: int,
        polygon: Polygon,
        building_polys: Optional[List[Polygon]] = None,
        vegetation_ratio: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculates all cadastral surveyed attributes for a parcel polygon.
        """
        px_area = polygon.area
        px_perimeter = polygon.length

        # Convert to real-world metric units using GSD
        area_sqm = px_area * (self.gsd ** 2)
        perimeter_m = px_perimeter * self.gsd
        area_hectares = area_sqm / 10000.0
        area_acres = area_sqm / 4046.8564
        area_gaj = area_sqm * 1.19599  # 1 sqm = 1.19599 sq yards (गज)

        # Compactness score: 4*pi*A / P^2 (1.0 = perfect circle, ~0.785 = square)
        if perimeter_m > 0:
            compactness = min(1.0, (4.0 * math.pi * area_sqm) / (perimeter_m ** 2))
        else:
            compactness = 0.0

        # Calculate building footprint coverage inside this parcel
        built_area_sqm = 0.0
        building_count = 0
        if building_polys:
            for b_poly in building_polys:
                if polygon.intersects(b_poly):
                    try:
                        inter = polygon.intersection(b_poly)
                        if not inter.is_empty:
                            built_area_sqm += inter.area * (self.gsd ** 2)
                            building_count += 1
                    except Exception:
                        pass

        builtup_pct = min(100.0, (built_area_sqm / area_sqm * 100.0)) if area_sqm > 0 else 0.0

        # Heuristic Land Use Classification
        if builtup_pct >= 20.0:
            if area_sqm > 1500:
                land_use = "Commercial / Institutional"
            else:
                land_use = "Residential"
        elif vegetation_ratio > 0.40 or (builtup_pct < 5.0 and area_sqm > 2000):
            land_use = "Agricultural / Green Space"
        elif builtup_pct < 5.0:
            land_use = "Vacant / Open Plot"
        else:
            land_use = "Mixed Use"

        # Centroid coordinates
        c_x, c_y = polygon.centroid.x, polygon.centroid.y
        geo_lon, geo_lat = self.pixel_to_geo(c_x, c_y)

        # Confidence rating based on polygon geometry regularity
        confidence = min(0.98, max(0.70, 0.75 + 0.20 * compactness))

        return {
            "parcel_id": f"P-{parcel_id:04d}",
            "numeric_id": parcel_id,
            "area_sqm": round(area_sqm, 2),
            "area_hectares": round(area_hectares, 4),
            "area_acres": round(area_acres, 4),
            "area_gaj": round(area_gaj, 2),
            "perimeter_m": round(perimeter_m, 2),
            "compactness": round(compactness, 3),
            "builtup_area_sqm": round(built_area_sqm, 2),
            "builtup_pct": round(builtup_pct, 1),
            "building_count": building_count,
            "land_use": land_use,
            "confidence": round(confidence, 2),
            "verification_status": "Pending Review",
            "centroid_px": [round(c_x, 1), round(c_y, 1)],
            "centroid_geo": [round(geo_lon, 6), round(geo_lat, 6)]
        }

