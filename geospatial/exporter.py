"""
Cadastral GIS Exporters.
Generates standard RFC 7946 GeoJSON FeatureCollections, Cadastral Land Register CSVs,
and saves vector outputs for GIS consumption (QGIS, ArcGIS, Mapbox).
"""

import os
import json
import csv
import numpy as np
from typing import Dict, Any, List, Optional
from shapely.geometry import Polygon, MultiPolygon, LineString, mapping


class CadastralExporter:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def to_geojson(
        self,
        parcels: List[Dict[str, Any]],
        buildings: Optional[List[Dict[str, Any]]] = None,
        roads: Optional[List[Dict[str, Any]]] = None,
        use_geo_coords: bool = True
    ) -> Dict[str, Any]:
        """
        Creates an RFC 7946 compliant GeoJSON FeatureCollection.
        """
        features = []

        # 1. Parcel Features
        for p in parcels:
            poly = p["geometry"]
            coords = self._polygon_to_coords(poly, p.get("coord_transform") if use_geo_coords else None)
            
            props = {k: v for k, v in p.items() if k not in ["geometry", "coord_transform"]}
            props["feature_type"] = "parcel"

            feature = {
                "type": "Feature",
                "id": p.get("parcel_id"),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coords
                },
                "properties": props
            }
            features.append(feature)

        # 2. Building Features
        if buildings:
            for b in buildings:
                poly = b["geometry"]
                coords = self._polygon_to_coords(poly, b.get("coord_transform") if use_geo_coords else None)
                props = {k: v for k, v in b.items() if k not in ["geometry", "coord_transform"]}
                props["feature_type"] = "building"

                feature = {
                    "type": "Feature",
                    "id": b.get("building_id"),
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coords
                    },
                    "properties": props
                }
                features.append(feature)

        # 3. Road Features
        if roads:
            for r in roads:
                line = r["geometry"]
                coords = self._linestring_to_coords(line, r.get("coord_transform") if use_geo_coords else None)
                props = {k: v for k, v in r.items() if k not in ["geometry", "coord_transform"]}
                props["feature_type"] = "road"

                feature = {
                    "type": "Feature",
                    "id": r.get("road_id"),
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    },
                    "properties": props
                }
                features.append(feature)

        return {
            "type": "FeatureCollection",
            "name": "Cadastral_Features_Paridhi",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:OGC:1.3:CRS84" if use_geo_coords else "pixel"
                }
            },
            "features": features
        }

    def _polygon_to_coords(self, poly: Polygon, transform_fn=None) -> List[List[List[float]]]:
        """Converts Shapely Polygon to nested GeoJSON coordinate rings."""
        rings = [list(poly.exterior.coords)]
        for interior in poly.interiors:
            rings.append(list(interior.coords))

        if transform_fn is None:
            return [[[round(x, 2), round(y, 2)] for x, y in ring] for ring in rings]
        else:
            return [[[round(c[0], 6), round(c[1], 6)] for c in [transform_fn(x, y) for x, y in ring]] for ring in rings]

    def _linestring_to_coords(self, line: LineString, transform_fn=None) -> List[List[float]]:
        """Converts Shapely LineString to GeoJSON coordinates."""
        pts = list(line.coords)
        if transform_fn is None:
            return [[round(x, 2), round(y, 2)] for x, y in pts]
        else:
            return [[round(c[0], 6), round(c[1], 6)] for c in [transform_fn(x, y) for x, y in pts]]

    def save_geojson(self, geojson_data: Dict[str, Any], filepath: str) -> str:
        """Saves GeoJSON dict to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)
        return filepath

    def save_csv_register(self, parcels: List[Dict[str, Any]], filepath: str) -> str:
        """
        Saves tabular Cadastral Land Register into CSV.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fieldnames = [
            "parcel_id", "land_use", "area_sqm", "area_hectares", "area_acres",
            "area_gaj", "perimeter_m", "builtup_pct", "builtup_area_sqm",
            "building_count", "compactness", "confidence", "verification_status",
            "centroid_lon", "centroid_lat"
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for p in parcels:
                row = dict(p)
                if "centroid_geo" in p and len(p["centroid_geo"]) == 2:
                    row["centroid_lon"] = p["centroid_geo"][0]
                    row["centroid_lat"] = p["centroid_geo"][1]
                writer.writerow(row)

        return filepath

