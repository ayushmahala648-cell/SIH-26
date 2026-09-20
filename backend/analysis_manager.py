"""
Analysis Manager & Workflow Coordinator.
Executes cadastral processing pipelines, caches results, manages file storage,
and coordinates human-in-the-loop parcel verification edits.
"""

import os
import time
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import cv2
from PIL import Image

from models.cadastral_pipeline import CadastralPipeline
from geospatial.vectorizer import CadastralVectorizer
from geospatial.regularization import CadastralRegularizer
from geospatial.georeference import CadastralGeoreferencer
from geospatial.exporter import CadastralExporter


class AnalysisManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, outputs_dir: str = "outputs", uploads_dir: str = "uploads"):
        if getattr(self, "_initialized", False):
            return
        self.outputs_dir = outputs_dir
        self.uploads_dir = uploads_dir
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.uploads_dir, exist_ok=True)

        self.pipeline = CadastralPipeline()
        self.vectorizer = CadastralVectorizer()
        self.exporter = CadastralExporter(self.outputs_dir)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._initialized = True

    def list_sample_images(self, images_dir: str = "images") -> List[Dict[str, Any]]:
        """Lists available sample aerial photos with metadata."""
        samples = []
        if not os.path.exists(images_dir):
            return samples

        for filename in sorted(os.listdir(images_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                p = os.path.join(images_dir, filename)
                try:
                    with Image.open(p) as img:
                        w, h = img.size
                    samples.append({
                        "filename": filename,
                        "path": f"/images/{filename}",
                        "width": w,
                        "height": h,
                        "size_bytes": os.path.getsize(p)
                    })
                except Exception:
                    pass
        return samples

    def run_pipeline(
        self,
        image_path: str,
        gsd_meters: float = 0.08,
        simplify_tolerance: float = 2.0,
        min_parcel_area_px: int = 100,
        snap_orthogonal: bool = True
    ) -> Dict[str, Any]:
        """Runs the complete AI & Geospatial analysis pipeline."""
        t0 = time.time()
        analysis_id = str(uuid.uuid4())[:8]
        out_subfolder = os.path.join(self.outputs_dir, analysis_id)
        os.makedirs(out_subfolder, exist_ok=True)

        # 1. Feature Extraction via ML Models
        ml_res = self.pipeline.extract_cadastral_features(
            image_path,
            min_parcel_area=min_parcel_area_px
        )

        scale_back = ml_res["scale_factor"]
        w_orig, h_orig = ml_res["image_dims"]

        # 2. Vectorization
        parcel_polys = self.vectorizer.extract_parcel_polygons(
            ml_res["parcel_labels"], scale_back=scale_back
        )
        bldg_polys = self.vectorizer.mask_to_polygons(
            ml_res["building_mask"], scale_back=scale_back
        )
        road_lines = self.vectorizer.extract_road_centerlines(
            ml_res["road_mask"], scale_back=scale_back
        )

        # 3. Regularization & Topology
        regularizer = CadastralRegularizer(
            parcel_simplify_tol=simplify_tolerance,
            snap_orthogonal=snap_orthogonal
        )
        clean_parcels = regularizer.enforce_parcel_topology(parcel_polys)
        clean_bldgs = [regularizer.regularize_building(b) for b in bldg_polys]

        # 4. Georeferencing & Spatial Metrics
        georeferencer = CadastralGeoreferencer(gsd_meters=gsd_meters)
        parcel_records = []
        total_surveyed_sqm = 0.0
        total_builtup_sqm = 0.0

        for pid, poly in clean_parcels.items():
            metrics = georeferencer.compute_parcel_metrics(pid, poly, clean_bldgs)
            metrics["geometry"] = poly
            metrics["coord_transform"] = georeferencer.pixel_to_geo
            parcel_records.append(metrics)
            total_surveyed_sqm += metrics["area_sqm"]
            total_builtup_sqm += metrics["builtup_area_sqm"]

        # 5. Building & Road records for export
        building_records = []
        for idx, b_poly in enumerate(clean_bldgs, start=1):
            building_records.append({
                "building_id": f"B-{idx:04d}",
                "area_sqm": round(b_poly.area * (gsd_meters ** 2), 2),
                "geometry": b_poly,
                "coord_transform": georeferencer.pixel_to_geo
            })

        road_records = []
        for idx, r_line in enumerate(road_lines, start=1):
            road_records.append({
                "road_id": f"R-{idx:04d}",
                "length_m": round(r_line.length * gsd_meters, 2),
                "geometry": r_line,
                "coord_transform": georeferencer.pixel_to_geo
            })

        # 6. Exporters: Save GeoJSON (Geo & Pixel) and CSV
        geojson_geo = self.exporter.to_geojson(parcel_records, building_records, road_records, use_geo_coords=True)
        geojson_pixel = self.exporter.to_geojson(parcel_records, building_records, road_records, use_geo_coords=False)

        geojson_path = os.path.join(out_subfolder, "cadastral_features.geojson")
        self.exporter.save_geojson(geojson_geo, geojson_path)

        csv_path = os.path.join(out_subfolder, "cadastral_register.csv")
        self.exporter.save_csv_register(parcel_records, csv_path)

        # 7. Save Annotated Overlay Image
        overlay_filename = "cadastral_overlay.jpg"
        overlay_path = os.path.join(out_subfolder, overlay_filename)
        # Convert RGB to BGR for cv2 write
        overlay_bgr = cv2.cvtColor(ml_res["annotated_overlay"], cv2.COLOR_RGB2BGR)
        if (w_orig, h_orig) != (overlay_bgr.shape[1], overlay_bgr.shape[0]):
            overlay_bgr = cv2.resize(overlay_bgr, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(overlay_path, overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        duration = round(time.time() - t0, 2)
        builtup_pct = round((total_builtup_sqm / total_surveyed_sqm * 100.0), 1) if total_surveyed_sqm > 0 else 0.0

        summary = {
            "analysis_id": analysis_id,
            "image_name": os.path.basename(image_path),
            "image_dims": [w_orig, h_orig],
            "total_parcels": len(parcel_records),
            "total_buildings": len(building_records),
            "total_roads": len(road_records),
            "total_surveyed_area_sqm": round(total_surveyed_sqm, 2),
            "total_surveyed_hectares": round(total_surveyed_sqm / 10000.0, 4),
            "total_builtup_sqm": round(total_builtup_sqm, 2),
            "overall_builtup_pct": builtup_pct,
            "gsd_meters": gsd_meters,
            "processing_time_sec": duration,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Serialized parcel properties without non-serializable geometry
        clean_parcels_props = []
        for p in parcel_records:
            clean_p = {k: v for k, v in p.items() if k not in ["geometry", "coord_transform"]}
            clean_parcels_props.append(clean_p)

        result_payload = {
            "success": True,
            "summary": summary,
            "overlay_url": f"/outputs/{analysis_id}/{overlay_filename}",
            "geojson_url": f"/outputs/{analysis_id}/cadastral_features.geojson",
            "csv_url": f"/outputs/{analysis_id}/cadastral_register.csv",
            "parcels": clean_parcels_props,
            "geojson_pixel": geojson_pixel,
            "geojson_wgs84": geojson_geo
        }

        # Save metadata JSON
        meta_path = os.path.join(out_subfolder, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)

        self.cache[analysis_id] = result_payload
        return result_payload

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves analysis result from cache or disk."""
        if analysis_id in self.cache:
            return self.cache[analysis_id]

        meta_path = os.path.join(self.outputs_dir, analysis_id, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.cache[analysis_id] = data
                return data
        return None

    def update_parcel(
        self,
        analysis_id: str,
        parcel_id: str,
        status: Optional[str] = None,
        land_use: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Updates parcel verification status or land-use classification."""
        analysis = self.get_analysis(analysis_id)
        if not analysis:
            return None

        updated_parcel = None
        for p in analysis.get("parcels", []):
            if p.get("parcel_id") == parcel_id:
                if status:
                    p["verification_status"] = status
                if land_use:
                    p["land_use"] = land_use
                updated_parcel = p
                break

        # Also update in geojson_wgs84 and geojson_pixel
        for gj_key in ["geojson_wgs84", "geojson_pixel"]:
            if gj_key in analysis and "features" in analysis[gj_key]:
                for f in analysis[gj_key]["features"]:
                    if f.get("id") == parcel_id or f.get("properties", {}).get("parcel_id") == parcel_id:
                        if status:
                            f["properties"]["verification_status"] = status
                        if land_use:
                            f["properties"]["land_use"] = land_use

        # Re-save metadata, geojson, and csv
        out_subfolder = os.path.join(self.outputs_dir, analysis_id)
        meta_path = os.path.join(out_subfolder, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)

        geojson_path = os.path.join(out_subfolder, "cadastral_features.geojson")
        if "geojson_wgs84" in analysis:
            self.exporter.save_geojson(analysis["geojson_wgs84"], geojson_path)

        csv_path = os.path.join(out_subfolder, "cadastral_register.csv")
        self.exporter.save_csv_register(analysis.get("parcels", []), csv_path)

        return updated_parcel

