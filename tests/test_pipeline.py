"""
End-to-End System & Regression Test Suite for Paridhi Cadastral Mapping.
Verifies Preprocessing, ML Inference, Geospatial Regularization, and FastAPI Endpoints.
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing.enhancement import enhance_aerial_image, load_image_rgb
from preprocessing.tiling import TileEngine
from models.cadastral_pipeline import CadastralPipeline
from geospatial.vectorizer import CadastralVectorizer
from geospatial.regularization import CadastralRegularizer
from geospatial.georeference import CadastralGeoreferencer
from geospatial.exporter import CadastralExporter
from backend.main import app


class TestCadastralPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_img_path = os.path.join(BASE_DIR, "images", "aerialPhoto1.jpg.jpeg")
        cls.client = TestClient(app)
        cls.pipeline = CadastralPipeline(device="cpu")

    def test_01_preprocessing_enhancement(self):
        """Test image loading, CLAHE enhancement, and normalization."""
        self.assertTrue(os.path.exists(self.sample_img_path), "Sample image must exist")
        img_rgb = load_image_rgb(self.sample_img_path)
        self.assertEqual(len(img_rgb.shape), 3)
        self.assertEqual(img_rgb.shape[2], 3)

        enhanced = enhance_aerial_image(img_rgb, apply_clahe=True, clip_limit=2.0)
        self.assertEqual(enhanced.shape, img_rgb.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_02_tiling_engine(self):
        """Test sliding window tiler and overlap reconstruction."""
        dummy_img = np.ones((800, 800, 3), dtype=np.uint8) * 128
        tiler = TileEngine(tile_size=512, overlap=64)
        tiles = list(tiler.get_tiles(dummy_img))
        self.assertGreater(len(tiles), 1, "Should generate multiple overlapping tiles for 800x800 image")

    def test_03_ml_inference_and_watershed(self):
        """Test deep segmentation, boundary affinity, and watershed parcel partition."""
        res = self.pipeline.extract_cadastral_features(self.sample_img_path, min_parcel_area=80)
        self.assertIn("parcel_labels", res)
        self.assertIn("total_parcels", res)
        self.assertGreater(res["total_parcels"], 5, "Should identify distinct urban parcels")
        self.assertIn("building_mask", res)
        self.assertIn("annotated_overlay", res)

    def test_04_geospatial_vectorization_and_topology(self):
        """Test vectorization of raster masks into valid Shapely polygons and topology enforcement."""
        res = self.pipeline.extract_cadastral_features(self.sample_img_path, min_parcel_area=80)
        vec = CadastralVectorizer()
        parcel_polys = vec.extract_parcel_polygons(res["parcel_labels"], scale_back=res["scale_factor"])
        self.assertGreater(len(parcel_polys), 0, "Should convert labels to polygons")

        reg = CadastralRegularizer(parcel_simplify_tol=2.0)
        clean_parcels = reg.enforce_parcel_topology(parcel_polys)
        for pid, poly in clean_parcels.items():
            self.assertTrue(poly.is_valid, f"Parcel {pid} polygon must be geometrically valid")

    def test_05_georeference_and_metrics(self):
        """Test Ground Sample Distance area/perimeter calculations and land use classification."""
        georef = CadastralGeoreferencer(gsd_meters=0.08)
        from shapely.geometry import box
        test_box = box(0, 0, 100, 100) # 100px x 100px = 10,000 px^2 => 64 m^2 at 0.08m/px
        metrics = georef.compute_parcel_metrics(1, test_box)
        self.assertEqual(metrics["parcel_id"], "P-0001")
        self.assertAlmostEqual(metrics["area_sqm"], 64.0, delta=0.5)
        self.assertGreater(metrics["area_gaj"], 0)
        self.assertIn(metrics["land_use"], ["Residential", "Commercial / Institutional", "Agricultural / Green Space", "Vacant / Open Plot", "Mixed Use"])

    def test_06_fastapi_endpoints(self):
        """Test FastAPI endpoints for health, sample images, analyze, parcel update, and export."""
        # 1. Health check
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "online")

        # 2. Sample images
        r_samples = self.client.get("/api/sample-images")
        self.assertEqual(r_samples.status_code, 200)
        samples = r_samples.json().get("samples", [])
        self.assertGreater(len(samples), 0)

        # 3. Analyze endpoint
        payload = {
            "image_name": "aerialPhoto1.jpg.jpeg",
            "gsd_meters": 0.08,
            "simplify_tolerance": 2.0,
            "min_parcel_area_px": 100,
            "snap_orthogonal": True
        }
        r_analyze = self.client.post("/api/analyze", json=payload)
        self.assertEqual(r_analyze.status_code, 200)
        data = r_analyze.json()
        self.assertTrue(data["success"])
        analysis_id = data["summary"]["analysis_id"]

        # 4. Parcel update verification endpoint
        first_pid = data["parcels"][0]["parcel_id"]
        r_update = self.client.post("/api/parcels/update", json={
            "analysis_id": analysis_id,
            "parcel_id": first_pid,
            "verification_status": "Verified",
            "land_use": "Residential"
        })
        self.assertEqual(r_update.status_code, 200)
        self.assertEqual(r_update.json()["updated_parcel"]["verification_status"], "Verified")

        # 5. Export GeoJSON
        r_geo = self.client.get(f"/api/export/{analysis_id}/geojson")
        self.assertEqual(r_geo.status_code, 200)
        self.assertIn("FeatureCollection", r_geo.text)

        # 6. Export CSV
        r_csv = self.client.get(f"/api/export/{analysis_id}/csv")
        self.assertEqual(r_csv.status_code, 200)
        self.assertIn("parcel_id", r_csv.text)


if __name__ == "__main__":
    unittest.main()

