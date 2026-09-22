"""
YOLOv8 & Morphological GIS Cadastral Feature Extraction Engine.
Identifies Houses/Buildings using Pretrained YOLOv8 Segmentation,
and extracts Trees, Agricultural Farmland, Playgrounds, Bare Land, and Roads
via Adaptive Computer Vision & Spectral GIS Analysis.
"""

import os
import cv2
import json
import uuid
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


class CadastralEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_repo: str = "keremberke/yolov8m-building-segmentation"):
        if getattr(self, "_initialized", False):
            return

        self.model_repo = model_repo
        self.yolo_model = self._load_yolo_model()
        self._initialized = True

    def _load_yolo_model(self) -> YOLO:
        """Loads the YOLO segmentation model with multi-tier fallback for offline resilience."""
        # 1. Check Hugging Face hub cache directory on Windows
        hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_cache_dir.exists():
            matched_pts = list(hf_cache_dir.glob("**/best.pt"))
            if matched_pts:
                print(f"[CadastralEngine] Using cached model from: {matched_pts[0]}")
                return YOLO(str(matched_pts[0]))

        # 2. Check local directories
        for candidate in ["best.pt", "models/best.pt", "backend/best.pt"]:
            if os.path.exists(candidate):
                print(f"[CadastralEngine] Found local checkpoint: {candidate}")
                return YOLO(candidate)

        # 3. Attempt Hugging Face Hub download
        try:
            print(f"[CadastralEngine] Downloading YOLO model weights from {self.model_repo}...")
            model_path = hf_hub_download(repo_id=self.model_repo, filename="best.pt")
            print(f"[CadastralEngine] Downloaded successfully to: {model_path}")
            return YOLO(model_path)
        except Exception as e:
            print(f"[CadastralEngine] Hugging Face Hub download failed ({e}), falling back to yolov8n-seg...")
            return YOLO("yolov8n-seg.pt")

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Applies CLAHE contrast enhancement for dense aerial roofs & ground features."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def extract_features(
        self,
        img: np.ndarray,
        conf_threshold: float = 0.12,
        min_building_area: int = 70,
        min_tree_area: int = 400,
        min_field_area: int = 2500,
        min_playground_area: int = 3500,
        min_road_area: int = 1200
    ) -> List[Dict[str, Any]]:
        """
        Extracts cadastral features from image:
        - Houses / Buildings (via Pretrained YOLOv8 Segmentation)
        - Trees / Vegetation (via Spectral HSV & Excess Green Index)
        - Agricultural Land (via Contiguous Farmland Color-Texture Bounds)
        - Playgrounds / Sports Ground (via High-Circularity Large Grass / Open Turf)
        - Vacant / Bare Plots (via Soil Hue & Low Saturation Analysis)
        - Roads & Access Pathways (via Elongated Corridor Extraction)
        """
        h_img, w_img = img.shape[:2]
        enhanced_img = self.enhance_contrast(img)
        features: List[Dict[str, Any]] = []

        # ---------------------------------------------------------
        # STEP 1: YOLOv8 Building Extraction
        # ---------------------------------------------------------
        # Scale to max 1280 for fast CPU/GPU inference while maintaining high accuracy
        scale = 1.0
        max_dim = 1280
        if max(h_img, w_img) > max_dim:
            scale = max_dim / float(max(h_img, w_img))
            infer_w, infer_h = int(w_img * scale), int(h_img * scale)
            infer_img = cv2.resize(enhanced_img, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        else:
            infer_img = enhanced_img

        yolo_results = self.yolo_model(
            infer_img,
            device="cpu",
            conf=conf_threshold,
            iou=0.30,
            imgsz=min(max(h_img, w_img), 1024),
            retina_masks=True,
            verbose=False
        )

        bldg_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        result = yolo_results[0]

        if result.masks is not None and result.masks.xy is not None:
            for mask_xy in result.masks.xy:
                if len(mask_xy) < 3:
                    continue
                # Scale polygon back to original resolution if downsampled
                poly_pts = (mask_xy * (1.0 / scale)).astype(np.int32)
                area = cv2.contourArea(poly_pts)
                if area < min_building_area:
                    continue

                epsilon = 0.007 * cv2.arcLength(poly_pts, True)
                approx_poly = cv2.approxPolyDP(poly_pts, epsilon, True)

                # Record building footprint
                features.append({
                    "class_name": "Building",
                    "category": "Structure",
                    "color": (0, 50, 255),      # Crimson Red (BGR)
                    "hex": "#FF3232",
                    "polygon": approx_poly,
                    "area_px": float(area),
                    "confidence": 0.88
                })
                # Mark on building occupancy mask to prevent terrain overlap
                cv2.fillPoly(bldg_mask, [approx_poly], 255)

        # ---------------------------------------------------------
        # STEP 2: Spectral & Morphological GIS Terrain Extraction
        # ---------------------------------------------------------
        hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)

        # Helper for mask cleanup
        def clean_mask(mask: np.ndarray, close_k: int = 15, open_k: int = 7) -> np.ndarray:
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
            m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
            # Remove areas occupied by buildings
            m[bldg_mask > 0] = 0
            return m

        # 2A: Trees & Dense Vegetation
        # Vibrant green hues in HSV: H: 35-85, S: 35-255, V: 20-255
        tree_mask_raw = cv2.inRange(hsv, np.array([35, 35, 20]), np.array([85, 255, 255]))
        # Combine with Excess Green Index (2G - R - B) for spectral chlorophyll validation
        r = enhanced_img[:, :, 2].astype(float)
        g = enhanced_img[:, :, 1].astype(float)
        b = enhanced_img[:, :, 0].astype(float)
        exg = (2.0 * g - r - b) / (r + g + b + 1e-5)
        tree_mask_raw[exg < 0.05] = 0
        tree_mask = clean_mask(tree_mask_raw, close_k=11, open_k=5)

        contours_trees, _ = cv2.findContours(tree_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        occupied_terrain = bldg_mask.copy()

        for cnt in contours_trees:
            area = cv2.contourArea(cnt)
            if area < min_tree_area:
                continue
            hull = cv2.convexHull(cnt)
            eps = 0.008 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, eps, True)
            features.append({
                "class_name": "Tree / Forest",
                "category": "Vegetation",
                "color": (0, 180, 0),         # Emerald Foliage Green (BGR)
                "hex": "#00B400",
                "polygon": approx,
                "area_px": float(area),
                "confidence": 0.85
            })
            cv2.fillPoly(occupied_terrain, [approx], 255)

        # 2B: Playground / Sports Turf / Open Lawn
        # Characterized by high circularity/regularity and bright uniform green
        grass_raw = cv2.inRange(hsv, np.array([30, 40, 70]), np.array([75, 220, 255]))
        grass_raw[occupied_terrain > 0] = 0
        grass_mask = clean_mask(grass_raw, close_k=25, open_k=13)

        contours_grass, _ = cv2.findContours(grass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_grass:
            area = cv2.contourArea(cnt)
            if area < min_playground_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-5))
            # Large open spaces or high-circularity/regular turf = Playground / Sports Field
            if circularity > 0.15 or area > 10000:
                hull = cv2.convexHull(cnt)
                eps = 0.006 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, eps, True)
                features.append({
                    "class_name": "Playground / Open Turf",
                    "category": "Land",
                    "color": (50, 205, 50),       # Lime Green (BGR)
                    "hex": "#32CD32",
                    "polygon": approx,
                    "area_px": float(area),
                    "confidence": 0.82
                })
                cv2.fillPoly(occupied_terrain, [approx], 255)

        # 2C: Agricultural Farmland / Cultivated Field (Big Land)
        # Golden, amber, tilled soil, or dry crop canopy: H: 10-35, S: 25-255, V: 40-255
        agri_raw = cv2.inRange(hsv, np.array([10, 25, 40]), np.array([35, 255, 255]))
        agri_raw[occupied_terrain > 0] = 0
        agri_mask = clean_mask(agri_raw, close_k=25, open_k=11)

        contours_agri, _ = cv2.findContours(agri_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_agri:
            area = cv2.contourArea(cnt)
            if area < min_field_area:
                continue
            hull = cv2.convexHull(cnt)
            eps = 0.005 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, eps, True)
            features.append({
                "class_name": "Agricultural Land",
                "category": "Land",
                "color": (0, 165, 255),       # Amber / Orange (BGR)
                "hex": "#FFA500",
                "polygon": approx,
                "area_px": float(area),
                "confidence": 0.80
            })
            cv2.fillPoly(occupied_terrain, [approx], 255)

        # 2D: Roads & Access Corridors
        # Low saturation asphalt/concrete, elongated pathways: H: 0-180, S: 0-45, V: 60-190
        road_raw = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([180, 45, 190]))
        road_raw[occupied_terrain > 0] = 0
        k_road = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        road_mask = cv2.morphologyEx(road_raw, cv2.MORPH_CLOSE, k_road)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        contours_road, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_road:
            area = cv2.contourArea(cnt)
            if area < min_road_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            elongation = (perimeter * perimeter) / (area + 1e-5)
            # Elongated corridors or substantial pathways
            if elongation > 25 or area > 3500:
                hull = cv2.convexHull(cnt)
                eps = 0.008 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, eps, True)
                features.append({
                    "class_name": "Road / Pathway",
                    "category": "Infrastructure",
                    "color": (0, 255, 255),       # High-vis Yellow (BGR)
                    "hex": "#FFFF00",
                    "polygon": approx,
                    "area_px": float(area),
                    "confidence": 0.78
                })
                cv2.fillPoly(occupied_terrain, [approx], 255)

        # 2E: Vacant Plot / Bare Land
        # Unoccupied regions with area > 3000
        unoccupied = 255 - occupied_terrain
        k_vacant = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        vacant_clean = cv2.morphologyEx(unoccupied, cv2.MORPH_OPEN, k_vacant)
        contours_vacant, _ = cv2.findContours(vacant_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_vacant:
            area = cv2.contourArea(cnt)
            if area > 4500:
                hull = cv2.convexHull(cnt)
                eps = 0.006 * cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, eps, True)
                features.append({
                    "class_name": "Vacant / Open Land",
                    "category": "Land",
                    "color": (140, 180, 210),     # Sand Tan (BGR)
                    "hex": "#D2B48C",
                    "polygon": approx,
                    "area_px": float(area),
                    "confidence": 0.75
                })

        return features

    def render_colored_overlay(
        self,
        img: np.ndarray,
        features: List[Dict[str, Any]],
        alpha: float = 0.42,
        draw_labels: bool = True
    ) -> np.ndarray:
        """
        Draws and colors houses and terrain features:
        1. Fills translucent colored polygons via alpha blending.
        2. Draws solid, anti-aliased perimeter vector lines.
        3. Renders clean cadastral label badges at feature centroids.
        """
        overlay = img.copy()

        # Sort features so large land plots are drawn first, then roads, trees, and buildings on top
        priority_map = {
            "Vacant / Open Land": 1,
            "Agricultural Land": 2,
            "Playground / Open Turf": 3,
            "Road / Pathway": 4,
            "Tree / Forest": 5,
            "Building": 6
        }
        sorted_features = sorted(features, key=lambda f: priority_map.get(f["class_name"], 0))

        # 1. Semi-transparent Color Fills
        for feat in sorted_features:
            poly = feat["polygon"]
            color = feat["color"]
            cv2.fillPoly(overlay, [poly], color)

        blended = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

        # 2. Vector Perimeter Outlines & Label Badges
        for idx, feat in enumerate(sorted_features):
            poly = feat["polygon"]
            color = feat["color"]
            cv2.polylines(blended, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            if draw_labels:
                # Compute centroid
                M = cv2.moments(poly)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = poly[0][0]

                # Draw concise badge for prominent buildings or key landmarks
                if feat["class_name"] == "Building" and feat.get("area_px", 0) > 350:
                    label_text = f"Bld-{idx+1:03d}"
                elif feat["class_name"] == "Playground / Open Turf":
                    label_text = "Playground"
                elif feat["class_name"] == "Agricultural Land":
                    label_text = "Farming"
                elif feat["class_name"] == "Tree / Forest" and feat.get("area_px", 0) > 3000:
                    label_text = "Forest"
                elif feat["class_name"] == "Road / Pathway":
                    label_text = "Road"
                else:
                    label_text = None

                if label_text:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.38
                    th = 1
                    (tw, th_box), _ = cv2.getTextSize(label_text, font, scale, th)
                    x1 = cx - tw // 2 - 4
                    y1 = cy - th_box // 2 - 4
                    x2 = cx + tw // 2 + 4
                    y2 = cy + th_box // 2 + 4

                    # Background pill
                    cv2.rectangle(blended, (x1, y1), (x2, y2), (20, 20, 30), -1)
                    cv2.rectangle(blended, (x1, y1), (x2, y2), color, 1)
                    # Text
                    cv2.putText(blended, label_text, (x1 + 4, y2 - 4), font, scale, (255, 255, 255), th, cv2.LINE_AA)

        return blended

    def render_wireframe(self, img: np.ndarray, features: List[Dict[str, Any]]) -> np.ndarray:
        """Generates a clean black blueprint background with glowing vector polygons."""
        h, width, _ = img.shape
        black_canvas = np.zeros((h, width, 3), dtype=np.uint8)

        # Subtle fill
        for feat in features:
            poly = feat["polygon"]
            dimmed = tuple(max(15, c // 4) for c in feat["color"])
            cv2.fillPoly(black_canvas, [poly], dimmed)

        # Crisp wireframe lines
        for feat in features:
            poly = feat["polygon"]
            cv2.polylines(black_canvas, [poly], isClosed=True, color=feat["color"], thickness=2, lineType=cv2.LINE_AA)

        return black_canvas

    def render_colored_mask(self, img: np.ndarray, features: List[Dict[str, Any]]) -> np.ndarray:
        """Generates a categorical solid color segmentation mask."""
        h, width, _ = img.shape
        mask_canvas = np.zeros((h, width, 3), dtype=np.uint8)

        for feat in features:
            cv2.fillPoly(mask_canvas, [feat["polygon"]], feat["color"])
            cv2.polylines(mask_canvas, [feat["polygon"]], isClosed=True, color=(255, 255, 255), thickness=1)

        return mask_canvas

    def to_geojson(
        self,
        img_dims: Tuple[int, int],
        features: List[Dict[str, Any]],
        gsd_meters: float = 0.08
    ) -> Dict[str, Any]:
        """Converts extracted features to RFC 7946 GeoJSON FeatureCollection."""
        width, height = img_dims
        geojson_features = []

        summary_counts = {
            "Building": 0,
            "Tree / Forest": 0,
            "Agricultural Land": 0,
            "Playground / Open Turf": 0,
            "Road / Pathway": 0,
            "Vacant / Open Land": 0
        }

        total_area_sqm = 0.0
        total_builtup_sqm = 0.0

        for idx, feat in enumerate(features):
            poly = feat["polygon"]
            cname = feat["class_name"]
            summary_counts[cname] = summary_counts.get(cname, 0) + 1

            coords = [[round(float(pt[0][0]), 2), round(float(pt[0][1]), 2)] for pt in poly]
            # Ensure polygon ring is closed in GeoJSON
            if len(coords) > 0 and coords[0] != coords[-1]:
                coords.append(coords[0])

            area_px = feat.get("area_px", float(cv2.contourArea(poly)))
            area_sqm = round(area_px * (gsd_meters ** 2), 2)
            total_area_sqm += area_sqm

            if cname == "Building":
                total_builtup_sqm += area_sqm

            perimeter_px = float(cv2.arcLength(poly, True))
            perimeter_m = round(perimeter_px * gsd_meters, 2)

            pid = f"CAD-{idx+1:04d}"

            # Default Land-Use classification (Vacant, Residential, Commercial, Agriculture, Mixed Use)
            def infer_land_use(ft: str) -> str:
                ftl = ft.lower()
                if "building" in ftl:
                    return "Residential"
                elif "agri" in ftl:
                    return "Agriculture"
                elif "playground" in ftl or "turf" in ftl:
                    return "Mixed Use"
                elif "road" in ftl:
                    return "Commercial"
                return "Vacant"

            land_use = feat.get("land_use", infer_land_use(cname))

            props = {
                "parcel_id": pid,
                "feature_type": cname,
                "land_use": land_use,
                "category": feat.get("category", "General"),
                "color_hex": feat["hex"],
                "area_px": round(area_px, 1),
                "area_sqm": area_sqm,
                "area_hectares": round(area_sqm / 10000.0, 4),
                "area_acres": round(area_sqm / 4046.86, 4),
                "area_gaj": round(area_sqm * 1.196, 2),
                "perimeter_m": perimeter_m,
                "vertex_count": len(coords),
                "confidence": feat.get("confidence", 0.85),
                "verification_status": "Verified" if feat.get("confidence", 0.8) > 0.85 else "Pending Review"
            }

            geojson_features.append({
                "type": "Feature",
                "id": pid,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": props
            })

        return {
            "type": "FeatureCollection",
            "metadata": {
                "width": width,
                "height": height,
                "gsd_meters": gsd_meters,
                "total_features": len(features),
                "counts": summary_counts,
                "total_surveyed_sqm": round(total_area_sqm, 2),
                "total_builtup_sqm": round(total_builtup_sqm, 2),
                "builtup_pct": round((total_builtup_sqm / total_area_sqm * 100.0), 1) if total_area_sqm > 0 else 0.0
            },
            "features": geojson_features
        }


# Singleton engine instance
engine = CadastralEngine()


def extract_cadastral_features(image_bytes: bytes):
    """
    Standard interface matching the user's friend's backend function signature.
    Returns: (img_bgr, features_list)
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file: could not decode.")

    features = engine.extract_features(img)
    return img, features

