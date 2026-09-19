"""
Unified Cadastral Feature Extraction Pipeline.
Combines Deep Learning Semantic Segmentation, Multi-scale Boundary Affinity,
and Marker-Controlled Watershed for Topological Urban Parcel Partitioning.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from preprocessing.enhancement import enhance_aerial_image, load_image_rgb
from preprocessing.tiling import TileEngine
from models.segmenter import CadastralUNet, get_segmentation_model
from models.boundary_detector import CadastralBoundaryDetector
from models.model_weights import initialize_aerial_weights, load_checkpoint


class CadastralPipeline:
    def __init__(
        self,
        device: Optional[str] = None,
        tile_size: int = 512,
        tile_overlap: int = 64
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = get_segmentation_model(self.device)
        # Attempt to load checkpoint if exists, otherwise initialize calibrated aerial filters
        if not load_checkpoint(self.model, device=self.device):
            initialize_aerial_weights(self.model)

        self.boundary_detector = CadastralBoundaryDetector()
        self.tiler = TileEngine(tile_size=tile_size, overlap=tile_overlap)

    def _preprocess_tensor(self, img_rgb: np.ndarray) -> torch.Tensor:
        """Converts RGB image [0..255] uint8 to normalized float PyTorch tensor."""
        img_f = img_rgb.astype(np.float32) / 255.0
        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (img_f - mean) / std
        tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def _predict_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for a single tile."""
        tensor = self._preprocess_tensor(tile_rgb)
        with torch.no_grad():
            logits, boundary_logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return probs

    def predict_semantic_maps(self, image_rgb: np.ndarray, max_dim: int = 1280) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Runs neural inference across the image.
        If the image is larger than max_dim, scales it for efficient processing while
        preserving scale factor for high-resolution vectorization.
        Returns:
            class_probs: [H, W, 5] (Background, Boundary, Building, Road, Vegetation)
            boundary_map: [H, W] float affinity
            scale_factor: float used to scale original image to processing size
        """
        h_orig, w_orig = image_rgb.shape[:2]
        scale = 1.0
        if max(h_orig, w_orig) > max_dim:
            scale = max_dim / float(max(h_orig, w_orig))
            proc_w, proc_h = int(w_orig * scale), int(h_orig * scale)
            proc_img = cv2.resize(image_rgb, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            proc_img = image_rgb.copy()

        h, w = proc_img.shape[:2]

        # Use tiling if dimension > 640, else single pass
        if h > 640 or w > 640:
            def tile_gen():
                for y1, y2, x1, x2, tile in self.tiler.get_tiles(proc_img):
                    yield y1, y2, x1, x2, self._predict_tile(tile)

            class_probs = self.tiler.stitch_predictions(tile_gen(), (h, w, CadastralUNet.NUM_CLASSES))
        else:
            class_probs = self._predict_tile(proc_img)

        # Morphological boundary affinity
        morph_boundary = self.boundary_detector.extract_boundary_map(proc_img)
        
        # Fuse neural boundary prediction with morphological boundary detector
        neural_boundary = class_probs[:, :, 1]
        fused_boundary = 0.55 * morph_boundary + 0.45 * neural_boundary
        fused_boundary = np.clip(fused_boundary, 0.0, 1.0)

        return class_probs, fused_boundary, scale

    def extract_cadastral_features(
        self,
        image_input: Any,
        min_parcel_area: int = 120,
        min_building_area: int = 50,
        watershed_compactness: float = 0.001
    ) -> Dict[str, Any]:
        """
        Full cadastral extraction pipeline:
        1. Enhance aerial contrast (CLAHE)
        2. Neural prediction + boundary affinity
        3. Marker-controlled watershed parcel segmentation
        4. Building footprint & road corridor extraction
        5. Returns labeled parcel masks, feature masks, and debug visuals.
        """
        if isinstance(image_input, str):
            image_rgb = load_image_rgb(image_input)
        else:
            image_rgb = image_input

        # Step 1: Preprocessing & Radiometric Enhancement
        enhanced_rgb = enhance_aerial_image(image_rgb, apply_clahe=True, clip_limit=2.2)

        # Step 2: Semantic Segmentation & Boundary Extraction
        class_probs, boundary_affinity, scale = self.predict_semantic_maps(enhanced_rgb)
        h, w = boundary_affinity.shape[:2]

        if scale != 1.0:
            proc_rgb = cv2.resize(enhanced_rgb, (w, h), interpolation=cv2.INTER_AREA)
        else:
            proc_rgb = enhanced_rgb

        # Step 3: Extract Spectral Vegetation & Structure Indicators
        # Excess Green Index: 2G - R - B
        r, g, b = proc_rgb[:, :, 0].astype(float), proc_rgb[:, :, 1].astype(float), proc_rgb[:, :, 2].astype(float)
        exg = (2.0 * g - r - b) / (r + g + b + 1e-5)
        veg_mask = (exg > 0.12).astype(np.uint8) * 255
        # Also incorporate neural vegetation channel
        neural_veg = (class_probs[:, :, 4] > 0.35).astype(np.uint8) * 255
        final_veg_mask = cv2.bitwise_or(veg_mask, neural_veg)

        # Step 4: Extract Building Footprints
        # Buildings have distinct contrast, roof textures, and sharp borders
        gray = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2GRAY)
        neural_bldg = class_probs[:, :, 2]
        
        # Adaptive thresholding for roof structures
        bldg_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -2
        )
        # Combine neural building confidence with morphological structure
        combined_bldg = (neural_bldg * 255).astype(np.uint8)
        combined_bldg = cv2.bitwise_or(combined_bldg, cv2.bitwise_and(bldg_thresh, (neural_bldg > 0.25).astype(np.uint8) * 255))
        
        # Morphological clean up: close small gaps inside roofs, remove noise
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bldg_clean = cv2.morphologyEx(combined_bldg, cv2.MORPH_CLOSE, kernel_rect, iterations=2)
        _, bldg_binary = cv2.threshold(bldg_clean, 80, 255, cv2.THRESH_BINARY)
        # Exclude pure dense vegetation from buildings
        bldg_binary[final_veg_mask > 200] = 0

        # Step 5: Extract Roads / Corridors
        road_prob = class_probs[:, :, 3]
        road_binary = (road_prob > 0.30).astype(np.uint8) * 255
        # Roads are elongated and non-building
        road_binary[bldg_binary > 0] = 0

        # Step 6: Marker-Controlled Watershed for Parcel Lot Segmentation
        # Invert boundary affinity to find parcel interior basins
        boundary_u8 = (boundary_affinity * 255).astype(np.uint8)
        _, boundary_thresh = cv2.threshold(boundary_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Distance transform on non-boundary pixels
        non_boundary = cv2.bitwise_not(boundary_thresh)
        dist_transform = cv2.distanceTransform(non_boundary, cv2.DIST_L2, 5)

        # Identify parcel seed markers from distance peaks
        dist_max = dist_transform.max()
        if dist_max > 0:
            _, markers_binary = cv2.threshold(dist_transform, 0.18 * dist_max, 255, 0)
        else:
            markers_binary = np.zeros_like(non_boundary)

        markers_binary = np.uint8(markers_binary)
        
        # Connected components labeling on markers
        num_markers, markers = cv2.connectedComponents(markers_binary)
        # Add 1 to all labels so background is 1 instead of 0
        markers = markers + 1
        # Mark unknown boundary regions as 0 for watershed
        markers[boundary_thresh > 128] = 0

        # Run Watershed algorithm
        proc_bgr = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR)
        markers = cv2.watershed(proc_bgr, markers)

        # Result: markers == -1 are watershed boundaries, > 1 are individual parcel segments
        parcel_labels = np.zeros_like(markers, dtype=np.int32)
        valid_mask = (markers > 1)
        parcel_labels[valid_mask] = markers[valid_mask] - 1

        # Re-index parcels to be compact 1..N and filter tiny slivers
        unique_labels = np.unique(parcel_labels)
        filtered_labels = np.zeros_like(parcel_labels)
        new_id = 1
        for lbl in unique_labels:
            if lbl == 0:
                continue
            lbl_mask = (parcel_labels == lbl)
            if np.sum(lbl_mask) >= min_parcel_area:
                filtered_labels[lbl_mask] = new_id
                new_id += 1

        # Step 7: Create Rich Annotated Visual Overlay
        overlay = proc_rgb.copy()
        
        # Draw road network in bright yellow
        overlay[road_binary > 0] = (overlay[road_binary > 0] * 0.4 + np.array([255, 235, 59]) * 0.6).astype(np.uint8)
        
        # Draw building footprints in warm orange
        overlay[bldg_binary > 0] = (overlay[bldg_binary > 0] * 0.35 + np.array([255, 112, 67]) * 0.65).astype(np.uint8)
        
        # Draw parcel boundaries in cyan / blue
        parcel_edges = (markers == -1) | (boundary_thresh > 180)
        overlay[parcel_edges] = np.array([0, 229, 255], dtype=np.uint8)

        return {
            "parcel_labels": filtered_labels,
            "total_parcels": new_id - 1,
            "building_mask": bldg_binary,
            "road_mask": road_binary,
            "boundary_mask": boundary_thresh,
            "vegetation_mask": final_veg_mask,
            "boundary_affinity": boundary_affinity,
            "enhanced_image": enhanced_rgb,
            "annotated_overlay": overlay,
            "scale_factor": scale,
            "image_dims": (image_rgb.shape[1], image_rgb.shape[0])
        }
