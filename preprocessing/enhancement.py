"""
Radiometric enhancement and normalization for aerial drone imagery.
Handles varying illumination, sunlight reflections, and cast shadows.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


def load_image_rgb(image_path: str) -> np.ndarray:
    """
    Load an image from disk and return as RGB numpy array.
    """
    image = cv2.imread(image_path)
    if image is None:
        # Fallback to PIL in case OpenCV has issues with unicode/path
        pil_img = Image.open(image_path).convert("RGB")
        return np.array(pil_img)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def enhance_aerial_image(
    image_rgb: np.ndarray,
    apply_clahe: bool = True,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    gamma: float = 1.05
) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB space
    and mild gamma adjustment to accentuate parcel boundaries and structural edges
    in aerial imagery without distorting color balance.
    """
    enhanced = image_rgb.copy()

    if apply_clahe:
        # Convert RGB to LAB color space
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l_channel)

        # Merge channels and convert back to RGB
        limg = cv2.merge((cl, a_channel, b_channel))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

    return enhanced


def create_thumbnail(image_rgb: np.ndarray, max_size: int = 512) -> np.ndarray:
    """
    Generate an aspect-ratio-preserving thumbnail for quick preview.
    """
    h, w = image_rgb.shape[:2]
    scale = min(max_size / h, max_size / w)
    if scale >= 1.0:
        return image_rgb.copy()
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

