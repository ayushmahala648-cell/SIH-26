"""
Sliding-window tiling engine for large-scale drone orthomosaics.
Enables processing of gigapixel aerial imagery without GPU/RAM OOM errors,
using smooth 2D Gaussian/Hann overlap blending to prevent tile border artifacts.
"""

import numpy as np
from typing import List, Tuple, Generator, Callable


class TileEngine:
    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = max(0, min(overlap, tile_size // 2))
        self.step = self.tile_size - self.overlap
        self._blend_window = self._create_2d_window(self.tile_size)

    def _create_2d_window(self, size: int) -> np.ndarray:
        """
        Creates a 2D Hann window for smooth weighted blending at tile seams.
        """
        w1d = np.hanning(size)
        # Avoid zeros at exact edges by clamping to small epsilon
        w1d = np.clip(w1d, 0.05, 1.0)
        w2d = np.outer(w1d, w1d)
        return w2d.astype(np.float32)

    def get_tiles(self, image: np.ndarray) -> Generator[Tuple[int, int, int, int, np.ndarray], None, None]:
        """
        Yields (y1, y2, x1, x2, tile) coordinates and tiles for an image.
        Pads the image if needed.
        """
        h, w = image.shape[:2]
        y_points = list(range(0, max(1, h - self.tile_size + 1), self.step))
        if len(y_points) == 0 or y_points[-1] + self.tile_size < h:
            y_points.append(max(0, h - self.tile_size))

        x_points = list(range(0, max(1, w - self.tile_size + 1), self.step))
        if len(x_points) == 0 or x_points[-1] + self.tile_size < w:
            x_points.append(max(0, w - self.tile_size))

        for y in y_points:
            for x in x_points:
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                tile = image[y:y_end, x:x_end]

                # If tile is smaller than tile_size at right/bottom edges, pad it
                th, tw = tile.shape[:2]
                if th < self.tile_size or tw < self.tile_size:
                    padded_tile = np.zeros((self.tile_size, self.tile_size, image.shape[2]), dtype=image.dtype)
                    padded_tile[:th, :tw] = tile
                    yield y, y_end, x, x_end, padded_tile
                else:
                    yield y, y_end, x, x_end, tile

    def stitch_predictions(
        self,
        tile_preds_generator: Generator[Tuple[int, int, int, int, np.ndarray], None, None],
        out_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Stitches multi-channel probability predictions back into a seamless full-sized map.
        out_shape: (height, width, num_classes)
        """
        h, w, c = out_shape
        full_pred = np.zeros((h, w, c), dtype=np.float32)
        weight_acc = np.zeros((h, w, 1), dtype=np.float32)

        window_3d = self._blend_window[:, :, np.newaxis]

        for y1, y2, x1, x2, pred_tile in tile_preds_generator:
            th = y2 - y1
            tw = x2 - x1
            cropped_pred = pred_tile[:th, :tw]
            cropped_win = window_3d[:th, :tw]

            full_pred[y1:y2, x1:x2] += cropped_pred * cropped_win
            weight_acc[y1:y2, x1:x2] += cropped_win

        weight_acc[weight_acc == 0] = 1.0
        full_pred /= weight_acc
        return full_pred

