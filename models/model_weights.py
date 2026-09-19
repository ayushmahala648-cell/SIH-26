"""
Weight management, calibration, and initialization for Cadastral Deep Learning models.
Allows saving and loading PyTorch weights (.pt) and provides calibrated aerial feature filters.
"""

import os
import math
import torch
import torch.nn as nn
from typing import Optional


WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def initialize_aerial_weights(model: nn.Module) -> None:
    """
    Initializes network weights with Kaiming He normal initialization for Conv layers,
    and configures the initial RGB feature filters to be sensitive to aerial
    cadastral properties (structural edges, spectral vegetation, roof color contrast).
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # Calibrate initial convolution layer with structured aerial edge & spectral filters
    with torch.no_grad():
        if hasattr(model, 'inc') and hasattr(model.inc, 'double_conv'):
            first_conv = model.inc.double_conv[0]
            if first_conv.weight.shape[1] == 3:
                # Set specific filters to act as luminance, green-excess (vegetation),
                # edge detectors (Sobel x/y), and roof-contrast filters
                w = first_conv.weight.data
                # Filter 0: Grayscale luminance
                w[0] = torch.tensor([[[0.299]], [[0.587]], [[0.114]]]).repeat(1, 1, 3, 3) / 9.0
                # Filter 1: Excess Green index (2*G - R - B) for vegetation detection
                w[1] = torch.tensor([[[-1.0]], [[2.0]], [[-1.0]]]).repeat(1, 1, 3, 3) / 9.0
                # Filter 2: Horizontal Sobel Edge
                sobel_h = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 8.0
                w[2] = sobel_h.unsqueeze(0).repeat(3, 1, 1)
                # Filter 3: Vertical Sobel Edge
                sobel_v = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
                w[3] = sobel_v.unsqueeze(0).repeat(3, 1, 1)
                # Filter 4: Laplacian edge detector for building corners
                laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]) / 4.0
                w[4] = laplacian.unsqueeze(0).repeat(3, 1, 1)


def save_checkpoint(model: nn.Module, filename: str = "cadastral_unet.pt") -> str:
    """Saves model weights to checkpoints directory."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    filepath = os.path.join(WEIGHTS_DIR, filename)
    torch.save(model.state_dict(), filepath)
    return filepath


def load_checkpoint(model: nn.Module, filename: str = "cadastral_unet.pt", device: str = "cpu") -> bool:
    """Loads weights from checkpoints directory if available."""
    filepath = os.path.join(WEIGHTS_DIR, filename)
    if os.path.exists(filepath):
        state_dict = torch.load(filepath, map_location=device)
        model.load_state_dict(state_dict)
        return True
    return False

