"""Build upper-canopy prompts and run optional photo segmentation."""

import numpy as np
from scipy import ndimage

from .geometry import center_piece


def canopy_box(x, y, height, center, origin, shape, p99):
    """Return the padded upper-canopy box, footprint, and height threshold."""
    west, north, res = origin
    n, width = shape
    k = int(round(0.1 / res))
    if n != width or k < 1 or n % k or not np.isclose(k * res, 0.1):
        raise ValueError("Crop grid must divide into exact 0.1 m cells")
    threshold = max(0.25, 0.5 * p99)
    selected = (
        (np.hypot(x - center[0], y - center[1]) <= 1.5)
        & (height >= threshold)
        & (height <= 6)
    )
    g = n // k
    footprint = np.zeros((g, g), dtype=bool)
    cols = ((x[selected] - west) / (res * k)).astype(int)
    rows = ((north - y[selected]) / (res * k)).astype(int)
    valid = (cols >= 0) & (cols < g) & (rows >= 0) & (rows < g)
    footprint[rows[valid], cols[valid]] = True
    footprint = center_piece(ndimage.binary_closing(footprint, iterations=1))
    box = None
    if footprint.any():
        rows, cols = np.nonzero(footprint)
        box = np.array(
            [
                cols.min() * k - 20,
                rows.min() * k - 20,
                (cols.max() + 1) * k + 20,
                (rows.max() + 1) * k + 20,
            ]
        ).clip(0, n - 1)
    return box, np.kron(footprint, np.ones((k, k), bool)), threshold


def load_predictor(checkpoint, device=None):
    """Load the checkpoint lazily, preferring MPS when available."""
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SamPredictor(
        sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to(device)
    )


def predict_masks(predictor, image, box):
    """Request multiple masks with a positive center point and optional box."""
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[image.shape[1] / 2, image.shape[0] / 2]]),
        point_labels=np.array([1]),
        box=box,
        multimask_output=True,
    )
    return masks, scores
