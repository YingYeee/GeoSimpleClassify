from __future__ import annotations
import numpy as np


def extract_pixel_features(
    cube,
    mask = None
):
    """
    Extract per-pixel feature vectors from a raster cube.

    Parameters
    ----------
    cube : ndarray
        Raster data with shape (B, H, W).
    mask : ndarray of bool, optional
        Valid-pixel mask with shape (H, W). True means valid.

    Returns
    -------
    features : ndarray
        Extracted features with shape (N, B).
    idx : ndarray
        Linear indices of selected pixels.
    (h, w) : tuple
        Original shape.
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be (B,H,W), got {cube.shape}")

    b, h, w = cube.shape

    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    else:
        if mask.shape != (h, w):
            raise ValueError(f"mask shape {mask.shape} != {(h,w)}")
        mask = mask.astype(bool, copy=False)

    cube_flat = cube.reshape(b, -1).T   # (H*W, B)
    mask_flat = mask.reshape(-1)        # (H*W,)

    idx = np.where(mask_flat)[0]
    features = cube_flat[idx]           # (N,B)

    return features, idx, (h, w)