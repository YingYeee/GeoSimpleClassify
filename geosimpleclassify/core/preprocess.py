# preprocess.py
# Basic preprocessing utilities for raster data.

import numpy as np


def to_float32(arr):
    """
    Convert input array to float32.
    """
    return np.asarray(arr, dtype="float32")


def normalize(arr, per_band=True):
    """
    Normalize array values to [0, 1].

    - For (H, W): normalize the single band.
    - For (B, H, W):
        * per_band=True  -> each band normalized separately.
        * per_band=False -> use global min/max over all bands.
    """
    arr = to_float32(arr)

    # If shape is (1, H, W) -> squeeze to (H, W)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    # Single band (H, W)
    if arr.ndim == 2:
        amin = arr.min()
        amax = arr.max()
        if amax > amin:
            return (arr - amin) / (amax - amin)
        else:
            # Constant image
            return np.zeros_like(arr, dtype="float32")

    # Multi-band (B, H, W)
    if not per_band:
        amin = arr.min()
        amax = arr.max()
        if amax > amin:
            return (arr - amin) / (amax - amin)
        out = np.zeros_like(arr, dtype="float32")
        return out

    # Per-band normalization
    out = np.empty_like(arr, dtype="float32")
    for i in range(arr.shape[0]):
        band = arr[i]
        amin = band.min()
        amax = band.max()
        if amax > amin:
            out[i] = (band - amin) / (amax - amin)
        else:
            out[i] = 0.0
    return out


def stack_bands(band_list):
    """
    Stack a list of 2D arrays (H, W) into a 3D cube (B, H, W).

    All bands must have the same height and width.
    """
    bands = [np.asarray(b) for b in band_list]
    return np.stack(bands, axis=0)


def make_valid_mask(arr, nodata_value=0):
    """
    Build a boolean mask of valid pixels.

    - For (H, W): True where pixel is not nodata (and not NaN).
    - For (1, H, W): same as single band.
    - For (B, H, W): True where ALL bands are valid.
      (useful when stacking multiple bands).
    """
    arr = np.asarray(arr)

    # Single band (H, W)
    if arr.ndim == 2:
        mask = np.ones_like(arr, dtype=bool)
        if np.issubdtype(arr.dtype, np.floating):
            mask &= ~np.isnan(arr)
        mask &= arr != nodata_value
        return mask

    # Shape (1, H, W) -> treat as single band
    if arr.ndim == 3 and arr.shape[0] == 1:
        band = arr[0]
        mask = np.ones_like(band, dtype=bool)
        if np.issubdtype(band.dtype, np.floating):
            mask &= ~np.isnan(band)
        mask &= band != nodata_value
        return mask

    # Multi-band (B, H, W)
    if np.issubdtype(arr.dtype, np.floating):
        valid = ~np.isnan(arr)
    else:
        valid = np.ones_like(arr, dtype=bool)

    valid &= arr != nodata_value

    # Collapse along band axis: pixel is valid if all bands are valid
    mask = np.all(valid, axis=0)   # (H, W)
    return mask


def clip_by_percentile(arr, low=1.0, high=99.0, per_band=True):
    """
    Clip extreme values using percentiles.

    - For (H, W): clip between low/high percentiles.
    - For (B, H, W):
        * per_band=True  -> clip each band separately.
        * per_band=False -> use global percentiles over all bands.
    """
    arr = np.asarray(arr)

    # If shape is (1, H, W) -> squeeze to (H, W)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr2d = arr[0]
        lo = np.percentile(arr2d, low)
        hi = np.percentile(arr2d, high)
        return np.clip(arr2d, lo, hi)

    # Single band
    if arr.ndim == 2:
        lo = np.percentile(arr, low)
        hi = np.percentile(arr, high)
        return np.clip(arr, lo, hi)

    # Multi-band
    if not per_band:
        lo = np.percentile(arr, low)
        hi = np.percentile(arr, high)
        return np.clip(arr, lo, hi)

    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        band = arr[i]
        lo = np.percentile(band, low)
        hi = np.percentile(band, high)
        out[i] = np.clip(band, lo, hi)
    return out
