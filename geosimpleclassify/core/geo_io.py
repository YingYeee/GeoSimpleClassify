import geopandas as gpd
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from typing import Tuple
from rasterio.windows import Window


ROI = Tuple[int, int, int, int]  # (row_start, row_end, col_start, col_end)


def load_vector(path):
    """Load vector data."""
    return gpd.read_file(path)


def load_raster(path, bands=None):
    """Load raster and return metadata + array."""
    with rasterio.open(path) as src:
        meta = src.meta.copy()

        if bands is None:
            arr = src.read()          # all bands
        elif isinstance(bands, int):
            arr = src.read(bands)     # one band → (H, W)
        else:
            arr = src.read(bands)     # list of bands → (B, H, W)

    return meta, arr


def save_raster(path, meta, array, dtype=None, nodata=None, compress="deflate"):
    """
    Save array as GeoTIFF.

    dtype/nodata are optional but recommended.
    """
    meta = meta.copy()

    # set band count
    if array.ndim == 2:
        meta["count"] = 1
    else:
        meta["count"] = array.shape[0]

    # set dtype
    if dtype is None:
        dtype = array.dtype
    meta["dtype"] = str(np.dtype(dtype))

    # set nodata
    if nodata is not None:
        meta["nodata"] = nodata

    # optional compression (keeps files smaller)
    if compress is not None:
        meta["compress"] = compress

    with rasterio.open(path, "w", **meta) as dst:
        if array.ndim == 2:
            dst.write(array.astype(dtype, copy=False), 1)
        else:
            dst.write(array.astype(dtype, copy=False))


# -----------------------------------------------------------------------------
# Added by Tian
# -----------------------------------------------------------------------------


def align_raster_to_meta(
    src_path,
    ref_meta,
    band=1,
    resampling="nearest",
):
    """
    Reproject and resample a raster band to exactly match a reference grid.

    Parameters
    ----------
    src_path : str
        Path to source raster.
    ref_meta : dict
        Raster metadata defining target grid (transform, crs, height, width).
    band : int, optional
        Band index to read, by default 1.
    resampling : str, optional
        Resampling method ("nearest", "bilinear", "cubic"), by default "nearest".

    Returns
    -------
    ndarray
        Aligned raster array with shape (H, W).
    """
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    rs = resampling_map.get(resampling, Resampling.nearest)

    dst_h, dst_w = ref_meta["height"], ref_meta["width"]
    dst = np.zeros((dst_h, dst_w), dtype="float32")

    with rasterio.open(src_path) as src:
        src_arr = src.read(band)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            resampling=rs,
        )
    return dst


def load_mask_aligned(
    mask_path,
    ref_meta,
    threshold=0,
):
    """
    Load a mask raster and align it to a reference grid.

    Parameters
    ----------
    mask_path : str
        Path to mask raster.
    ref_meta : dict
        Target grid metadata.
    threshold : float, optional
        Values greater than this are considered valid, by default 0.

    Returns
    -------
    ndarray of bool
        Boolean mask aligned to reference grid.
    """
    aligned = align_raster_to_meta(mask_path, ref_meta, band=1, resampling="nearest")
    return aligned > threshold


def _roi_to_window(roi: ROI):
    """
    Convert a ROI tuple to a rasterio Window.

    Parameters
    ----------
    roi : tuple
        (row_start, row_end, col_start, col_end)

    Returns
    -------
    rasterio.windows.Window
        Window corresponding to the ROI.
    """
    r0, r1, c0, c1 = roi
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"Invalid ROI: {roi}")
    return Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)


def load_raster_roi(
    path,
    roi: ROI,
    bands=None,
):
    """
    Load a raster subset defined by a ROI.

    Parameters
    ----------
    path : str
        Path to raster.
    roi : tuple
        (row_start, row_end, col_start, col_end).
    bands : int or list, optional
        Bands to read. "None" reads all bands.

    Returns
    -------
    meta : dict
        Updated metadata for the ROI.
    arr : ndarray
        Raster data for the ROI.
    """
    with rasterio.open(path) as src:
        win = _roi_to_window(roi)

        # Update meta of ROI
        meta = src.meta.copy()
        meta.update(
            height=int(win.height),
            width=int(win.width),
            transform=rasterio.windows.transform(win, src.transform),
        )

        if bands is None:
            arr = src.read(window=win)
        else:
            arr = src.read(bands, window=win)

    return meta, arr