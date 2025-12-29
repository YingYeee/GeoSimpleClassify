from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def reshape_labels_to_raster(
    labels,
    height,
    width,
    mask,
    nodata_label = 0,
    label_offset = 1,
):
    """
    Scatter 1D labels back to a 2D raster using a validity mask.

    Parameters
    ----------
    labels : ndarray
        Labels for valid pixels.
    height : int
        Output raster height.
    width : int
        Output raster width.
    mask : ndarray of bool
        Valid-pixel mask with shape (height, width).
    nodata_label : int, optional
        Label value for invalid pixels.
    label_offset : int, optional
        Offset added to labels before writing (e.g., reserve 0 as nodata, 1 as the first class).

    Returns
    -------
    ndarray
        Label map with shape (height, width).
    """
    if mask.shape != (height, width):
        raise ValueError("mask shape mismatch.")

    mask_flat = mask.reshape(-1)
    idx = np.where(mask_flat)[0]

    if labels.shape[0] != idx.shape[0]:
        raise ValueError(f"labels length {labels.shape[0]} != valid pixels {idx.shape[0]}")

    out = np.full((height * width,), nodata_label, dtype=np.int32)
    out[idx] = labels + label_offset
    return out.reshape(height, width)


def colorize_label_map(
    label_map,
    nodata = 0,
    seed = 42,
):
    """
    Map integer labels to random RGB colors for visualization.

    Parameters
    ----------
    label_map : ndarray
        Integer label map with shape (H, W).
    nodata : int, optional
        Label value treated as background.
    seed : int, optional
        RNG seed for color generation.

    Returns
    -------
    ndarray
        RGB image with shape (H, W, 3), dtype = uint8.
    """
    if label_map.ndim != 2:
        raise ValueError("label_map must be (H,W)")

    rgb = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)

    classes = np.unique(label_map)
    classes = classes[classes != nodata]

    rng = np.random.default_rng(seed)
    for c in classes:
        color = rng.integers(40, 256, size=3, dtype=np.uint8)
        rgb[label_map == c] = color

    return rgb


def _stretch_percentile(
    image,
    low = 2.0,
    high = 98.0
):
    """
    Apply per-channel percentile stretch for visualization.

    Parameters
    ----------
    image : ndarray
        RGB float image with shape (H, W, 3).
    low : float, optional
        Lower percentile.
    high : float, optional
        Upper percentile.

    Returns
    -------
    ndarray
        Stretched RGB float image in [0, 1].
    """
    out = np.empty_like(image, dtype=np.float32)
    for k in range(3):
        band = image[..., k]
        lo = np.percentile(band, low)
        hi = np.percentile(band, high)
        if hi <= lo:
            out[..., k] = 0.0
        else:
            out[..., k] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def _cube_to_rgb_u8(
    cube,
    band_order = "B02_B03_B04_B08",
    stretch = None,
    low = 1.0,
    high = 99.5,
    gamma = 1.4,
):
    """
    Convert a multispectral cube to an RGB uint8 image.

    Parameters
    ----------
    cube : ndarray
        Image cube in (B, H, W) or (H, W, B), containing at least B02/B03/B04.
    band_order : str, optional
        Band layout identifier (currently only "B02_B03_B04_B08").
    stretch : bool or None, optional
        If None, auto-detect normalized vs DN-like scale.
    low : float, optional
        Lower percentile for stretching.
    high : float, optional
        Upper percentile for stretching.
    gamma : float or None, optional
        Gamma correction (None disables).

    Returns
    -------
    ndarray
        RGB image with shape (H, W, 3), dtype uint8.
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D, got {cube.shape}")

    if cube.shape[0] in (3, 4):  # (B,H,W)
        cube_bhw = cube
    elif cube.shape[-1] in (3, 4):  # (H,W,B)
        cube_bhw = np.transpose(cube, (2, 0, 1))
    else:
        raise ValueError(f"cube must be (B,H,W) or (H,W,B), got {cube.shape}")

    if band_order != "B02_B03_B04_B08":
        raise ValueError("Only band_order='B02_B03_B04_B08' supported.")

    b02 = cube_bhw[0].astype(np.float32)
    b03 = cube_bhw[1].astype(np.float32)
    b04 = cube_bhw[2].astype(np.float32)

    rgb = np.stack([b04, b03, b02], axis=-1)  # (H,W,3)

    # auto detect scale
    finite = np.isfinite(rgb)
    if not np.any(finite):
        return np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)

    vmin = float(np.min(rgb[finite]))
    vmax = float(np.max(rgb[finite]))

    # If data normalized to [0,1], stretching is usually unnecessary and may fail if range is tiny.
    looks_normalized = (vmin >= -0.05) and (vmax <= 1.05)

    if stretch is None:
        stretch_use = not looks_normalized  # stretch only for DN-like data
    else:
        stretch_use = bool(stretch)

    if stretch_use:
        rgb = _stretch_percentile(rgb, low=low, high=high)
    else:
        # For normalized data, just clip to [0,1]
        if looks_normalized:
            rgb = np.clip(rgb, 0.0, 1.0)
        else:
            # If forced no-stretch on DN, still do a simple min-max to avoid black.
            denom = (vmax - vmin) if vmax > vmin else 1.0
            rgb = np.clip((rgb - vmin) / denom, 0.0, 1.0)

    # gamma correction: brighten dark tones (remote-sensing friendly)
    if gamma is not None and gamma > 0:
        rgb = np.clip(rgb, 0.0, 1.0) ** (1.0 / gamma)

    return (rgb * 255.0).round().astype(np.uint8)


def visualize(
    data,
    kind,
    nodata = 0,
    seed = 42,
    title = "",
    band_order = "B02_B03_B04_B08",
    stretch = None,
    low = 2.0,
    high = 98.0,
    show = True,
    save_path = None,
    gamma = 1.4,
):
    """
    Visualize a label map or multispectral cube as an RGB uint8 image.

    Parameters
    ----------
    data : ndarray
        Input data.
    kind : {"label", "cube"}
        Visualization type.
    nodata : int, optional
        Background label for "label" mode.
    seed : int, optional
        RNG seed for "label" mode.
    title : str, optional
        Plot title (if displayed/saved).
    band_order : str, optional
        Band layout identifier for "cube" mode.
    stretch : bool or None, optional
        Stretch control for "cube" mode.
    low : float, optional
        Lower percentile for stretching.
    high : float, optional
        Upper percentile for stretching.
    show : bool, optional
        Show the figure if True.
    save_path : str or None, optional
        Save figure to this path if provided.
    gamma : float or None, optional
        Gamma correction for "cube" mode.

    Returns
    -------
    ndarray
        RGB image with shape (H, W, 3), dtype uint8.
    """
    if kind == "label":
        rgb_u8 = colorize_label_map(data, nodata=nodata, seed=seed)
    elif kind == "cube":
        rgb_u8 = _cube_to_rgb_u8(
            data,
            band_order=band_order,
            stretch=stretch,
            low=low,
            high=high,
            gamma=gamma,
        )
    else:
        raise ValueError("kind must be 'label' or 'cube'")

    if show or save_path is not None:
        plt.figure()
        plt.imshow(rgb_u8)
        if title:
            plt.title(title)
        plt.axis("off")

        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

    return rgb_u8


def compare_and_save(
    left_rgb_u8,
    right_rgb_u8,
    left_title = "Left",
    right_title = "Right",
    out_path = "compare.png",
):
    """
    Save a side-by-side comparison of two RGB images.

    Parameters
    ----------
    left_rgb_u8 : ndarray
        Left RGB image.
    right_rgb_u8 : ndarray
        Right RGB image.
    left_title : str, optional
        Title for left panel.
    right_title : str, optional
        Title for right panel.
    out_path : str, optional
        Output file path.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(left_rgb_u8)
    plt.title(left_title)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(right_rgb_u8)
    plt.title(right_title)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()