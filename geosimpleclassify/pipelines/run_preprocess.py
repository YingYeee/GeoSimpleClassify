# build 4-band cube (B02, B03, B04, B08) and preprocess.

from geosimpleclassify.core.geo_io import save_raster
from geosimpleclassify.core.mosaic import merge_rasters
from geosimpleclassify.core.clip import clip_raster_with_vector
from geosimpleclassify.core.preprocess import (
    stack_bands,
    to_float32,
    clip_by_percentile,
    normalize,
    make_valid_mask,
)
from geosimpleclassify.config.schema import default_cfg, PipelineCfg


# -----------------------------------------------------------------------------
# Helper: process one band (merge tiles + clip by Milano)
# -----------------------------------------------------------------------------


def find_band_tif(tile_dir, band_code):
    """
    Find the GeoTIFF file corresponding to a given band in a tile directory.

    Parameters
    ----------
    tile_dir : pathlib.Path
        Directory containing raster files for a single tile.
    band_code : str
        Band identifier (e.g. "B02", "B03", "B04", "B08").

    Returns
    -------
    pathlib.Path
        Path to the matching GeoTIFF file.
    """
    matches = sorted(tile_dir.glob(f"*_{band_code}_*.tif"))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 tif for band={band_code} in {tile_dir}, got {len(matches)}: "
            + ", ".join(p.name for p in matches)
        )
    return matches[0]


def process_band(cfg: PipelineCfg, band_code: str):
    """
    Build mosaic and clipped raster for one band.

    band_code: e.g. "B02", "B03", "B04", "B08".
    Returns (meta_clip, arr_clip).
    """
    tile_dirs = sorted([p for p in cfg.paths.preprocess_input_raster_dir.iterdir() if p.is_dir()])

    if not tile_dirs:
        raise FileNotFoundError(f"No tile directories found under: {band_code}")

    raster_paths = [find_band_tif(td, band_code) for td in tile_dirs]

    # 1) merge tiles (mosaic)
    mosaic_out = cfg.paths.preprocess_derived_dir / f"{band_code}_mosaic.tif"
    meta_mosaic, arr_mosaic = merge_rasters(
        [str(p) for p in raster_paths],
        out_path=str(mosaic_out),
    )
    print(f"[{band_code}] mosaic shape:", arr_mosaic.shape)

    # 2) clip mosaic with Milano boundary (DEN_CM == 'Milano')
    clip_out = cfg.paths.preprocess_derived_dir / f"{band_code}_clip.tif"
    meta_clip, arr_clip = clip_raster_with_vector(
        raster_path=str(mosaic_out),
        vector_path=str(cfg.paths.preprocess_input_vector_path),
        out_path=str(clip_out),
        attr_name=cfg.preprocess.clip_attr_name,
        attr_value=cfg.preprocess.clip_attr_value,
    )
    print(f"[{band_code}] clip shape:", arr_clip.shape)

    # Ensure 2D (H, W)
    if arr_clip.ndim == 3 and arr_clip.shape[0] == 1:
        arr_clip = arr_clip[0]

    return meta_clip, arr_clip


# -----------------------------------------------------------------------------
# Main pipeline for Milano 4-band cube
# -----------------------------------------------------------------------------


def run_preprocess(cfg: PipelineCfg):
    clipped_bands = []
    meta_ref = None

    # process each band separately
    for b in cfg.preprocess.band_codes:
        meta_clip, arr_clip = process_band(cfg, b)
        if meta_ref is None:
            meta_ref = meta_clip
        clipped_bands.append(arr_clip)

    # 3) stack bands -> (4, H, W)
    cube = stack_bands(clipped_bands)
    print("Cube shape before preprocess:", cube.shape)

    # 4) preprocess: float32, percentile clipping, normalization
    cube = to_float32(cube)
    cube = clip_by_percentile(cube, low=cfg.preprocess.clip_low, high=cfg.preprocess.clip_high, per_band=cfg.preprocess.per_band_clip)
    cube_norm = normalize(cube, per_band=cfg.preprocess.per_band_normalize)
    mask = make_valid_mask(cube, nodata_value=cfg.preprocess.nodata_value)

    print("Cube_norm shape:", cube_norm.shape)
    print("Mask shape:", mask.shape)
    print("Valid pixels %:", mask.sum() / mask.size)

    # 5) save normalized cube as 4-band GeoTIFF
    bands_tag = "_".join(cfg.preprocess.band_codes)
    cube_out = cfg.paths.preprocess_final_dir / f"{bands_tag}_norm.tif"
    save_raster(str(cube_out), meta_ref, cube_norm, dtype="float32", nodata=cfg.preprocess.nodata_value)
    print("Saved normalized cube to:", cube_out)

    # (optional) save mask as 0/1 raster
    mask_out = cfg.paths.preprocess_final_dir / f"{bands_tag}_mask.tif"
    save_raster(str(mask_out), meta_ref, mask.astype("uint8"), dtype="uint8", nodata=cfg.preprocess.nodata_value)

    return cube_norm, mask, meta_ref


if __name__ == "__main__":
    cfg = default_cfg()
    run_preprocess(cfg)
