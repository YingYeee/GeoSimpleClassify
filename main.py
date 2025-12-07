# build 4-band cube (B02, B03, B04, B08) and preprocess.
from pathlib import Path

from geosimpleclassify.geo_io import save_raster
from geosimpleclassify.mosaic import merge_rasters
from geosimpleclassify.clip import clip_raster_with_vector
from geosimpleclassify.preprocess import (
    stack_bands,
    to_float32,
    clip_by_percentile,
    normalize,
    make_valid_mask,
)



# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RASTER_DIR = DATA_DIR / "Raster" / "R10m_geotiff" / "Milano"
VECTOR_PATH = DATA_DIR / "Vector" / "ProvCM01012025_WGS84.shp"

# tile folders
T32TMR_DIR = RASTER_DIR / "T32TMR_R10m"
T32TNR_DIR = RASTER_DIR / "T32TNR_R10m"

OUTPUT_DIR = RASTER_DIR / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Helper: process one band (merge tiles + clip by Milano)
# -----------------------------------------------------------------------------

def process_band(band_code: str):
    """
    Build mosaic and clipped raster for one band.

    band_code: e.g. "B02", "B03", "B04", "B08".
    Returns (meta_clip, arr_clip).
    """

    # file names for this date – adjust if your filenames differ
    fname_tmr = f"T32TMR_20251103T102221_{band_code}_10m.tif"
    fname_tnr = f"T32TNR_20251103T102221_{band_code}_10m.tif"

    raster_paths = [
        T32TMR_DIR / fname_tmr,
        T32TNR_DIR / fname_tnr,
    ]

    # 1) merge tiles (mosaic)
    mosaic_out = OUTPUT_DIR / f"Milano_{band_code}_mosaic.tif"
    meta_mosaic, arr_mosaic = merge_rasters(
        [str(p) for p in raster_paths],
        out_path=str(mosaic_out),
    )
    print(f"[{band_code}] mosaic shape:", arr_mosaic.shape)

    # 2) clip mosaic with Milano boundary (DEN_CM == 'Milano')
    clip_out = OUTPUT_DIR / f"Milano_{band_code}_clip.tif"
    meta_clip, arr_clip = clip_raster_with_vector(
        raster_path=str(mosaic_out),
        vector_path=str(VECTOR_PATH),
        out_path=str(clip_out),
        attr_name="DEN_CM",
        attr_value="Milano",
    )
    print(f"[{band_code}] clip shape:", arr_clip.shape)

    # Ensure 2D (H, W)
    if arr_clip.ndim == 3 and arr_clip.shape[0] == 1:
        arr_clip = arr_clip[0]

    return meta_clip, arr_clip


# -----------------------------------------------------------------------------
# Main pipeline for Milano 4-band cube
# -----------------------------------------------------------------------------


def build_milano_cube():
    # Bands to use
    band_codes = ["B02", "B03", "B04", "B08"]

    clipped_bands = []
    meta_ref = None

    # process each band separately
    for b in band_codes:
        meta_clip, arr_clip = process_band(b)
        if meta_ref is None:
            meta_ref = meta_clip
        clipped_bands.append(arr_clip)

    # 3) stack bands -> (4, H, W)
    cube = stack_bands(clipped_bands)
    print("Cube shape before preprocess:", cube.shape)

    # 4) preprocess: float32, percentile clipping, normalization
    cube = to_float32(cube)
    cube = clip_by_percentile(cube, low=1.0, high=99.0, per_band=True)
    cube_norm = normalize(cube, per_band=True)
    mask = make_valid_mask(cube, nodata_value=0)

    print("Cube_norm shape:", cube_norm.shape)
    print("Mask shape:", mask.shape)
    print("Valid pixels %:", mask.sum() / mask.size)

    # 5) save normalized cube as 4-band GeoTIFF
    cube_out = OUTPUT_DIR / "Milano_B02_B03_B04_B08_norm.tif"
    save_raster(str(cube_out), meta_ref, cube_norm)
    print("Saved normalized cube to:", cube_out)

    # (optional) save mask as 0/1 raster
    # mask_out = OUTPUT_DIR / "Milano_mask.tif"
    # save_raster(str(mask_out), meta_ref, mask.astype("uint8"))

    return cube_norm, mask, meta_ref


if __name__ == "__main__":
    build_milano_cube()
