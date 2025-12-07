import geopandas as gpd
import rasterio

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


def save_raster(path, meta, array):
    """Save array as GeoTIFF."""
    meta = meta.copy()
    
    # array format: single band or multi-band
    if array.ndim == 2:
        meta["count"] = 1
    else:
        meta["count"] = array.shape[0]

    with rasterio.open(path, "w", **meta) as dst:
        if array.ndim == 2:
            dst.write(array, 1)
        else:
            dst.write(array)
