import rasterio
from rasterio.merge import merge

def merge_rasters(raster_paths, out_path=None):
    """
    Merge multiple rasters into a single mosaic.
    All rasters must have the same CRS and resolution.
    """

    src_files = [rasterio.open(p) for p in raster_paths]
    ref = src_files[0]

    for i, src in enumerate(src_files[1:], start=1):

        # Check CRS
        if src.crs != ref.crs:
            raise ValueError(
                f"CRS mismatch between raster[0] and raster[{i}]: "
                f"{ref.crs} != {src.crs}"
            )

        # Check resolution (pixel size)
        ref_res = (abs(ref.transform.a), abs(ref.transform.e))
        src_res = (abs(src.transform.a), abs(src.transform.e))

        if src_res != ref_res:
            raise ValueError(
                f"Resolution mismatch between raster[0] and raster[{i}]: "
                f"{ref_res} != {src_res}"
            )

    # mosaic: arr shape = (bands, H, W), transform = new affine
    mosaic_arr, mosaic_transform = merge(src_files)

    # use metadata from the first raster as template
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "height": mosaic_arr.shape[1],
        "width": mosaic_arr.shape[2],
        "transform": mosaic_transform,
        "count": mosaic_arr.shape[0],
    })

    # close all opened files
    for src in src_files:
        src.close()

    # optionally save to disk
    if out_path is not None:
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(mosaic_arr)

    return out_meta, mosaic_arr
