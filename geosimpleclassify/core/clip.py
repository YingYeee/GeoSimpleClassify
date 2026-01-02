import geopandas as gpd
import rasterio
from rasterio.mask import mask


def clip_raster_with_vector(
    raster_path,
    vector_path,
    out_path=None,
    attr_name=None,
    attr_value=None,
):
    """
    Clip a raster using a vector boundary.
    Optionally filter the vector layer by an attribute.
    Returns clipped metadata and array.
    """

    # Load full vector layer (all features)
    gdf = gpd.read_file(vector_path)

    # Optional attribute filter, e.g. attr_name="DEN_CM", attr_value="Milano"
    if attr_name is not None and attr_value is not None:
        gdf = gdf[gdf[attr_name] == attr_value]

    if gdf.empty:
        raise ValueError("No features found after filtering. Check attr_name and attr_value.")

    with rasterio.open(raster_path) as src:

        # Reproject vector to raster CRS if needed
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # Convert geometries to the format required by rasterio.mask
        shapes = [geom.__geo_interface__ for geom in gdf.geometry]

        # Perform clip / mask
        out_image, out_transform = mask(src, shapes, crop=True)

        # Update metadata for the clipped raster
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "count": out_image.shape[0]
        })

    # Optionally save the clipped raster to disk
    if out_path is not None:
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_image)
            print("Clipped raster saved to", out_path)

    return out_meta, out_image
