import unittest
import os
import numpy as np
from geosimpleclassify.core.geo_io import load_raster, save_raster, load_vector, load_raster_roi

class TestGeoIO(unittest.TestCase):
    def setUp(self):
        # Paths to sample data
        self.raster_path = "test/data/raw/Raster/Milano_sample/T32TMR_R10m_sample/T32TMR_20251103T102221_B02_10m_sample.tif"
        self.vector_path = "test/data/raw/Vector/ProvCM01012025_WGS84.shp"
        self.output_path = "test_output_io.tif"

    def test_load_vector(self):
        """Test if vector data is loaded as a GeoDataFrame."""
        gdf = load_vector(self.vector_path)
        self.assertFalse(gdf.empty)
        # Ensure CRS exists before calling to_epsg()
        self.assertIsNotNone(gdf.crs, "The vector file must have a CRS")
        # Check if CRS is correctly identified (EPSG:32632 for WGS 84 / UTM zone 32N)
        epsg_code = gdf.crs.to_epsg()   # type: ignore
        expected_epsg = 32632
        self.assertEqual(epsg_code, expected_epsg)

    def test_load_raster(self):
        """Test if raster metadata and array are loaded correctly."""
        meta, arr = load_raster(self.raster_path)
        self.assertIn('driver', meta)
        self.assertEqual(arr.ndim, 3) # 3 dimensions: (bands, H, W)
        self.assertTrue(arr.size > 0)


    def test_save_raster(self):
        """Test saving a raster array to a file."""
        meta, arr = load_raster(self.raster_path)
        save_raster(self.output_path, meta, arr)
        self.assertTrue(os.path.exists(self.output_path))
        
        # Clean up
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def test_load_raster_roi(self):
        """Test loading a specific Region of Interest (ROI)."""
        # ROI format: (row_start, row_end, col_start, col_end)
        roi = (0, 10, 0, 10)
        meta, arr = load_raster_roi(self.raster_path, roi)
        self.assertEqual(arr.shape[1], 10)
        self.assertEqual(arr.shape[2], 10)
        self.assertEqual(meta['height'], 10)
        self.assertEqual(meta['width'], 10)

if __name__ == "__main__":
    unittest.main()