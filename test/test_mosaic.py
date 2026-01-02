import unittest
import os
from geosimpleclassify.core.mosaic import merge_rasters

class TestMosaic(unittest.TestCase):
    def setUp(self):
        # Paths to sample data (two tiles)
        self.raster_list = [
            "test/data/raw/Raster/Milano_sample/T32TMR_R10m_sample/T32TMR_20251103T102221_B02_10m_sample.tif",
            "test/data/raw/Raster/Milano_sample/T32TNR_R10m_sample/T32TNR_20251103T102221_B02_10m_sample.tif"
        ]
        self.output_path = "test_mosaic_result.tif"

    def test_merge_rasters(self):
        """Test merging multiple rasters into a single mosaic."""
        meta, arr = merge_rasters(self.raster_list, out_path=self.output_path)
        
        self.assertTrue(os.path.exists(self.output_path))
        self.assertEqual(arr.ndim, 3)
        # The result should have the same number of bands as the input
        self.assertEqual(meta['count'], arr.shape[0])
        
        # Clean up
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

if __name__ == "__main__":
    unittest.main()