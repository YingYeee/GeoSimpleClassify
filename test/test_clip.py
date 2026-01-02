import unittest
import os
from geosimpleclassify.core.clip import clip_raster_with_vector

class TestClip(unittest.TestCase):
    def setUp(self):
        # Paths to sample data
        self.raster_path = "test/data/raw/Raster/Milano_sample/T32TMR_R10m_sample/T32TMR_20251103T102221_B02_10m_sample.tif"
        self.vector_path = "test/data/raw/Vector/ProvCM01012025_WGS84.shp"
        self.output_path = "test_clipped.tif"

    def test_clip_raster_with_filter(self):
        """Test clipping a raster using a vector boundary with an attribute filter."""
        meta, arr = clip_raster_with_vector(
            self.raster_path, 
            self.vector_path, 
            out_path=self.output_path,
            attr_name="DEN_CM", 
            attr_value="Milano"
        )
        
        self.assertTrue(os.path.exists(self.output_path))
        # Verify that metadata height/width were correctly updated after clipping
        self.assertEqual(meta['height'], arr.shape[1])
        self.assertEqual(meta['width'], arr.shape[2])
        
        # Clean up
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

if __name__ == "__main__":
    unittest.main()