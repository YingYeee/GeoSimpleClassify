import unittest
import numpy as np

from geosimpleclassify.core.feature import extract_pixel_features


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        # small synthetic cube: (B,H,W)
        self.cube = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        self.mask = np.zeros((4, 5), dtype=bool)
        self.mask[1:3, 2:5] = True  # 2 rows x 3 cols = 6 pixels valid

    def test_extract_without_mask(self):
        features, idx, (h, w) = extract_pixel_features(self.cube, mask=None)
        self.assertEqual((h, w), (4, 5))
        self.assertEqual(features.shape, (4 * 5, 3))
        self.assertEqual(idx.shape[0], 4 * 5)
        # first pixel features should match cube[:,0,0]
        np.testing.assert_allclose(features[0], self.cube[:, 0, 0])

    def test_extract_with_mask(self):
        features, idx, (h, w) = extract_pixel_features(self.cube, mask=self.mask)
        self.assertEqual((h, w), (4, 5))
        self.assertEqual(features.shape, (6, 3))
        self.assertEqual(idx.shape[0], 6)

        # Verify that idx corresponds to mask True positions
        mask_flat = self.mask.reshape(-1)
        idx_expected = np.where(mask_flat)[0]
        np.testing.assert_array_equal(idx, idx_expected)

        # Verify one known position
        # Mask includes (row=1,col=2) => linear index 1*5+2 = 7
        pos = np.where(idx == 7)[0][0]
        np.testing.assert_allclose(features[pos], self.cube[:, 1, 2])

    def test_mask_shape_mismatch_raises(self):
        bad_mask = np.ones((4, 4), dtype=bool)
        with self.assertRaises(ValueError):
            extract_pixel_features(self.cube, mask=bad_mask)

    def test_cube_dim_mismatch_raises(self):
        bad_cube = np.zeros((4, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            extract_pixel_features(bad_cube, mask=None)


if __name__ == "__main__":
    unittest.main()
