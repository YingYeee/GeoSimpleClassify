import unittest
import numpy as np
from geosimpleclassify.core.preprocess import normalize, stack_bands, make_valid_mask, clip_by_percentile

class TestPreprocess(unittest.TestCase):
    def test_normalize_single_band(self):
        """Test normalization logic on a 2D array."""
        # Define a dummy 2D array for testing the normalization formula
        data = np.array([[10, 20], [30, 40]], dtype="float32")
        result = normalize(data)
        self.assertEqual(result.min(), 0.0) # Minimum should map to 0
        self.assertEqual(result.max(), 1.0) # Maximum should map to 1
        self.assertEqual(result[0, 1], (20-10)/(40-10)) # Check 20 maps correctly

    def test_stack_bands(self):
        """Test stacking multiple 2D bands into a 3D array."""
        b1 = np.ones((5, 5))
        b2 = np.zeros((5, 5))
        stacked = stack_bands([b1, b2])
        self.assertEqual(stacked.shape, (2, 5, 5)) # Should have 2 bands
        self.assertTrue(np.all(stacked[0] == 1)) # First band all ones
        self.assertTrue(np.all(stacked[1] == 0)) # Second band all zeros

    def test_make_valid_mask(self):
        """Test creation of a boolean mask for valid pixels (not nodata or NaN)."""
        data = np.array([[1.0, np.nan], [0.0, 2.0]])
        # Assuming 0.0 is the nodata value
        mask = make_valid_mask(data, nodata_value=0.0) # type: ignore
        expected = np.array([[True, False], [False, True]]) # Only 1.0 and 2.0 are valid
        np.testing.assert_array_equal(mask, expected)

    def test_clip_by_percentile_scenarios(self):
        """Test all clipping scenarios using subtests to reduce redundancy."""
        
        # Define test scenarios: (name, input_data, params, expected_values)
        scenarios = [
            (
                "2D_Single_Band", 
                np.arange(100, dtype="float32").reshape(10, 10),
                {"low": 1.0, "high": 99.0},
                {"min": 0.99, "max": 98.01, "ndim": 2} # Expected min/max after clipping
            ),
            (
                "3D_Per_Band_True",
                np.stack([np.arange(10, dtype="float32").reshape(2, 5), 
                          np.arange(100, 110, dtype="float32").reshape(2, 5)]),
                {"low": 10.0, "high": 90.0, "per_band": True},
                # Band 0 [0..9] (N=10): P10 at index 0.9 is 0.9; P90 at index 8.1 is 8.1
                # Band 1 [100..109] (N=10): P10 at index 0.9 is 100.9; P90 at index 8.1 is 108.1
                {"min": 0.9, "max": 108.1, "ndim": 3} 
            ),
            (
                "3D_Per_Band_False",
                np.stack([np.arange(10, dtype="float32").reshape(2, 5), 
                          np.arange(100, 110, dtype="float32").reshape(2, 5)]),
                {"low": 10.0, "high": 90.0, "per_band": False},
                # Combined data [0..9, 100..109] (N=20)
                # P10 index = (20-1) * 0.1 = 1.9. Value: 1 + (2-1)*0.9 = 1.9
                # P90 index = (20-1) * 0.9 = 17.1. Value: 107 + (108-107)*0.1 = 107.1
                {"min": 1.9, "max": 107.1, "ndim": 3}
            ),
            (
                "3D_Single_Slice_Squeeze",
                np.arange(100, dtype="float32").reshape(1, 10, 10),
                {"low": 1.0, "high": 99.0},
                {"min": 0.99, "max": 98.01, "ndim": 2} # Should be squeezed to 2D
            )
        ]

        for name, data, params, expected in scenarios:
            with self.subTest(msg=name):
                # Perform clipping
                result = clip_by_percentile(data, **params)
                # Assertions
                self.assertEqual(result.ndim, expected["ndim"])
                self.assertAlmostEqual(float(result.min()), expected["min"], places=2)
                self.assertAlmostEqual(float(result.max()), expected["max"], places=2)

if __name__ == "__main__":
    unittest.main()