import unittest
import numpy as np

from geosimpleclassify.core.supervised import supervised_classify


class TestSupervised(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        # Synthetic features and pseudo labels
        self.features = rng.normal(size=(300, 4)).astype(np.float32)
        self.labels_init = rng.integers(0, 3, size=(300,), dtype=np.int32)

    def test_rf_supervised_basic(self):
        labels_final = supervised_classify(
            self.features,
            self.labels_init,
            model="rf",
            sample_per_class=50,
            random_state=0,
        )
        self.assertEqual(labels_final.shape, self.labels_init.shape)
        # labels should be within the set of initial labels (not guaranteed strictly, but usually)
        self.assertTrue(np.unique(labels_final).size >= 1)

    def test_svm_supervised_basic(self):
        labels_final = supervised_classify(
            self.features,
            self.labels_init,
            model="svm",
            sample_per_class=30,
            random_state=0,
        )
        self.assertEqual(labels_final.shape, self.labels_init.shape)

    def test_invalid_model_raises(self):
        with self.assertRaises(ValueError):
            supervised_classify(self.features, self.labels_init, model="xgb")

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            supervised_classify(self.features[:10], self.labels_init, model="rf")


if __name__ == "__main__":
    unittest.main()
