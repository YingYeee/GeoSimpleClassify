import unittest
import numpy as np

from geosimpleclassify.core.unsupervised import unsupervised_cluster


class TestUnsupervised(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # two obvious clusters in 2D
        a = rng.normal(loc=-5.0, scale=0.3, size=(200, 2))
        b = rng.normal(loc=+5.0, scale=0.3, size=(200, 2))
        self.features = np.vstack([a, b]).astype(np.float32)

    def test_kmeans_basic(self):
        labels = unsupervised_cluster(
            self.features,
            method="kmeans",
            n_clusters=2,
            random_state=0,
            fast_mode=False,
        )
        self.assertEqual(labels.shape, (self.features.shape[0],))
        # should produce exactly 2 classes (label ids may be 0/1)
        self.assertEqual(np.unique(labels).size, 2)

    def test_fast_mode_chunked_predict(self):
        labels = unsupervised_cluster(
            self.features,
            method="kmeans",
            n_clusters=2,
            random_state=0,
            fast_mode=True,
            sample_size=50,
            batch_size=32,
            max_iter=10,
            predict_chunk_size=64,
        )
        self.assertEqual(labels.shape, (self.features.shape[0],))
        self.assertEqual(np.unique(labels).size, 2)

    def test_empty_features(self):
        empty = np.empty((0, 2), dtype=np.float32)
        labels = unsupervised_cluster(empty, n_clusters=2)
        self.assertEqual(labels.shape, (0,))
        self.assertEqual(labels.dtype, np.int32)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            unsupervised_cluster(self.features, method="gmm")

    def test_invalid_features_dim_raises(self):
        with self.assertRaises(ValueError):
            unsupervised_cluster(np.zeros((10,), dtype=np.float32), method="kmeans")


if __name__ == "__main__":
    unittest.main()
