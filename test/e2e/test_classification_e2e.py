import unittest
import os
import numpy as np

from geosimpleclassify.core.feature import extract_pixel_features
from geosimpleclassify.core.unsupervised import unsupervised_cluster
from geosimpleclassify.core.supervised import supervised_classify
from geosimpleclassify.core.postprocess import (
    reshape_labels_to_raster,
    visualize,
    save_label_summary,
    plot_label_histogram,
)


class _DummyTransform:
    def __init__(self, a=10.0, e=-10.0):
        self.a = a
        self.e = e


class TestEndToEndStage2(unittest.TestCase):
    def setUp(self):
        self.h, self.w = 64, 64
        rng = np.random.default_rng(123)

        # Synthetic 4-band cube (B,H,W)
        self.cube = rng.normal(size=(4, self.h, self.w)).astype(np.float32)

        # Valid mask: center area
        self.mask = np.zeros((self.h, self.w), dtype=bool)
        self.mask[16:48, 16:48] = True  # 32x32=1024 valid pixels

        self.out_csv = "test_e2e_summary.csv"
        self.out_png = "test_e2e_hist.png"
        self.out_vis = "test_e2e_labels.png"

    def tearDown(self):
        for p in [self.out_csv, self.out_png, self.out_vis]:
            if os.path.exists(p):
                os.remove(p)

    def test_stage2_pipeline_minimal(self):
        # 1) features
        feats, idx, (h, w) = extract_pixel_features(self.cube, self.mask)
        self.assertEqual((h, w), (self.h, self.w))
        self.assertEqual(feats.shape[1], 4)
        self.assertEqual(feats.shape[0], self.mask.sum())

        # 2) unsupervised (fast mode to emulate real pipeline intent)
        labels_init = unsupervised_cluster(
            feats,
            method="kmeans",
            n_clusters=4,
            random_state=0,
            fast_mode=True,
            sample_size=256,
            batch_size=64,
            max_iter=20,
            predict_chunk_size=200,
        )
        self.assertEqual(labels_init.shape, (feats.shape[0],))
        self.assertEqual(np.unique(labels_init).size, 4)

        # 3) supervised refinement
        labels_final = supervised_classify(
            feats,
            labels_init,
            model="rf",
            sample_per_class=50,
            random_state=0,
        )
        self.assertEqual(labels_final.shape, labels_init.shape)

        # 4) reshape back to raster
        label_map = reshape_labels_to_raster(
            labels_final,
            self.h,
            self.w,
            self.mask,
            nodata_label=0,
            label_offset=1,
        )
        self.assertEqual(label_map.shape, (self.h, self.w))
        self.assertTrue(np.all(label_map[~self.mask] == 0))

        # 5) visualize (no display, save to png)
        rgb = visualize(
            label_map,
            kind="label",
            nodata=0,
            seed=0,
            show=False,
            save_path=self.out_vis,
            title="e2e",
        )
        self.assertEqual(rgb.shape, (self.h, self.w, 3))
        self.assertTrue(os.path.exists(self.out_vis))

        # 6) stats CSV + histogram
        transform = _DummyTransform(a=10.0, e=-10.0)
        save_label_summary(
            label_map=label_map,
            mask=self.mask,
            transform=transform,
            out_csv_path=self.out_csv,
            nodata=0,
        )
        self.assertTrue(os.path.exists(self.out_csv))

        plot_label_histogram(
            summary_csv_path=self.out_csv,
            out_png_path=self.out_png,
            value_col="percent",
            title="e2e",
        )
        self.assertTrue(os.path.exists(self.out_png))


if __name__ == "__main__":
    unittest.main()
