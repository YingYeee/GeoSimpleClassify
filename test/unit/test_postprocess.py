import unittest
import os
import numpy as np

from geosimpleclassify.core.postprocess import (
    reshape_labels_to_raster,
    visualize,
    save_label_summary,
    plot_label_histogram,
)


class _DummyTransform:
    # Minimal affine-like object for save_label_summary: needs .a and .e
    def __init__(self, a=10.0, e=-10.0):
        self.a = a
        self.e = e


class TestPostprocess(unittest.TestCase):
    def setUp(self):
        self.h, self.w = 4, 5
        self.mask = np.zeros((self.h, self.w), dtype=bool)
        self.mask[1:3, 2:5] = True  # 6 valid pixels
        self.labels = np.array([0, 1, 1, 2, 2, 2], dtype=np.int32)

        self.out_csv = "test_labels_summary.csv"
        self.out_png = "test_labels_hist.png"
        self.out_vis = "test_vis.png"

    def tearDown(self):
        for p in [self.out_csv, self.out_png, self.out_vis]:
            if os.path.exists(p):
                os.remove(p)

    def test_reshape_labels_to_raster(self):
        label_map = reshape_labels_to_raster(
            self.labels,
            self.h,
            self.w,
            self.mask,
            nodata_label=0,
            label_offset=1,
        )
        self.assertEqual(label_map.shape, (self.h, self.w))
        # nodata should be 0 outside mask
        self.assertTrue(np.all(label_map[~self.mask] == 0))
        # inside mask should be labels + offset
        self.assertTrue(np.all(label_map[self.mask] == (self.labels + 1)))

    def test_visualize_label_returns_rgb(self):
        # Create a small label map including nodata
        label_map = np.zeros((self.h, self.w), dtype=np.int32)
        label_map[self.mask] = (self.labels + 1)
        rgb = visualize(
            label_map,
            kind="label",
            nodata=0,
            seed=0,
            show=False,
            save_path=self.out_vis,
            title="test",
        )
        self.assertEqual(rgb.shape, (self.h, self.w, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue(os.path.exists(self.out_vis))

    def test_save_summary_and_histogram(self):
        label_map = np.zeros((self.h, self.w), dtype=np.int32)
        label_map[self.mask] = (self.labels + 1)
        transform = _DummyTransform(a=10.0, e=-10.0)  # 100 m^2 per pixel

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
            title="test",
        )
        self.assertTrue(os.path.exists(self.out_png))


if __name__ == "__main__":
    unittest.main()
