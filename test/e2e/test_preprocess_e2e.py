import unittest
import os
import shutil
from pathlib import Path
import numpy as np

from geosimpleclassify.config.schema import default_cfg
from geosimpleclassify.pipelines.run_preprocess import run_preprocess


class TestPreprocessE2E(unittest.TestCase):
    def setUp(self):
        # test data paths 
        self.raster_dir = Path("test/data/raw/Raster/Milano_sample")
        self.vector_path = Path("test/data/raw/Vector/ProvCM01012025_WGS84.shp")

        # fixed output dirs
        self.out_root = Path("test/_tmp_preprocess_e2e")
        self.out_derived = self.out_root / "derived"
        self.out_final = self.out_root / "final"

        self.out_derived.mkdir(parents=True, exist_ok=True)
        self.out_final.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # remove generated files
        shutil.rmtree(self.out_root, ignore_errors=True)


    def test_run_preprocess_e2e_contract(self):
        # Skip gracefully if test data is not present
        if not self.raster_dir.exists():
            self.skipTest(f"Missing raster test data dir: {self.raster_dir}")
        if not self.vector_path.exists():
            self.skipTest(f"Missing vector test data: {self.vector_path}")

        # Build cfg
        cfg = default_cfg()
        cfg.paths.preprocess_input_raster_dir = self.raster_dir
        cfg.paths.preprocess_input_vector_path = self.vector_path
        cfg.paths.preprocess_derived_dir = self.out_derived
        cfg.paths.preprocess_final_dir = self.out_final
        if cfg.preprocess.band_codes is None:
            cfg.preprocess.band_codes = ["B02", "B03", "B04", "B08"]
        if getattr(cfg.preprocess, "clip_attr_name", None) is None:
            cfg.preprocess.clip_attr_name = "DEN_CM"
        if getattr(cfg.preprocess, "clip_attr_value", None) is None:
            cfg.preprocess.clip_attr_value = "Milano"   

        # 1) Run end-to-end
        cube_norm, mask, meta_ref = run_preprocess(cfg)

        # 2) Contract checks
        self.assertIsNotNone(meta_ref)
        self.assertIsInstance(cube_norm, np.ndarray)
        self.assertIsInstance(mask, np.ndarray)

        self.assertEqual(cube_norm.ndim, 3)  # (B, H, W)
        self.assertEqual(mask.ndim, 2)  # (H, W)

        b, h, w = cube_norm.shape
        self.assertEqual(b, len(cfg.preprocess.band_codes))
        self.assertEqual(mask.shape, (h, w))
        self.assertGreater(mask.sum(), 0)

        # 3) Output existence checks
        # derived outputs（mosaics/clips）should have at least one tif
        self.assertTrue(os.path.exists(self.out_derived))
        self.assertTrue(len(list(self.out_derived.glob("*.tif"))) > 0)

        # final outputs (norm + mask) should have at least one tif 
        self.assertTrue(os.path.exists(self.out_final))
        self.assertTrue(len(list(self.out_final.glob("*.tif"))) > 0)

if __name__ == "__main__":
    unittest.main()
