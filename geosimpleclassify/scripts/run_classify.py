"""
Classify pipeline: Feature extraction + Unsupervised (KMeans) + Supervised refinement (RF/SVM) + Visualization

Inputs (preprocessed data):
  - B02_B03_B04_B08_norm.tif   (4-band, normalized)
  - mask.tif                   (mask; will be aligned to cube grid)

Outputs:
  - labels_init.tif            (unsupervised labels)
  - labels_final.tif           (supervised refined labels)
"""


from pathlib import Path
import numpy as np

from geosimpleclassify.core.geo_io import load_raster, load_raster_roi, save_raster, load_mask_aligned
from geosimpleclassify.core.feature import extract_pixel_features
from geosimpleclassify.core.unsupervised import unsupervised_cluster
from geosimpleclassify.core.supervised import supervised_classify
from geosimpleclassify.core.postprocess import reshape_labels_to_raster, visualize, compare_and_save


# Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT /"test"/ "data"
RASTER_OUTPUT_DIR = DATA_DIR / "Raster" / "R10m_geotiff" / "Milano" / "Output"

CUBE_PATH = RASTER_OUTPUT_DIR / "Milano_B02_B03_B04_B08_norm.tif"
MASK_PATH = RASTER_OUTPUT_DIR / "Milano_mask.tif"

OUT_DIR = RASTER_OUTPUT_DIR / "Classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Parameters

# ROI (recommended for demo runs)
USE_ROI = True

# ROI format: (row_start, row_end, col_start, col_end)
# Example: a 1024x1024 window from the top-left area
ROI_WINDOW = (2000, 3024, 2000, 3024)

# fast mode (recommended ON for demo run)
FAST_MODE = True

# Unsupervised params
N_CLUSTERS = 8
UNSUP_METHOD = "kmeans"

# Fast-mode params for unsupervised (MiniBatchKMeans)
UNSUP_SAMPLE_SIZE = 200000         # how many pixels to fit
UNSUP_BATCH_SIZE = 4096
UNSUP_MAX_ITER = 100
UNSUP_PREDICT_CHUNK = 500000       # chunk size for predicting labels on all pixels

# Supervised params
SUP_MODEL = "rf"                   # "rf" or "svm"

# In fast_mode, reduce pseudo-label training size to keep it quick
SAMPLE_PER_CLASS = 5000 if FAST_MODE else 20000

# output coding
NODATA_LABEL = 0
LABEL_OFFSET = 1

RANDOM_STATE = 42

VISUALIZE = True
EXPORT_LABELS_RGB = True


def run_classification():
    # 1) load cube (full or ROI)
    if USE_ROI:
        meta_cube, cube = load_raster_roi(str(CUBE_PATH), ROI_WINDOW)  # (B,H,W)
    else:
        meta_cube, cube = load_raster(str(CUBE_PATH))  # (B,H,W)

    if cube.ndim != 3:
        raise ValueError(f"Expected multiband cube (B,H,W), got shape {cube.shape}")

    b, h, w = cube.shape
    print("Cube shape:", cube.shape)
    print("Cube dtype:", cube.dtype)
    print("USE_ROI:", USE_ROI)
    if USE_ROI:
        print("ROI_WINDOW:", ROI_WINDOW)

    # 2) load & align mask to cube grid
    mask = load_mask_aligned(str(MASK_PATH), meta_cube)  # (H,W) boolean

    print("Mask shape:", mask.shape)
    print("Valid pixels %:", float(mask.sum()) / mask.size)

    # 3) feature extraction: (N,B)
    features, idx, (H, W) = extract_pixel_features(cube, mask)
    print("Features shape:", features.shape)

    mins = np.nanmin(features, axis=0)
    maxs = np.nanmax(features, axis=0)
    print("Per-band feature min:", mins)
    print("Per-band feature max:", maxs)

    # 4) unsupervised pre-classification
    labels_init = unsupervised_cluster(
        features,
        method=UNSUP_METHOD,
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        fast_mode=FAST_MODE,
        sample_size=UNSUP_SAMPLE_SIZE,
        batch_size=UNSUP_BATCH_SIZE,
        max_iter=UNSUP_MAX_ITER,
        predict_chunk_size=UNSUP_PREDICT_CHUNK,
    )
    print("labels_init classes:", np.unique(labels_init).size)

    # 5) supervised refinement (pseudo-label training)
    labels_final = supervised_classify(
        features,
        labels_init,
        model=SUP_MODEL,
        sample_per_class=SAMPLE_PER_CLASS,
        random_state=RANDOM_STATE,
    )
    print("labels_final classes:", np.unique(labels_final).size)

    # 6) labels reshape back to raster
    init_map = reshape_labels_to_raster(
        labels_init, H, W, mask,
        nodata_label=NODATA_LABEL,
        label_offset=LABEL_OFFSET,
    ).astype(np.uint16)

    final_map = reshape_labels_to_raster(
        labels_final, H, W, mask,
        nodata_label=NODATA_LABEL,
        label_offset=LABEL_OFFSET,
    ).astype(np.uint16)

    # 7) save label rasters
    init_out = OUT_DIR / "Milano_labels_init.tif"
    final_out = OUT_DIR / "Milano_labels_final.tif"

    save_raster(str(init_out), meta_cube, init_map, dtype="uint16", nodata=NODATA_LABEL)
    save_raster(str(final_out), meta_cube, final_map, dtype="uint16", nodata=NODATA_LABEL)

    print("FAST_MODE:", FAST_MODE)
    print("Saved:", init_out)
    print("Saved:", final_out)

    # 8) visualize
    roi_rgb = visualize(
        cube,
        kind="cube",
        title="ROI RGB",
        show=VISUALIZE,
        stretch=None,
        low=0.5,
        high=99.8,
        gamma=1.5,
    )

    label_rgb = visualize(
        final_map,
        kind="label",
        nodata=NODATA_LABEL,
        seed=42,
        title="labels_final",
        show=VISUALIZE,
    )

    # side-by-side compare and save
    compare_png = OUT_DIR / "ROI_vs_labels_final.png"
    compare_and_save(
        left_rgb_u8=roi_rgb,
        right_rgb_u8=label_rgb,
        left_title="ROI RGB",
        right_title="Labels (colored)",
        out_path=str(compare_png),
    )
    print("Saved comparison PNG:", compare_png)

    # optional: export label RGB as a 3-band GeoTIFF
    if EXPORT_LABELS_RGB:
        rgb_out = OUT_DIR / "Milano_labels_final_RGB.tif"
        rgb_bhw = np.transpose(label_rgb, (2, 0, 1))  # (B,H,W)
        save_raster(str(rgb_out), meta_cube, rgb_bhw, dtype="uint8", nodata=0)
        print("Saved RGB preview:", rgb_out)


if __name__ == "__main__":
    run_classification()
