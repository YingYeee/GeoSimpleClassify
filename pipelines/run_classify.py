"""
Classify pipeline: Feature extraction + Unsupervised (KMeans) + Supervised refinement (RF/SVM) + Visualization

Inputs (preprocessed data):
  - B02_B03_B04_B08_norm.tif   (4-band, normalized)
  - B02_B03_B04_B08_mask.tif   (mask; will be aligned to cube grid)

Outputs:
  - B02_B03_B04_B08_labels_init.tif            (unsupervised labels)
  - B02_B03_B04_B08_labels_final.tif           (supervised refined labels)
"""


import numpy as np

from geosimpleclassify.core.geo_io import load_raster, load_raster_roi, save_raster, load_mask_aligned
from geosimpleclassify.core.feature import extract_pixel_features
from geosimpleclassify.core.unsupervised import unsupervised_cluster
from geosimpleclassify.core.supervised import supervised_classify
from geosimpleclassify.core.postprocess import reshape_labels_to_raster, visualize, compare_and_save
from geosimpleclassify.config.schema import default_cfg, PipelineCfg


def find_cube_and_mask(preprocessed_dir, band_codes):
    """
    Locate the preprocessed data cube and corresponding mask.

    Parameters
    ----------
    preprocessed_dir : pathlib.Path
        Directory containing preprocessed raster outputs or external directory
    band_codes : list of str
        Band identifiers used to build the cube (e.g. ["B02","B03","B04","B08"]).

    Returns
    -------
    cube_path : pathlib.Path
        Path to the normalized multiband cube.
    mask_path : pathlib.Path
        Path to the corresponding mask raster.
    """
    bands_tag = "_".join(band_codes)

    cube_candidates = sorted(preprocessed_dir.glob(f"*{bands_tag}*_norm.tif"))
    mask_candidates = sorted(preprocessed_dir.glob(f"*{bands_tag}*_mask.tif"))

    if len(cube_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 cube '*{bands_tag}*_norm.tif' in {preprocessed_dir}, "
            f"got {len(cube_candidates)}: " + ", ".join(p.name for p in cube_candidates)
        )
    if len(mask_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 mask '*{bands_tag}*_mask.tif' in {preprocessed_dir}, "
            f"got {len(mask_candidates)}: " + ", ".join(p.name for p in mask_candidates)
        )

    return cube_candidates[0], mask_candidates[0]


def get_classify_output_subdir(
    root_dir,
    use_roi = False,
    roi_window = None
):
    """
    Generate the output subdirectory for a classification run.

    Parameters
    ----------
    root_dir : pathlib.Path
        Root directory for classification outputs.
    use_roi : bool, optional
        Whether classification is performed on a region of interest (ROI).
        If False, the subdirectory name will be "full".
    roi_window : tuple of int, optional
        ROI window defined as (row_start, row_end, col_start, col_end).
        Required if use_roi is True.

    Returns
    -------
    pathlib.Path
        Path to the classification output subdirectory.
    """
    if not use_roi:
        sub = "full"
    else:
        if not roi_window:
            raise ValueError("Expected 1 tuple as the ROI_window")
        else:
            r0, r1, c0, c1 = roi_window
            sub = f"ROI_r{r0}_r{r1}_c{c0}_c{c1}"

    return root_dir / sub


def run_classification(cfg: PipelineCfg):
    # 1) load cube (full or ROI)
    cube_path, mask_path = find_cube_and_mask(cfg.paths.classify_input_dir, cfg.classify.band_codes)
    if cfg.classify.use_roi:
        meta_cube, cube = load_raster_roi(str(cube_path), cfg.classify.roi_window)  # (B,H,W)
    else:
        meta_cube, cube = load_raster(str(cube_path))  # (B,H,W)

    if cube.ndim != 3:
        raise ValueError(f"Expected multiband cube (B,H,W), got shape {cube.shape}")

    b, h, w = cube.shape
    print("Cube shape:", cube.shape)
    print("Cube dtype:", cube.dtype)
    print("USE_ROI:", cfg.classify.use_roi)
    if cfg.classify.use_roi:
        print("ROI_WINDOW:", cfg.classify.roi_window)

    # 2) load & align mask to cube grid
    mask = load_mask_aligned(str(mask_path), meta_cube)  # (H,W) boolean

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
        method=cfg.classify.unsup_method,
        n_clusters=cfg.classify.n_clusters,
        random_state=cfg.classify.random_state,
        fast_mode=cfg.classify.fast_mode,
        sample_size=cfg.classify.unsup_sample_size,
        batch_size=cfg.classify.unsup_batch_size,
        max_iter=cfg.classify.unsup_max_iter,
        predict_chunk_size=cfg.classify.unsup_predict_chunk,
    )
    print("labels_init classes:", np.unique(labels_init).size)

    # 5) supervised refinement (pseudo-label training)
    labels_final = supervised_classify(
        features,
        labels_init,
        model=cfg.classify.sup_model,
        sample_per_class=cfg.classify.sample_per_class,
        random_state=cfg.classify.random_state,
    )
    print("labels_final classes:", np.unique(labels_final).size)

    # 6) labels reshape back to raster
    init_map = reshape_labels_to_raster(
        labels_init, H, W, mask,
        nodata_label=cfg.classify.nodata_label,
        label_offset=cfg.classify.label_offset,
    ).astype(np.uint16)

    final_map = reshape_labels_to_raster(
        labels_final, H, W, mask,
        nodata_label=cfg.classify.nodata_label,
        label_offset=cfg.classify.label_offset,
    ).astype(np.uint16)

    # 7) save label rasters
    output_dir = get_classify_output_subdir(cfg.paths.classify_final_dir, cfg.classify.use_roi, cfg.classify.roi_window)
    output_dir.mkdir(parents=True, exist_ok=True)

    bands_tag = "_".join(cfg.classify.band_codes)
    init_out  = output_dir / f"{bands_tag}_labels_init.tif"
    final_out = output_dir / f"{bands_tag}_labels_final.tif"

    save_raster(str(init_out), meta_cube, init_map, dtype="uint16", nodata=cfg.classify.nodata_label)
    save_raster(str(final_out), meta_cube, final_map, dtype="uint16", nodata=cfg.classify.nodata_label)

    print("FAST_MODE:", cfg.classify.fast_mode)
    print("Saved:", init_out)
    print("Saved:", final_out)

    # 8) visualize
    roi_rgb = visualize(
        cube,
        kind="cube",
        title="ROI RGB",
        show=cfg.classify.visualize_show,
        stretch=None,
        low=0.5,
        high=99.8,
        gamma=1.5,
    )

    label_rgb = visualize(
        final_map,
        kind="label",
        nodata=cfg.classify.nodata_label,
        seed=cfg.classify.random_state,
        title="labels_final",
        show=cfg.classify.visualize_show,
    )

    # side-by-side compare and save
    compare_png = output_dir / "ROI_vs_labels_final.png"
    compare_and_save(
        left_rgb_u8=roi_rgb,
        right_rgb_u8=label_rgb,
        left_title="ROI RGB",
        right_title="Labels (colored)",
        out_path=str(compare_png),
    )
    print("Saved comparison PNG:", compare_png)

    # optional: export label RGB as a 3-band GeoTIFF
    if cfg.classify.export_labels_rgb:
        rgb_out = output_dir / "labels_final_RGB.tif"
        rgb_bhw = np.transpose(label_rgb, (2, 0, 1))  # (B,H,W)
        save_raster(str(rgb_out), meta_cube, rgb_bhw, dtype="uint8", nodata=0)
        print("Saved RGB preview:", rgb_out)


if __name__ == "__main__":
    cfg = default_cfg()
    run_classification(cfg)
