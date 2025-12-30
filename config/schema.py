from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


# PATH
@dataclass
class PathsCfg:
    project_root: Optional[Path] = None
    data_dir: Optional[Path] = None

    # preprocess inputs
    preprocess_input_raster_dir: Optional[Path] = None
    preprocess_input_vector_path: Optional[Path] = None

    # classify inputs
    classify_input_dir: Optional[Path] = None

    # derived outputs
    preprocess_derived_dir: Optional[Path] = None

    # final outputs
    preprocess_final_dir: Optional[Path] = None
    classify_final_dir: Optional[Path] = None


# PREPROCESS PARAMETERS
@dataclass
class PreprocessCfg:
    band_codes: Tuple[str, ...] = None  # bands to use for building the data cube. (e.g. ["B02", "B03", "B04", "B08"])
    clip_attr_name: str = None  # attribute name used for vector clipping. (e.g. "DEN_CM")
    clip_attr_value: str = None # attribute value used for vector clipping. (e.g. "Milano")
    clip_low: float = 1.0   # lower percentile for value clipping
    clip_high: float = 99.0 # upper percentile for value clipping
    per_band_clip: bool = True  # apply percentile clipping per band
    per_band_normalize: bool = True # normalize each band independently
    nodata_value: int = 0   # nodata value used for mask and outputs


# CLASSIFY PARAMETERS
@dataclass
class ClassifyCfg:
    band_codes: Tuple[str, ...] = None

    # ROI (recommended for demo runs)
    use_roi: bool = True    # whether to classify only use a region of interest
    roi_window: Tuple[int, int, int, int] = None    # (row_start, row_end, col_start, col_end)

    # Unsupervised params
    unsup_method: str = "kmeans"    # unsupervised method ("kmeans")
    n_clusters: int = 8 # number of clusters

    # Fast-mode params for unsupervised (MiniBatchKMeans)
    fast_mode: bool = True  # use fast (approximate) clustering
    unsup_sample_size: int = 200000 # number of samples for fitting
    unsup_batch_size: int = 4096    # batch size for MiniBatchKMeans
    unsup_max_iter: int = 100   # maximum training iterations
    unsup_predict_chunk: int = 500000   # chunk size for prediction

    # Supervised params
    sup_model: str = "rf"  # supervised model, ["rf", "svm"]
    sample_per_class: Optional[int] = 5000 if fast_mode else 20000  # samples per class

    # Output coding
    nodata_label: int = 0   # label value for nodata
    label_offset: int = 1   # offset applied to predicted labels

    random_state: int = 42  # random seed
    visualize_show: bool = True # show visualization results
    export_labels_rgb: bool = True  # save RGB visualization of labels


# COMBINED PIPELINE CONFIG
@dataclass
class PipelineCfg:
    paths: PathsCfg = field(default_factory=PathsCfg)
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    classify: ClassifyCfg = field(default_factory=ClassifyCfg)


def default_cfg():
    return PipelineCfg()