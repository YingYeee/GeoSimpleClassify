# GeoSimpleClassify
A two-stage geospatial pipeline for remote-sensing land-cover classification.

- **Stage 1 (Preprocess)**: Mosaics and clips raster tiles, stacks spectral bands, applies numerical preprocessing, 
and exports a normalized multiband cube with a validity mask.  
  (heavy I/O, low-frequency, cache-oriented).
- **Stage 2 (Classify)**: Performs pixel-wise land-cover classification using unsupervised clustering 
to generate pseudo-labels, followed by supervised refinement, 
with optional **ROI-based** processing and a **fast mode (MiniBatchKMeans)** for efficient experimentation on large rasters.

The design goal is an **engineering-friendly workflow**: minimal manual annotation,
clear stage separation, and CPU-friendly execution.

---

## Installation

```bash
conda env create -f environment.yml
conda activate geosimpleclassify_env
```

---

## Project Structure

```
project_root/
├── geosimpleclassify/                  # Core library
│   ├── core/                           # Core algorithms
│   │   ├── geo_io.py                   # Raster/vector I/O, reprojection, ROI loading, mask alignment
│   │   ├── mosaic.py                   # Raster tile mosaicking logic (Stage 1)
│   │   ├── clip.py                     # Vector-based raster clipping (Stage 1)
│   │   ├── preprocess.py               # Numerical preprocessing and mask generation (Stage 1)
│   │   ├── feature.py                  # Pixel-level feature extraction (Stage 2)
│   │   ├── unsupervised.py             # Unsupervised clustering, using KMeans or MiniBatchKMeans (Stage 2)
│   │   ├── supervised.py               # Supervised refinement, using RF or SVM (Stage 2)
│   │   └── postprocess.py              # Convert raster, visualization and statistics (Stage 2)
│   │
│   ├── pipelines/                      # Pipeline entry scripts
│   │   ├── run_preprocess.py           # Stage 1: preprocessing pipeline
│   │   └── run_classify.py             # Stage 2: classification pipeline
│   │
│   └── config/                         # Configuration schemas
│       └── schema.py                   # Dataclass-based configuration definitions
│
├── test/                               # Tests and sample data
│   ├── data/                           # Shared test and demo data
│   │   └── raw/                        # Raw raster and vector samples (small subset)
│   ├── unit/                           # Unit and module integration tests
│   └── e2e/                            # End-to-end tests
│
├── notebooks/                          # Development and demonstration notebooks
│
├── demo.py                             # End-to-end demo, including stage 1 and stage 2 (uses test/data)
├── environment.yml                     # Conda environment definition
├── .gitignore
├── LICENSE
└── README.md

```

---

## Pipeline Overview

### A. Inputs and Outputs

**Inputs**
- Multiple single-band GeoTIFF rasters (tiles)
- Vector data (Shapefile) for spatial clipping (AOI/ROI boundary)
- (Stage 2) Preprocessed multi-band raster + aligned mask

**Outputs**
- Normalized multi-band GeoTIFF (data cube)
- Binary mask raster
- Initial label raster (unsupervised clustering)
- Final label raster (supervised refinement)
- RGB previews and statistics:
  - `labels_final_summary.csv`
  - `labels_final_histogram.png`
  - side-by-side comparison PNG, including raw image and classification result
  - (optional) `labels_final_RGB.tif`

---

### B. Overall Workflow

**Stage 1 – Preprocessing (low-frequency, heavy computation)**

Input:
- Raw raster tiles
- Vector boundary (Shapefile)

Operations:
1. Mosaic raster tiles
2. Clip by vector boundary
3. Stack selected bands into a multi-band cube
4. Normalize bands and generate valid-pixel mask
5. Save outputs as cache for Stage 2

Output:
- Normalized multi-band raster
- Aligned mask raster

**Stage 2 – Classification (high-frequency, parameter-tunable)**

Input:
- Preprocessed raster cube + mask

Operations:
1. Optional ROI window loading
2. Pixel-wise feature extraction
3. Unsupervised clustering to generate pseudo labels
4. Supervised refinement using pseudo labels
5. Raster reconstruction and visualization

Output:
- Initial and final label rasters
- RGB previews and statistics

---

### C. Module Responsibilities

#### geo_io.py
Raster/vector I/O and geospatial alignment utilities.

This module handles all low-level raster and vector access. Its main responsibility is to ensure that
rasters, masks, and vector-derived outputs are aligned on the same grid and share consistent spatial
metadata, which is critical for pixel-wise operations in later stages.

Core functions:
- `load_raster(path)`: Load a raster into `(B, H, W)` along with metadata (CRS, transform, resolution).
- `load_raster_roi(path, roi_window)`: Load a raster subset defined by `(row_start, row_end, col_start, col_end)`.
- `load_mask_aligned(mask_path, meta_ref)`: Align a mask raster to a reference grid (reproject/resample if needed).
- `align_raster_to_meta(src_path, meta_ref, ...)`: Reproject and align a raster to a target grid.

Key idea:
- Keep raster cubes and masks on the **same grid** to guarantee correct pixel indexing.

Used by:
- `run_preprocess.py`
- `run_classify.py`

Consumes output of:
- External raster and vector data sources

---

#### mosaic.py
Raster tile mosaicking logic (Stage 1).

This module merges multiple raster tiles of the same spectral band into a single raster. It validates CRS,
resolution, and grid consistency before merging to avoid downstream misalignment issues.

Core function:
- `merge_rasters(raster_paths, out_path)`: Merges multiple raster tiles into a single mosaic after validating spatial consistency.

Used by:
- `run_preprocess.py`

Consumes output of:
- Raw raster tiles

---

#### clip.py
Vector-based raster clipping (Stage 1).

This module applies polygon boundaries to raster data to restrict processing to a region of interest.
It supports attribute-based filtering to select specific geometries from vector files.

Core function:
- `clip_raster_with_vector(raster_path, vector_path, ...)`: Clips a raster using vector boundaries with optional attribute-based feature selection.

Used by:
- `run_preprocess.py`

Consumes output of:
- `mosaic.py`

---

#### preprocess.py
Numerical preprocessing utilities (Stage 1).

This module performs array-level preprocessing independent of geospatial metadata. It focuses on cleaning
and normalizing raster values prior to classification.

Core functions:
- `stack_bands(band_arrays)`: Stack multiple 2D bands into a `(B, H, W)` cube.
- `clip_by_percentile(cube, lower, upper)`: Remove extreme outliers.
- `normalize(cube)`: Apply per-band min–max normalization.
- `make_valid_mask(cube)`: Generate a unified valid-pixel mask across bands.

Used by:
- `run_preprocess.py`

Consumes output of:
- `clip.py`

---

#### feature.py
Pixel-level feature extraction (Stage 2).

This module converts a preprocessed raster cube into a feature matrix suitable for machine learning,
while preserving index mappings required for reconstructing raster-shaped outputs.

Core function:
- `extract_pixel_features(cube, mask)`: Extract `(N, B)` feature matrix and index mapping from raster data.

Used by:
- `run_classify.py`

Consumes output of:
- `run_preprocess.py`

---

#### unsupervised.py
Unsupervised clustering for pseudo-label generation (Stage 2).

This module generates coarse labels via clustering without manual annotation. It supports both standard
KMeans and a fast MiniBatchKMeans mode optimized for large rasters.

Core function:
- `unsupervised_cluster(features, n_clusters, fast_mode=...)`: Cluster pixel features into pseudo classes.

Used by:
- `run_classify.py`

Consumes output of:
- `feature.py`

---

#### supervised.py
Supervised refinement on pseudo labels (Stage 2).

This module trains a supervised classifier (Random Forest or SVM) using pseudo labels and refines
predictions across all valid pixels.

Core function:
- `supervised_classify(features, labels_init, model=...)`: Refine labels using supervised learning.

Used by:
- `run_classify.py`

Consumes output of:
- `unsupervised.py`

---

#### postprocess.py
Result reconstruction, visualization and statistics (Stage 2).

This module converts 1D label arrays back into raster grids and produces human-readable outputs,
including RGB previews and statistical summaries.

Core functions:
- `reshape_labels_to_raster`: Reconstruct raster-shaped label maps.
- `visualize`: Generate RGB previews for visualization.
- `save_label_summary`: Compute class area statistics.
- `plot_label_histogram`: Plot class distribution histograms.

Used by:
- `run_classify.py`

Consumes output of:
- `supervised.py`

---

### D. Configuration (key parameters)

Configuration is defined in `config/schema.py` using dataclasses. The demo and pipelines use `default_cfg()`
and override selected fields as needed.

**Stage 1 key parameters**
- `cfg.preprocess.band_codes`: Bands used to build the multi-band data cube  
  (e.g. `("B02", "B03", "B04", "B08")`).

- `cfg.preprocess.clip_attr_name`: Vector attribute name used to select the target region for clipping  
  (e.g. `"DEN_CM"`).

- `cfg.preprocess.clip_attr_value`: Attribute value identifying the region of interest  
  (e.g. `"Milano"`).

- `cfg.preprocess.clip_low` / `cfg.preprocess.clip_high`: Lower and upper percentiles for value clipping to remove outliers.

**Stage 2 key parameters**
- `cfg.classify.use_roi`: enable ROI-based classification
- `cfg.classify.roi_window`: `(r0, r1, c0, c1)` window
- `cfg.classify.fast_mode`: enable MiniBatchKMeans
- `cfg.classify.n_clusters`: number of clusters
- `cfg.classify.sup_model`: `"rf"` or `"svm"`
- `cfg.classify.sample_per_class`: samples per class for pseudo-label training
- `cfg.classify.nodata_label` / `cfg.classify.label_offset`: output label encoding
- `cfg.classify.export_labels_rgb`: export `labels_final_RGB.tif`

**Paths**
- `cfg.paths.preprocess_*`: Stage 1 inputs/outputs
- `cfg.paths.classify_input_dir`: Stage 1 output directory
- `cfg.paths.classify_final_dir`: Stage 2 outputs

---

### E. Assumptions and Limitations

#### Data Assumptions
- **Input data consistency**:  
  * Input rasters are assumed to follow a consistent band order, CRS, nodata definition, folder structures and naming conventions.  
  * The pipeline does not perform automatic validation or correction of inconsistent inputs.

#### Runtime Environment
- Python 3.10  
- Dependencies are managed via `environment.yml`

#### Current Functional Limitations
The following features are intentionally not supported in the current implementation:
- **Alternative clustering methods**: only uses KMeans / MiniBatchKMeans / RF / SVM.
- **Automatic sliding-window classification**: ROI windows are user-defined and processed one at a time via explicit configuration.

- **Automatic semantic label naming**: output labels remain numeric class IDs.


---

## Testing

Tests are implemented with `unittest` and organized as follows:

```
test/
├── data/                     # shared test data
│   └── raw/                  # raw raster and vector samples (small subset)
├── unit/                     # unit integration tests
│   ├── test_geo_io.py
│   ├── test_mosaic.py
│   ├── test_clip.py
│   ├── test_preprocess.py
│   ├── test_feature.py
│   ├── test_unsupervised.py
│   ├── test_supervised.py
│   └── test_postprocess.py
└── e2e/                      # end-to-end tests (slower)
    ├── test_preprocess_e2e.py
    └── test_classification_e2e.py

```

Run only unit tests:

```bash
python -m unittest discover -s test/unit -p "test_*.py"
```

Run only e2e tests:

```bash
python -m unittest discover -s test/e2e -p "test_*.py"
```

Notes:
- Unit tests focus on individual core functions integration,
  mainly validating shapes, types, and deterministic outputs using
  small synthetic arrays or minimal fixtures.

- E2E tests validate full pipeline contracts:
  * `test_preprocess_e2e.py` checks the Stage 1 preprocessing pipeline
    (mosaic → clip → stack/normalize → mask / GeoTIFF export) using
    small sample data under test/data/raw.
  * `test_classification_e2e.py` checks the Stage 2 classification pipeline
    (feature extraction → unsupervised clustering → supervised refinement
    → postprocess) on small synthetic data for speed and robustness.

---

## Example

```bash
python demo.py
```

For detailed examples and figures, see `notebooks/demo_milano_sample.ipynb` please.

---

## Contribution

### Ye Ying
- Prepared the raw input datasets.
- Implemented preprocessing modules.
- Developed the Jupyter notebook examples.
- Developed unit tests for preprocessing-related modules.

### Tian Yueling
- Implemented classification modules and related tests.
- Refactored experimental code into a reusable, configurable two-stage Python pipeline.
- Developed the end-to-end demo script.
- Consolidated the overall project structure and final documentation.
