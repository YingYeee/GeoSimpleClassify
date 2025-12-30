from pathlib import Path

from geosimpleclassify.config.schema import default_cfg
from geosimpleclassify.pipelines.run_preprocess import run_preprocess
from geosimpleclassify.pipelines.run_classify import run_classification


def main():
    cfg = default_cfg()


    # 1) define path config
    cfg.paths.project_root = Path(__file__).resolve().parent
    cfg.paths.data_dir = cfg.paths.project_root /"test"/ "data"

    # preprocess paths
    cfg.paths.preprocess_input_raster_dir = cfg.paths.data_dir /"raw"/ "Raster"/"Milano_sample"
    cfg.paths.preprocess_input_vector_path = cfg.paths.data_dir /"raw"/"Vector" / "ProvCM01012025_WGS84.shp"
    cfg.paths.preprocess_derived_dir = cfg.paths.data_dir /"derived"
    cfg.paths.preprocess_derived_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.preprocess_final_dir = cfg.paths.data_dir /"final"/"preprocess"
    cfg.paths.preprocess_final_dir.mkdir(parents=True, exist_ok=True)

    # classify paths
    cfg.paths.classify_input_dir = cfg.paths.preprocess_final_dir
    cfg.paths.classify_final_dir = cfg.paths.data_dir /"final"/"classify"
    cfg.paths.classify_final_dir.mkdir(parents=True, exist_ok=True)


    # 2) define preprocess parameters
    cfg.preprocess.band_codes = ["B02", "B03", "B04", "B08"]    # bands to use for building the data cube
    cfg.preprocess.clip_attr_name = "DEN_CM"    # attribute name used for vector clipping
    cfg.preprocess.clip_attr_value = "Milano"   # attribute value used for vector clipping
    cfg.preprocess.nodata_value = 0 # nodata value used for mask and outputs


    # 3) define classify parameters
    cfg.classify.band_codes = ["B02", "B03", "B04", "B08"]  # bands (should match preprocess output)
    cfg.classify.random_state = 42  # random seed
    cfg.classify.visualize_show = True  # show visualization results
    cfg.classify.export_labels_rgb = True   # save RGB visualization of labels

    # ROI settings
    cfg.classify.use_roi = False # Select ROI to classify to avoid overload when the data is large
    cfg.classify.roi_window = (2000, 3024, 2000, 3024)  # 1024x1024, (row_start, row_end, col_start, col_end)

    # unsupervised classify parameters
    cfg.classify.unsup_method = "kmeans"    # unsupervised method; it cannot be changed now
    cfg.classify.n_clusters = 8 # number of clusters

    # FAST MODE: use MiniBatchKMeans with sampling to reduce classification time (recommend to use in demo)
    cfg.classify.fast_mode = True
    cfg.classify.unsup_sample_size = 200000 # number of samples for MiniBatchKMeans fitting
    cfg.classify.unsup_batch_size = 4096    # batch size for MiniBatchKMeans
    cfg.classify.unsup_max_iter = 100   # maximum training iterations
    cfg.classify.unsup_predict_chunk = 500000   # chunk size for prediction

    # supervised classify parameters
    cfg.classify.sup_model = "rf"   # supervised model, ["rf", "svm"]

    # output coding settings
    cfg.classify.nodata_label = 0   # label value for nodata
    cfg.classify.label_offset = 1   # Offset added to labels before writing (e.g., reserve 0 as nodata, 1 as the first class)


    # 4) run preprocess
    run_preprocess(cfg)


    # 5) run classification
    run_classification(cfg)


if __name__ == "__main__":
    main()
