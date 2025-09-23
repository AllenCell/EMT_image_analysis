# Instructions to run all cells mask (ACM) inference generation pipeline and postprocessing steps

## Installation

1. Clone this git repository and enter the inference directory.

   `git clone https://github.com/AllenCell/EMT_image_analysis.git`

   `cd EMT_image_analysis/Colony_mask_training_inference`

2. Install Python 3.10, either from [python.org](https://www.python.org/downloads/), your operating system package manager, or [pyenv](https://github.com/pyenv/pyenv-installer).
Check that it is installed correctly by running `python --version` in the terminal.
Then, you can use `Conda` or `PDM` to set up the Python environment using the provided requirements.txt file.

For Conda, follow these instructions:
```
conda env create -f acm-eval.yaml
conda activate acm-eval
```

For PDM, follow these instructions:
```
pipx install pdm
pdm init
pdm add -d -r requirements.txt
```

---

## Run all cells mask model training [Under Development - ZARR support not yet implemented]
Users are welcome to retrain the models by accessing all the images provided in https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/

To re-run the whole training run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.train experiment=im2im/train.yaml`

This part is not yet fully supported as CytoDL training using OME ZARR files is still under development. For retraining, users can use tiff files instead.

---

The following steps outline the workflow for generating All Cells Mask (ACM) outputs and postprocessing them using the `cyto-dl` pipeline.

## Run multi-scale patch-based evaluation to generate probability maps

**Step 1 - Download the model checkpoint**  
The model checkpoint path is required to generate the ACM. A pretrained model is provided and can be downloaded from this link -  https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/supplemental_files/cytodl_checkpoints/all_cells_mask_seg_model_checkpoint.ckpt 
Access the link and click on the "DOWNLOAD FILE" button [top left].  

Alternatively, user can use `curl` to download using the link - https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/supplemental_files/cytodl_checkpoints/all_cells_mask_seg_model_checkpoint.ckpt?versionId=ejf07rBiw5slyx1zQyurfX6.zpSJ92JM


Create a new sub-directory using the command `mkdir data` inside `Colony_mask_training_inference` directory.

Save the downloaded checkpoint file in `EMT_image_analysis/Colony_mask_training_inference/data/`  
Users are welcome to train their own models or finetune the existing model with their own data.  

---

**Step 2 - Prepare CSV Files from `.ome.zarr` (or your data files) for ACM generation**

Data (3D Z-stack of a single timepoint or a timelapse) is provided as an input to the model to predict its all-cells-mask. `cyto-dl` expects inputs in CSV format, use the `csv_creator.py` script to generate the initial CSV file with path to your data files.

The CSV file should contain these 2 columns: movie_path and bf_channel. E.g., 

|count            |movie_path         |bf_channel        |
|-----------------|-------------------|------------------|
|0                |https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/data/3500006062_2_raw_converted.ome.zarr    |0                 |
|1                |https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/data/3500006062_4_raw_converted.ome.zarr    |0                 |

We recommend storing this CSV inside the `Colony_mask_training_inference/data` directory.

Users are welcome to experiment using this table and save it as a CSV file named  `predict_all_cells_mask_zarr_aws_v0.csv` inside the `Colony_mask_training_inference/data` directory.

---

**Step 3 - Split CSV for Parallel Processing**

This step is optional (but recommended) and you can proceed with a single csv generated from the previous step if your compute allows for it.

`cyto-dl` uses a `DataFrameLoader` that loads all entries from the input CSV. For large datasets, this leads to inefficient processing. To improve performance, we **split the CSV** into smaller chunks using the `csv_splitter.py` script. You can set the desired chunk size in the script. In case your csv is named `predict_all_cells_mask_zarr_aws_v0.csv` inside the `Colony_mask_training_inference/data` directory, you can run `csv_splitter.py --csv_file data/predict_all_cells_mask_zarr_aws_v0.csv --chunk_size 5` and this will create `n%5` (where `n` is the total number of rows in the csv and `5` is the `chunk size`) number of csv files with names such as `predict_all_cells_mask_zarr_aws_v0_p1.csv` and so on in the same path as the input csv.

---

**Step 4 - Generate YAML Configs for Inference**

multi-scale patch-based evaluation runs on 3 different patch sizes to generate the prediction. To run prediction on each patch, the evaluation config files (provided in configs/experiment/im2im/eval_scale1.yaml, configs/experiment/im2im/eval_scale2.yaml, and configs/experiment/im2im/eval_scale3.yaml) has to be modified.

As `cyto-dl` requires YAML config files for inference. You can start from a base YAML and generate customized versions for each data chunk using the `generate_eval_yamls.py` script. This script allows **scale-specific YAML generation**. Here, we provide `template_scale1.yaml` as a base yaml file to generate your own yaml files.

### Arguments

- `--scale` (int, required): Scale level. Must be 1, 2, or 3.
- `--batch_size` (int, required): Batch size for each YAML config.
- `--save_dir` (str, required): Directory to save generated YAMLs.
- `--start_index` (int, default=1): Start index for CSV parts (e.g., 1 for _p1.csv).
- `--template_yaml` (str, required): Path to the base template YAML file.
- `--data_dir` (str, required): Directory containing the chunked CSV files.
- `--csv_base_name` (str, required): Base name for chunked CSV files (e.g., 'all_moviepaths_qc').

In the `template_scale1.yaml` file, make sure to adjust the correct checkpoint path in line 21 (`ckpt_path`). If you downloaded the checkpoint following step 1 and saved in `EMT_image_analysis/Colony_mask_training_inference/data/`, then your `ckpt_path` should be `EMT_image_analysis/Colony_mask_training_inference/data/all_cells_mask_seg_model_checkpoint.ckpt`.

### Example Usage

```
python generate_eval_yamls.py \
  --scale 1 \
  --batch_size 4 \
  --save_dir ./eval_yamls \
  --template_yaml ./template_scale1.yaml \
  --data_dir ./data \
  --csv_base_name all_moviepaths_qc
```

This will generate YAML files like `all_moviepaths_qc_scale1_p1.yaml`, `all_moviepaths_qc_scale1_p2.yaml`, etc., in the `./eval_yamls` directory, one for each chunked CSV file found in `./data` per scale.

---

**Step 5 - Run Inference to generate outputs per scale**

To run inference on patch1 run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale1.yaml`

To run inference on patch2 run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale2.yaml`

To run inference on patch3 run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale3.yaml`

(e.g. if your yaml is named as `all_moviepaths_qc_scale1_p1.yaml` and stored under `EMT_image_analysis/Colony_mask_training_inference/configs/experiment/im2im/generated_yamls/`, make sure to run with `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/generated_yamls/all_moviepaths_qc_scale1_p1.yaml`)

To run on your own model, edit the yaml file ckpt_path with the path to your model.

Predictions for each patch based predictions will be stored at the target location provided in save_dir. By default this location is `Colony_mask_training_inference/data/infer_movie_multiscale_patch1`, `Colony_mask_training_inference/data/infer_movie_multiscale_patch2`, and `Colony_mask_training_inference/data/infer_movie_multiscale_patch3` for different patches.

---

**Step 6 - Run postprocessing script to generate the all cells mask**

Run the command --> `python process_scene_runner.py`

The code expects patch-based probabilty masks have already been generated and were stored at `/path/to/Colony_mask_training_inference/data/all_cells_mask_test_dir/eval_whole_movie_multiscale_patchX` (where, X={1, 2, 3})

The output of this script will be the binarized all cells masks and they can be accessed at `/pathto/Colony_mask_training_inference/data/all_cells_mask_test_dir/multiscale_all_cells_mask_v0`

---

**Step 7 - CSV creation for feature extraction step**

Run `acm_postprocessing.py` to obtain a CSV with initial data paths and their corresponding all cells mask path. This csv can be used for feature extraction step.

---

## Summary of Scripts

| Script                   | Purpose                                 |
| ------------------------ | --------------------------------------- |
| `csv_creator.py`         | Generate base CSV from `.ome.zarr`      |
| `csv_splitter.py`    | Split large CSVs into chunks            |
| `generate_eval_yamls.py` | Generate inference YAMLs per chunk      |
| `run.sh`                 | Template bash script for inference runs |
| `process_scene_runner.py`| Runs the merging and thresholding operation for all scenes in parallel |
| `acm_postprocessing.py`| Creates the CSV mapping input paths with all cells mask output path for feature extraction |

---

* Feel free to modify any of the provided scripts to better fit your pipeline setup.
