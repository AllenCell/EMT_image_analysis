# Instructions to run all cells mask (ACM) inference generation pipeline and postprocessing steps

## Installation

1. Clone this git repository and enter the inference directory.

   `git clone https://github.com/AllenCell/EMT_image_analysis.git`

   `cd Colony_mask_training_inference`

2. Install Python 3.10, either from [python.org](https://www.python.org/downloads/), your operating system package manager, or [pyenv](https://github.com/pyenv/pyenv-installer).
Check that it is installed correctly by running `python --version` in the terminal.
Then, you can use either `UV` or `PDM` to set up the Python environment using the provided requirements.txt file.

For UV, follow these instructions:
```bash
pipx install uv
uv venv acm
source acm/bin/activate
uv pip install -r requirements.txt
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

`cyto-dl` uses a `DataFrameLoader` that loads all entries from the input CSV. For large datasets, this leads to inefficient processing. To improve performance, we **split the CSV** into smaller chunks using the `csv_splitter.py` script. You can set the desired chunk size in the script.

---

**Step 4 - Generate YAML Configs for Inference**

multi-scale patch-based evaluation runs on 3 different patch sizes to generate the prediction. To run prediction on each patch, the evaluation config files (provided in configs/experiment/im2im/eval_scale1.yaml, configs/experiment/im2im/eval_scale2.yaml, and configs/experiment/im2im/eval_scale3.yaml) has to be modified.

As `cyto-dl` requires YAML config files for inference. You can start from a base YAML and generate customized versions for each data chunk using the `generate_eval_yamls.py` script. Here, we provide `template_scale1.yaml` as a base yaml file to generate your own yaml files.

* The `generate_eval_yamls.py` script allows **scale-specific YAML generation**.
* You can modify the script to suit your dataset or experiment needs.
* Ensure that each YAML correctly references:

  * The corresponding **CSV file**
  * The appropriate **model checkpoint path**

> Tip: Adjust the base YAML before generating the set, or manually edit the generated files if needed.

---

**Step 5 - Run Inference to generate outputs per scale**

To run inference on patch1 run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale1.yaml`

To run inference on patch2 run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale2.yaml`

To run inference on patch3 run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale3.yaml`

To run on your own model, edit the yaml file ckpt_path with the path to your model.

Predictions for each patch based predictions will be stored at the target location provided in save_dir. By default this location is `Colony_mask_training_inference/data/infer_movie_multiscale_patch1`, `Colony_mask_training_inference/data/infer_movie_multiscale_patch2`, and `Colony_mask_training_inference/data/infer_movie_multiscale_patch3` for different patches.

**Optional speedup for Step 5**

Since multiple CSVs and YAMLs are involved, automate this using a bash script. See the provided `run.sh` template.

* **Edit the GPU UUID** and other arguments in the script.
* For multiple GPUs or MIG instances, duplicate and customize the runner script accordingly.
* Place `run.sh` in the `EMT_image_analysis/Colony_mask_training_inference/` directory, or modify internal paths if you’re running it from elsewhere.

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
