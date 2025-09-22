
# ACM Generation and Postprocessing Steps

## Installation

1. Clone this git repository.

`git clone https://github.com/AllenCell/EMT_image_analysis.git`

`cd Colony_mask_training_inference`

2. You can use either `UV` or `PDM` to set up the Python environment using the provided requirements.txt file.

For UV, follow these instructions:

`pipx install uv`

`uv venv acm`

`source acm/bin/activate`

`uv pip install -r requirements.txt`

For PDM, follow these instructions:

`pipx install pdm`

`pdm init`

`pdm add -d -r requirements.txt`

## Overview

The following steps outline the workflow for generating All Cells Mask (ACM) outputs and postprocessing them using the `cyto-dl` pipeline.

---

## 1. Prepare CSV Files from `.ome.zarr` (or your data files)

Since `cyto-dl` expects inputs in CSV format, use the `csv_creator.py` script to generate the initial CSV file with path to your data files.

The CSV file should contain these 2 columns: movie_path and bf_channel. E.g., 

|count            |movie_path         |bf_channel        |
|-----------------|-------------------|------------------|
|0                |https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/data/3500006062_2_raw_converted.ome.zarr    |0                 |
|1                |https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/data/3500006062_4_raw_converted.ome.zarr    |0                 |

We recommend storing this CSV inside the `Colony_mask_training_inference/data` directory.

---

## 2. Split CSV for Parallel Processing (optional)

This step is optional and you can proceed with a single csv generated from step 1 if your compute allows for it.

`cyto-dl` uses a `DataFrameLoader` that loads all entries from the input CSV. For large datasets, this leads to inefficient processing. To improve performance, we **split the CSV** into smaller chunks using the `csv_splitter.py` script. You can set the desired chunk size in the script.

---

## 3. Generate YAML Configs for Inference

`cyto-dl` requires YAML config files for inference. You can start from a base YAML and generate customized versions for each data chunk using the `generate_eval_yamls.py` script.

* The script allows **scale-specific YAML generation**.
* You can modify the script to suit your dataset or experiment needs.
* Ensure that each YAML correctly references:

  * The corresponding **CSV file**
  * The appropriate **model checkpoint path**

> Tip: Adjust the base YAML before generating the set, or manually edit the generated files if needed.

---

## 4. Run Inference to generate outputs per scale

A typical command looks like:

```bash
CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale1.yaml
```

Since multiple CSVs and YAMLs are involved, automate this using a bash script. See the provided `run.sh` template.

* **Edit the GPU UUID** and other arguments in the script.
* For multiple GPUs or MIG instances, duplicate and customize the runner script accordingly.
* Place `run.sh` in the `EMT_image_analysis/Colony_mask_training_inference/` directory, or modify internal paths if you’re running it from elsewhere.

## 5. Postprocessing to merge outputs and generate all cells mask

Run the command --> `process_scene_runner.py`

Step 4 generates outputs per scale and this script is used to merge the scales and obtain the final all cells mask. The code expects patch-based probabilty masks have already been generated and were stored at `/pathto/Colony_mask_training_inference/data/all_cells_mask_test_dir/eval_whole_movie_multiscale_patchX` (where, X={1, 2, 3})

The output of this script will be the binarized all cells masks and they can be accessed at `/pathto/Colony_mask_training_inference/data/all_cells_mask_test_dir/multiscale_all_cells_mask`.

## 6. CSV creation for feature extraction step

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

* Refer to the [instructions](https://github.com/AllenCell/EMT_image_analysis/tree/sm_colony_mask_pred_test_v0/Colony_mask_training_inference#instructions-to-run-the-all-cells-maskacm-inference-generation-pipeline-on-linux-machines) for more details.

* Feel free to modify any of the provided scripts to better fit your pipeline setup.
