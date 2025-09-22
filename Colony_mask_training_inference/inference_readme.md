
# ACM Generation and Postprocessing Steps

## Installation

1. Clone this git repository.

`git clone https://github.com/AllenCell/EMT_image_analysis.git`

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

---

## 2. Split CSV for Parallel Processing

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

## 4. Run Inference

A typical command looks like:

```bash
CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale1.yaml
```

Since multiple CSVs and YAMLs are involved, automate this using a bash script. See the provided `run.sh` template.

* **Edit the GPU UUID** and other arguments in the script.
* For multiple GPUs or MIG instances, duplicate and customize the runner script accordingly.
* Place `run.sh` in the `EMT_image_analysis/Colony_mask_training_inference/` directory, or modify internal paths if you’re running it from elsewhere.

---

## Summary of Scripts

| Script                   | Purpose                                 |
| ------------------------ | --------------------------------------- |
| `csv_creator.py`         | Generate base CSV from `.ome.zarr`      |
| `csv_splitter.py`    | Split large CSVs into chunks            |
| `generate_eval_yamls.py` | Generate inference YAMLs per chunk      |
| `run.sh`                 | Template bash script for inference runs |

---

* Refer to the [instructions](https://github.com/AllenCell/EMT_image_analysis/tree/sm_colony_mask_pred_test_v0/Colony_mask_training_inference#instructions-to-run-the-all-cells-maskacm-inference-generation-pipeline-on-linux-machines) for more details.

* Feel free to modify any of the provided scripts to better fit your pipeline setup.
