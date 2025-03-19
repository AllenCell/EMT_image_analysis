The workflow for generating basement membrane is divided into three parts.

# Part 1: Create csv file for running segmentations

This is done using `prepare_cytodl_csv.py`. It's core function, `generate_csv()`
can be used programaticlly to process multiple czi files together, but if only
processing a single czi file then the script can be run from the terminal as such.

```bash
# This will prepare the csv using default parameters for scene, channel, and timepoints
python prepare_cytodl_csv.py \
    --czi /path/to/czi/file \
    --output_dir /path/to/save/csv


# For information on additional options to manually specify scenes, channel, and timepoints use
python prepare_cytodl_csv.py --help
```

# Part 2: CytoDL basement membrane segmentation

The data, model, and experiment config are found in the `cytoDL_configs` directory. The model weights that we are using for the segmentation model can be found here:

```bash
/allen/aics/assay-dev/computational/data/EMT_deliverable_processing/cytodl_experiments/logs/train/runs/basement_membrane_semseg/basement_membrane_semseg_version_6_early_model/2023-12-15_16-09-07/checkpoints/epoch_478.ckpt
```

# Part 3: Basement membrane postprocessing

This step processes the basement membrane segmentation to keep only the largest connected component in the prediction. This is computed using `postprocess_collagen_mask.py`

```bash
# Basement membrane postprocessing example
cd processing_workflow/

python postprocess_collagen_mask.py \
    -s /path/to/cytodl/output/directory/ \
    -o /directory/to/save/postprocessing/results/to/
```