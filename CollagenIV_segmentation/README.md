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
The csv's generated for individual czi files can be concatenated to segment
all czi's at once.

The csv file generated will have the columns, with one row for each scene that
is being segmented.

| Column | Description |
| --- | --- |
| path | File path to the czi file containing the scene |
| channel | Channel containing Collagen IV signal (default 2) |
| scene | Scene name as found in the czi file using `BioImage(path).scenes` |
| start | The timepoint from which segmentations will start. (default 0) |
| end | The timepoint at which segmentations wills stop. (default length of scene) |
| step | The interval between timepoints segmented (default 1) |


# Part 2: CytoDL basement membrane segmentation

The data, model, and experiment config are found in the `cytoDL_configs` directory. The model weights that we are using for the segmentation model can be found here:

```bash
# ToDo: provide instructions on getting model weights from quilt
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