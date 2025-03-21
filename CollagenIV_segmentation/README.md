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
| end | The timepoint at which segmentations wills stop. (default end of scene) |
| step | The interval between timepoints segmented (default 1) |


# Part 2: CytoDL basement membrane segmentation

The template data, model, and experiment config are found in the `cytoDL_configs` 
directory. The model weights that we are using for the segmentation model can be 
downloaded through curl into the directory of your choice.


```bash
cd /directory/to/save/weights/

curl -O https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/supplemental_files/cytodl_checkpoints/collagenIV_mask_seg_model_checkpoint.ckpt
```

Make sure to change `ckpt_path` in `cytoDL_configs/experiment/segmentation_basement_membrane.yaml`
has be changed to your loacally saved model weights.

# Part 3: Basement membrane postprocessing

This step processes the basement membrane segmentation to keep only the largest connected component in the prediction. This is computed using `postprocess_collagen_mask.py`

```bash
# Basement membrane postprocessing example
cd processing_workflow/

python postprocess_collagen_mask.py \
    -s /path/to/cytodl/output/directory/ \
    -o /directory/to/save/postprocessing/results/to/
```

# Part 4 (Optional): Compiling Segmentations for Mesh Generation

If you want to run the mesh generation workflow on newly generated scripts
they will have to be compiled into single `tif` files per scene and an 
accompanying `csv` manifest generated. To do that you can run the following 
script. Within the specified output folder will be saved both the compiled
segmentations as well as a csv which you can use as an input in the [mesh generation
workflow](../CollagenIV_mesh_generation/README.md)