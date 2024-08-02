There are three parts to this workflow


Part 1: Split out basement membrane channel and timepoints

This is done using the slurm cluster using Snakemake:



```
# creates logs for snakemake
python create_BM_snakemake_configs.py
cd /basement_membrane_segmentation/processing_workflow/step_save_out_basement_membrane

# run snakemake file for splitting out each timepoint
snakemake --profile ../configs/profile --conda-frontend conda --printshellcmds --configfile ../configs/your_config.yaml
```
# This is optional ---- You can also read directly the channel/timepoint using CytoDL


Part 2: CytoDL Basement Membrane Segmentation

The data, model, and experiment config are found in the cytoDL_configs directory

The model weights we are using for the segmentation model is found here:
/allen/aics/assay-dev/computational/data/EMT_deliverable_processing/cytodl_experiments/logs/train/runs/basement_membrane_semseg/basement_membrane_semseg_version_6_early_model/2023-12-15_16-09-07/checkpoints/epoch_478.ckpt

# Maybe move this to an S3 bucket?
















