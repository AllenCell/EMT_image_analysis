There are three parts to this workflow


# Part 1: Split out basement membrane channel and timepoints and create snakemake configs

This is done using the slurm cluster using Snakemake:
```
# use environment goutham_snakemake on slurm for all snakemake related jobs

# creates logs for snakemake - These same logs are also initated to run the basement membrane postprocessing
python create_BM_snakemake_configs.py 

# run snakemake file for splitting out each timepoint
## This step is optional, you can also use cytoDL to load the timepoints directly
cd processing_workflow/step_save_out_basement_membrane
snakemake --profile ../configs/profile --conda-frontend conda --printshellcmds --configfile ../configs/your_config.yaml
```

# Part 2: CytoDL Basement Membrane Segmentation

The data, model, and experiment config are found in the cytoDL_configs directory
The model weights we are using for the segmentation model is found here:
```
/allen/aics/assay-dev/computational/data/EMT_deliverable_processing/cytodl_experiments/logs/train/runs/basement_membrane_semseg/basement_membrane_semseg_version_6_early_model/2023-12-15_16-09-07/checkpoints/epoch_478.ckpt
```

# Part 3: Basement Membrane Postprocessing

This step processes the basement membrane segmentation to keep only the largest connected component in the prediction. This is currently done using Snakemake where we do this in parrallel for each fms id. 
Please modify the configfile according to your own system paths!
The profile refers to your compute recourses. This is specific to our slurm cluster.
```
# Basement Membrane postprocessing example
cd step_postprocess_basement_membrane_mask/

#For EOMES
snakemake --profile ../configs/profile --conda-frontend conda --printshellcmds --configfile ../configs/config_basement_membrane_segmentation_EOMES.yaml

#For H2B
snakemake --profile ../configs/profile --conda-frontend conda --printshellcmds --configfile ../configs/config_basement_membrane_segmentation_H2B.yaml

```


















