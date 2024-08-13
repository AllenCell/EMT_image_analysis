The workflow for generating basement membrane is divided into three parts.

# Part 1: Create snakemake configs and split out basement membrane channel and timepoints and 

This is done in the slurm cluster using Snakemake. Make sure you use the envirorment goutham_snakemake for running Snakemake scripts.

```
module load anaconda3/5.3.0
source activate
conda activate goutham_snakemake
cd processing_workflow
# Creates logs for snakemake. These same logs are also initated to run the basement membrane postprocessing
python create_BM_snakemake_configs.py

# Run snakemake file for splitting out each timepoint
cd processing_workflow/step_save_out_basement_membrane
# See configs available in the configs directory
snakemake --profile ../configs/profile --conda-frontend conda --printshellcmds --configfile ../configs/your_config.yaml
```

# Part 2: CytoDL basement membrane segmentation

The data, model, and experiment config are found in the cytoDL_configs directory. The model weights that we are using for the segmentation model can be found here:

```
/allen/aics/assay-dev/computational/data/EMT_deliverable_processing/cytodl_experiments/logs/train/runs/basement_membrane_semseg/basement_membrane_semseg_version_6_early_model/2023-12-15_16-09-07/checkpoints/epoch_478.ckpt
```

# Part 3: Basement membrane postprocessing

This step processes the basement membrane segmentation to keep only the largest connected component in the prediction. This is currently done using Snakemake to do this in parrallel for each FMS ID. Please modify the config file according to your own system paths. The profile refers to your compute recourses. This is specific to our slurm cluster.

```
# Basement membrane postprocessing example
cd step_postprocess_basement_membrane_mask/

# For EOMES
snakemake --profile ../configs/profile --conda-frontend conda --printshellcmds --configfile ../configs/config_basement_membrane_segmentation_EOMES.yaml

# For H2B
snakemake --profile ../configs/profile --conda-frontend conda --printshellcmds --configfile ../configs/config_basement_membrane_segmentation_H2B.yaml
```