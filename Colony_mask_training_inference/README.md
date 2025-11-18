# Instructions to run the all cells mask(ACM) inference generation pipeline on Linux machines

## Installation
1. Clone this git repository.  
   `git clone https://github.com/AllenCell/EMT_image_analysis.git`
2. Go inside the directory and switch to sm_colony_mask_pred_test_v0 branch.  
   `cd EMT_image_analysis`  
   `git checkout origin/sm_colony_mask_pred_test_v0`
3. Install Python 3.10, either from [python.org](https://www.python.org/downloads/), your operating system package manager, or [pyenv](https://github.com/pyenv/pyenv-installer).
Check that it is installed correctly by running `python --version` in the terminal.
Then, use the following steps to create a new virtual environment and install the dependencies.
   ```bash
   cd Colony_mask_training_inference
   python -m venv .venv
   source .venv/bin/activate
   pip install .
   ```
4. Alternatively use Conda package manager to create a virtual environment with python 3.10
   ```
   conda create -n emt-acm-env python=3.10
   conda activate emt-acm-env
   cd Colony_mask_training_inference
   pip install .
   pip install bioio-ome-zarr
   ```



## Run all cells mask model trianing [Under Development - ZARR support not yet implemented]
Users are welcome to retrain the models by accessing all the images provided in https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/

To re-run the whole training run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.train experiment=im2im/train.yaml`

This part is not yet fully supported as CytoDL training using OME ZARR files is still under development. For retraining, users can use tiff files instead.


## Run multi-scale patch-based evaluation to generate probability maps

**Step 1 -  Download the model checkpoint**  
The model checkpoint path is required to generate the ACM. A pretrained model is provided and can be downloaded from this link -  https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/supplemental_files/cytodl_checkpoints/all_cells_mask_seg_model_checkpoint.ckpt 
Access the link and click on the "DOWNLOAD FILE" button [top left].  

Alternatively, user can use `curl` to download using the link and the command - https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/supplemental_files/cytodl_checkpoints/all_cells_mask_seg_model_checkpoint.ckpt?versionId=ejf07rBiw5slyx1zQyurfX6.zpSJ92JM

`curl -o all_cells_mask_seg_model_checkpoint.ckpt https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/supplemental_files/cytodl_checkpoints/all_cells_mask_seg_model_checkpoint.ckpt?versionId=ejf07rBiw5slyx1zQyurfX6.zpSJ92JM`

Create a new sub-directory using the command `mkdir data` inside `Colony_mask_training_inference` directory.

Save the downloaded checkpoint file in `EMT_image_analysis/Colony_mask_training_inference/data/`  
Users are welcome to train their own models or finetune the existing model with their own data.  



**Step 2 - Prepare data on which the ACM prediction has to be performed**  
Data (3D Z-stack of a single timepoint or a timelapse) is provided as an input to the model to predict its all-cells-mask. Data is provided as a CSV file. The CSV file should contain these 2 columns: movie_path and bf_channel. E.g., 

|count            |movie_path         |bf_channel        |
|-----------------|-------------------|------------------|
|0                |https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/data/3500006062_2_raw_converted.ome.zarr    |0                 |
|1                |https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/data/3500006062_4_raw_converted.ome.zarr    |0                 |

We recommend storing this CSV inside the `Colony_mask_training_inference/data` directory

Users are welcome to experiment using this table and save it as a CSV file named  `predict_all_cells_mask_zarr_aws_v0.csv` inside the `Colony_mask_training_inference/data` directory.


**Step 3 - Edit the evaluation conifg file**  
multi-scale patch-based evaluation runs on 3 different patch sizes to generate the prediction. To run prediction on each patch, the evaluation config files (provided in configs/experiment/im2im/eval_sacle1.yaml, configs/experiment/im2im/eval_sacle2.yaml, and configs/experiment/im2im/eval_sacle3.yaml) has to be modified.

To run inference on patch1 run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale1.yaml`

To run inference on patch2 run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale2.yaml`

To run inference on patch3 run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale3.yaml`

To run on your own model, edit the yaml file ckpt_path with the path to your model.

Predictions for each patch based predictions will be stored at the target location provided in save_dir. By default this location is `Colony_mask_training_inference/data/infer_movie_multiscale_patch1`, `Colony_mask_training_inference/data/infer_movie_multiscale_patch2`, and `Colony_mask_training_inference/data/infer_movie_multiscale_patch3` for for different patches.

## Run thresholding and merging script to generate the all cells mask
To run the thresholding and merging script, make sure your environment has aicsimageio and skimage installed

Run the command --> `python ColonyMask_merging_thresholding_rearranging.py`

The code expects patch-based probabilty masks have already been generated and were stored at `/pathto/Colony_mask_training_inference/data/all_cells_mask_test_dir/eval_whole_movie_multiscale_patchX` (where, X={1, 2, 3})

The output of this script will be the binarized all cells masks and they can be accessed at `/pathto/Colony_mask_training_inference/data/all_cells_mask_test_dir/multiscale_all_cells_mask_v0`
