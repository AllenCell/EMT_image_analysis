# Instructions to run the all cells mask(ACM) inference generation pipeline

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
cd EMT_image_analysis/Colony_mask_training_inference
python -m venv .venv
source .venv/bin/activate
pip install .
```
4. Alternatively use Conda package manager to create a virtual environment with python 3.10
   ```
   conda create -n emt-acm-env python=3.10
   conda activate emt-acm-env
   pip install .
   ```



## Run all cells mask model trianing [Under Development - ZARR support not yet implemented]
Users are welcome to retrain the models by accessing all the images provided in https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/

To re-run the whole training run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.train experiment=im2im/train.yaml`



## Run multi-scale patch-based evaluation to generate probability maps

**Step 1 -  Download the model checkpoint**  
The model checkpoint path is required to generate the ACM. A pretrained model is provided and can be downloaded from  https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/supplemental_files/cytodl_checkpoints/all_cells_mask_seg_model_checkpoint.ckpt  
Save the downloaded checkpoint file in `EMT_image_analysis/Colony_mask_training_inference/`  
Users are welcome to train their own models or finetune the existing model with their own data.  



**Step 2 - Prepare data on which the ACM prediction has to be performed**  
Data (3D Z-stack of a single timepoint or a timelapse) is provided as an input to the model to predict its all-cells-mask. Data is provided as a CSV file. The CSV file should contain 2 columns: movie_path and bf_channel. E.g., 

|count            |movie_path         |bf_channel        |
|-----------------|-------------------|------------------|
|0                |\path\to\movie1    |0                 |
|1                |\apth\to\movie2    |0                 |

This CSV has to be stored inside a directory. We recommend creating a directory named `test_data` inside the `Colony_mask_training_inference` directory using the command `mkdir test_data`

A sample CSV containing few example timelapse movies used in this work is provided Here --> `/allen/aics/assay-dev/users/Suraj/EMT_Work/image_analysis_test/EMT_image_analysis/Colony_mask_training_inference/sample_csv/predict_all_cells_mask_v0.csv`

---> ToDo: Provide an example CSV in AWS
 Keep a copy of the provided CSV file using the command --> `cp /allen/aics/assay-dev/users/Suraj/EMT_Work/image_analysis_test/EMT_image_analysis/Colony_mask_training_inference/sample_csv/predict_all_cells_mask_v0.csv test_data/`


**Step 3 - Edit the evaluation conifg file**  
multi-scale patch-based evaluation runs on 3 different patch sizes to generate the prediction. To run prediction on each patch, the evaluation config files (provided in configs/experiment/im2im/eval_sacle1.yaml, configs/experiment/im2im/eval_sacle2.yaml, and configs/experiment/im2im/eval_sacle3.yaml) has to be modified.

_a. in line 23, ckpt_path: insert the path of the downloaded all cells mask model checkpoint_  
_b. in line 30, save_dir: insert the target location where the putput will be stored_  
_c. in line 33, csv_path: provide the path to CSV file from Step 2_

Similarly edit the eval_sacle2.yaml and eval_scale3.yaml files before running the inference. 

To run inference on patch1 run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale1.yaml`

To run inference on patch2 run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale2.yaml`

To run inference on patch3 run --> `CYTODL_CONFIG_PATH=$PWD/configs python -m cyto_dl.eval experiment=im2im/eval_scale3.yaml`

To run on your own model, edit the yaml file ckpt_path with the path to your model.

Predictions for each patch based predictions will be stored at the target location provided in save_dir. We recommend using the `test_data` directory to save the predictions, e.g., directory `multiscale_patch1`, `multiscale_patch2`, and `multiscale_patch3` can be used to save predictions for different patches.

## Run thresholding and merging script to generate the all cells mask
To run the thresholding and merging script, make sure your environment has aicsimageio and skimage installed

Run the command --> `python ColonyMask_merging_thresholding_rearranging.py`

The code expects patch-based probabilty masks has already been generated and are stored at `/pathto/cyto-dl/data/all_cells_mask_test_dir/eval_whole_movie_multiscale_patchX` (where, X={1, 2, 3})

The output will be binarized all cells masks and they can be accessed at `/pathto/cyto-dl/data/all_cells_mask_test_dir/multiscale_all_cells_mask_v0`
