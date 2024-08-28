# Instructions to run the all cells mask inference generation pipeline

## CytoDL installation 
[Install CytoDL](https://github.com/AllenCellModeling/cyto-dl)

## Config file downloading
Download the configuration files available in `data`, `experiment`, and `model` directories and place them inside the respective sub-directories in `cyto-dl/configs/[***]/im2im/`, where [***] is either the `data`, the `experiment`, or the `model` directory

## Run all cells mask model trianing
Users are welcome to retrain the models by accessing all the images provided in https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/

To re-run the whole training run --> `python cyto_dl/train.py experiment=im2im/train.yaml`

## Run multi-scale patch-based evaluation to generate probability maps
multi-scale patch-based evaluation runs on 3 different patch sizes to generate the prediction

The pretrained checkpoint can be downloaded at https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/supplemental_files/cytodl_checkpoints/all_cells_mask_seg_model_checkpoint.ckpt

To run inference on patch1 run --> `python cyto_dl/eval.py experiment=im2im/eval_scale1.yaml`

To run inference on patch2 run --> `python cyto_dl/eval.py experiment=im2im/eval_scale2.yaml`

To run inference on patch3 run --> `python cyto_dl/eval.py experiment=im2im/eval_scale3.yaml`

To run on your own model, edit the yaml file ckpt_path with the path to your model.

Predictions for each patch based predictions will be stored at `/pathto/cyto-dl/data/all_cells_mask_test_dir/eval_whole_movie_multiscale_patchX` (where, X={1, 2, 3})

## Run thresholding and merging script to generate the all cells mask
To run the thresholding and merging script, make sure your environment has aicsimageio and skimage installed

Run the command --> `python ColonyMask_merging_thresholding_rearranging.py`

The code expects patch-based probabilty masks has already been generated and are stored at `/pathto/cyto-dl/data/all_cells_mask_test_dir/eval_whole_movie_multiscale_patchX` (where, X={1, 2, 3})

The output will be binarized all cells masks and they can be accessed at `/pathto/cyto-dl/data/all_cells_mask_test_dir/multiscale_all_cells_mask_v0`
