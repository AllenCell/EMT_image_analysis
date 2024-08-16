# Instructions to run the all cells mask inference generation pipeline

  # Run all cells mask model trianing
    # Use the conda environment sandi-cytodl2 on any A100 machines
    # Go to the directory (use 'cd' command) "\\allen\aics\assay-dev\users\Sandi\cyto-dl"
    # To re-run the whole training run --> python cyto_dl/train.py experiment=local/train_cytoGFP_segmentation_model.yaml
  # Run multi-scale patch-based evaluation to generate probability maps
    # multi-scale patch-based evaluation runs on 3 different patch sizes to generate the prediction
    # 3 example movies are provided in "\\allen\aics\assay-dev\users\Sandi\cyto-dl\data\all_cells_mask_test_dir\predict_all_cells_mask_v0.csv" which will be used for testing
    # Use the conda environment sandi-cytodl2 on any A100 machines
    # Go to the directory (use 'cd' command) "\\allen\aics\assay-dev\users\Sandi\cyto-dl"
    # To run inference on patch1 run --> python cyto_dl/eval.py experiment=local/infer_segmentation_model_multiscale_patch1.yaml
    # To run inference on patch2 run --> python cyto_dl/eval.py experiment=local/infer_segmentation_model_multiscale_patch2.yaml
    # To run inference on patch3 run --> python cyto_dl/eval.py experiment=local/infer_segmentation_model_multiscale_patch3.yaml
    # Above commands will run the predictions on the checkpoint that has been already trained (/allen/aics/assay-dev/users/Sandi/cyto-dl/logs/train/runs/fijimasks_cytogfp_ALL_v3/v3/2024-03-23_14-33-36/checkpoints/epoch_426.ckpt)
    # To run on your own model, edit the yaml file ckpt_path with the path to your model.
    # Predictions for each patch based predictions will be storde at \\allen\aics\assay-dev\users\Sandi\cyto-dl\data\all_cells_mask_test_dir\eval_whole_movie_multiscale_patchX (where, X={1, 2, 3})
  # Run thresholding and merging script to generate the all cells mask
    # To run the thresholding and merging script, make sure your environment has aicsimageio and skimage installed
    # Run the command --> python ColonyMask_merging_thresholding_rearranging.py
    # The code expects patch-based probabilty masks has already been generated and are stored at /allen/aics/assay-dev/users/Sandi/cyto-dl/data/all_cells_mask_test_dir/eval_whole_movie_multiscale_patchX
    # The output will be binarized all cells masks and they can be accessed at /allen/aics/assay-dev/users/Sandi/cyto-dl/data/all_cells_mask_test_dir/eval_whole_movie_multiscale_patch2
  
    
