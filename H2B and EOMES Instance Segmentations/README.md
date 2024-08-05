**H2B and EOMES 3D Instance Segmentations**

To improve 3D instance segmentation accuracy a comprehensive processing pipeline we developed by leveraging multithreaded denoising, cellpose 3 cyto model segmentation, and object filtering. The pipeline was applied to H2B and Eomes cell lines, consisting of five main processing steps:

1. **Denoising**: A Cellpose3 DenoiseModel script was developed to denoise raw 3D image stacks in parallel. This script processes images concurrently, converting the float32 datatype from the DenoiseModel to uint16 before saving. Images were rescaled based on raw intensity values to ensure consistent denoising across the dataset.

2. **Segmentation**: The pretrained Cellpose 3 cyto segmentation model was applied to the denoised images. The model parameters, including channels, diameter, and thresholds, were configured to optimize detection sensitivity and 3D segmentation accuracy. The process was parallelized to handle large datasets efficiently, and segmentation masks were generated and saved for all denoised 3D image stacks.

3. **Object Filtering**: Utilized CellProfiler pipeline to convert segmented masks into objects, measure their intensity, and filter based on a minimum integrated intensity threshold value. Objects below the intensity threshold were removed to ensure only relevant nuclear structures were retained.

4. **Mask Clipping**: The script reads raw and mask images, clips intensity pixels within instance object masks based on a min relative intensity threshold, optionally relabels objects, and removes small disconnected components. This step ensures that the final masks represent well-defined, high-intensity objects.

5. **Post-Processing**: A final CellProfiler pipeline was used to filter objects based on integrated intensity after the clipping step. This pipeline also converts the filtered objects back into images, saving the final processed masks for downstream analysis.

This comprehensive processing pipeline effectively enhances 3D instance segmentation accuracy, producing reliable masks that capture nearly all detectable 3D nuclei in our H2B and Eomes cell lines.

# 3D Instance Segmentation Pipeline

This repository contains a comprehensive pipeline for 3D instance segmentation of H2B and Eomes cell lines. The pipeline includes image denoising, segmentation, and post-processing steps to ensure high-quality 3D segmentation masks. The following scripts are included and should be run in the specified order.

## 1. CP3DenoiseModel_RescalePerTimepoint.py

### Description
This script denoises 3D image stacks using the Cellpose3 DenoiseModel. It processes images in parallel, rescales intensity values, and saves the denoised images in uint16 format.

### Usage
```
python CP3DenoiseModel_Multithreaded_tqdm_GlobalPercentile.py --input_path="path/to/raw/images" --output_path="path/to/denoised/images" --max_workers=8
```

## 2. CP3DenoiseModel_ScaleValuesFromCSV_ConstantScalingOutput.py

### Description
This script processes images based on scaling values provided in a CSV file, ensuring consistent denoising across the dataset.

### Usage
```
python denoise_ScaleValuesFromCSV_ConstantDenoiseMinMaxScalingOutput.py --input_path="path/to/raw/images" --output_path="path/to/denoised/images" --max_workers=8
```

## 3. CellposeSeg_3D_Multithread_CSVOrInputFolder.py

### Description
This script segments 3D image stacks using the Cellpose model, employing multithreading to handle large datasets efficiently. Segmentation masks are saved for each image stack.

### Usage
```
python CellposeSeg_3D_Multithread_CSVOrInputFolder.py --input_path="path/to/denoised/images" --output_path="path/to/segmentation/masks" --workers=16
```

## 4. ClipMaskBasedOnRelativeIntensityZSD20x_gpu.py

### Description
This script clips intensity pixels within instance object masks based on a relative intensity threshold, optionally relabels objects, and removes small disconnected components.

### Usage
```
python ClipMaskBasedOnRelativeIntensityZSD20x_gpu.py --raw_dir="path/to/raw/images" --mask_dir="path/to/segmentation/masks" --output_dir="path/to/clipped/masks" --scaling_factor=0.75 --pre_clipping_min_size=1 --post_clipping_min_size=1 --pre_clipping_min_mean_intensity=1 --post_clipping_min_integrated_intensity=1 --workers=8 --relabel=False
```

## 5. H2B_EomesPrePostClippingIntegIntObjectFiltering.cppipe

### Description
This CellProfiler pipeline converts segmented masks into objects, measures their intensity, and filters based on integrated intensity values. Final processed masks are saved for downstream analysis.

### Usage
Load the `H2B_EomesPrePostClippingIntegIntObjectFiltering.cppipe` pipeline into CellProfiler and process the images as per the configuration.

## Package Dependencies
- numpy
- tifffile
- cellpose
- concurrent.futures
- fire
- tqdm
- pathlib
- glob
- cupy
- skimage
- pandas

Install the required dependencies using:
```
pip install numpy tifffile cellpose==3.0.8 fire tqdm pathlib glob cupy scikit-image pandas
```

For more info contact Derek Thirstrup (derekt@alleninstitute.org)
