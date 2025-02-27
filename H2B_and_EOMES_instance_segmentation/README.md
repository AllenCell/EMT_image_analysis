
# Biofile Segmentation (BFS) Workflow Application

This repository contains a multi‑threaded, Streamlit‑based workflow pipeline for denoising, segmentation, and clipping of microscopy images. The workflow supports multiple file formats (e.g. OME‑Zarr, TIFF, CZI) and handles both single‑ and multi‑channel data. It features both dynamic intensity normalization/denoising or constant intensity normalization/denoising options, segmentation using the  cyto3 generalist Cellpose base model or optional pretrained models for either nuclei or cell segmentation targets, and adaptive post‑processing relative intensity pixel clipping of cellpose segmentation mask outputs. Processed results can be interactively visualized in Napari with the provided viewer launcher button at the bottom of the app.  Annotation of the segmentation mask outputs are also supported in the Napari viewer.

## Processing Pipeline Overview

The processing pipeline is organized into several steps:

1. **Input and Parameter Configuration**  
   - **User Interface:**  
     The Streamlit GUI lets you specify input and output directories, choose the processing mode (Denoising Only, Denoising + Segmentation, or Denoising + Segmentation + Clipping), and set detailed parameters for each processing step.
   - **Denoising Parameters:**  
     You specify which channel(s) to process via the “Denoising Channels” input (for example, `"1"` for nuclei segmentation or `"0,1"` for (v2 feature) label-free cell segmentation). if enabled, saves the raw offset‑corrected images in a dedicated “Raw” subfolder.  #v2 feature: When two channels are provided, the pipeline performs two separate denoising passes (saving outputs in separate subfolders)
   - **Segmentation Parameters:**  
     Choose the segmentation type (Nuclei, Cell, or Nuclei + Cell). Different pretrained model paths can be provided for nuclei and cell segmentation. Segmentation is then applied (using multi‑threading) on the denoised images.
   - **Clipping Parameters:**  
     An adaptive clipping algorithm refines the segmentation masks by removing low‑intensity pixels on a per‑slice basis. Additional options allow you to fill small holes and apply size/intensity filtering. A summary CSV file with object properties is also generated.

2. **Denoising**  
   - **Dynamic Rescaling:**  
     Raw images are loaded, a camera offset is subtracted, and intensities are rescaled based on measured raw minimum and maximum values.
   - **Multi‑Channel Handling:**  
     If a single channel is specified, one pass is run. If two channels are provided (e.g. `"0,1"`), two separate denoising passes are executed—with results saved into subfolders (e.g. `Denoised_ch0` and `Denoised_ch1`).
   - **Saving Raw Images:**  
     If enabled, the offset‑corrected raw images are saved in a “Raw” subfolder for reference.

3. **Segmentation**  
   - **Cellpose‑Based Segmentation:**  
     The denoised images are segmented using a Cellpose‑based model. Depending on your selected segmentation type, the appropriate set of denoised images (nuclei or cell) and pretrained model are used. Both 2D and 3D segmentation are supported.
  
4. **Clipping**  
   - **Adaptive Clipping:**  
     The segmentation masks are refined using an adaptive clipping algorithm that removes low‑intensity pixels on a per‑z‑slice basis. Additional post‑filtering (such as hole filling) is performed, and a CSV summary of object properties is generated.

5. **Visualization**  
   - **Napari Integration:**  
     Once processing is complete, you can launch a Napari viewer to load and inspect the results. Different processing outputs (Raw, Denoised, Segmentation, Clipped Masks) are loaded as separate layers.

## Generating a List of S3 Paths Using the AICS BFF Application

The pipeline can accept a CSV file containing a list of S3 paths for raw images. To generate this CSV file, you can use the AICS BFF application. The AICS BFF (available at https://bff.allencell.org) is a tool that allows you to query and list files stored in the Allen Institute’s S3 buckets. Follow these steps to generate your list:


1. **Generate Your CSV List:**

   Run the AICS BFF application with your desired query parameters (e.g., dataset, experiment, file type). Ensure that the CSV output includes a column (e.g. `"Raw Converted File Download"`) containing the S3 paths to the raw timelapse ome-zarr images.

2. **Use the CSV as Input for :**

   When launching the pipeline, select **"CSV Import"** as the parameter source and upload your CSV file. The pipeline will parse the S3 paths and use them for processing.

## Installation

Ensure you have Python 3.12 installed. Install required dependencies by running:
#*Note:* CUDA GPU REQUIRED, make sure the proper CUDA drivers are installed.
# pick GPU version of torch which matches you CUDA version (https://pytorch.org/ 11.8, 12.4, or 12.6)

BFS segmentation workflow app install instructions:

1. conda create --name bfs python=3.12

2. conda activate bfs

3. python -m pip install cellpose

4. pip uninstall torch #removes cpu version of torch which is installed by default

5. pip install torch --index-url https://download.pytorch.org/whl/cu126

6. pip install "cellpose[gui]"

7. run "cellpose --Zstack". You should see the following output:

(bfs) C:\Users\derek>cellpose --Zstack

2025-02-25 09:19:24,152 [INFO] WRITING LOG OUTPUT TO C:\Users\derek\.cellpose\run.log
2025-02-25 09:19:24,153 [INFO]
cellpose version:       3.1.1.1
platform:               win32
python version:         3.12.9
torch version:          2.6.0+cu126
2025-02-25 09:19:25,604 [INFO] ** TORCH CUDA version installed and working. **

8. pip install streamlit pandas numpy cupy-cuda12x scikit-image dask tqdm bioio==1.2.0 bioio-ome-zarr bioio-tifffile bioio-czi napari[all] matplotlib==3.10.0 tifffile==2025.1.10 PyOpenGL_accelerate

9. cd to the directory path containing the python script.

```
cd C:\path\to\scripts\BFS_segmentation_workflow
```

## Usage

Start the processing pipeline http streamlit server app the with:

```
streamlit run BFS_v1.py --server.fileWatcherType none
```

Then use the GUI to:

- Set the input/output paths (or upload a CSV of S3 paths created with AICS BFF).
- Choose the processing mode.
- Configure denoising (select channels, choose between dynamic rescaling or constant denoising, and decide whether to save raw images).
- Set segmentation parameters (select segmentation type and specify pretrained model paths for nuclei and/or cell segmentation).
- Configure clipping parameters.
- Launch processing and view results in Napari using the provided viewer launcher.

## Processing Workflow Summary

1. **Input/Parameter Setup:**  
   Configure paths, processing modes, and detailed parameters for each step. You may either enter all these parameters manually or import a preconfigured parameter csv to replicate a previous processing run output.

2. **Denoising:**  
   Raw images are loaded, offset‑corrected, and rescaled. (v2 feature) If multiple channels are specified, separate passes are run for each channel. Optionally, the raw images are saved.

3. **Segmentation:**  
   The denoised images are segmented using a Cellpose‑based model with configurable settings and pretrained models.

4. **Clipping:**  
   Segmentation masks are refined using an adaptive clipping algorithm with additional filtering and optional hole filling. Measured instance object properties are summarized in an output CSV.

5. **Visualization:**  
   Processed outputs can be loaded into Napari for interactive 3D visualization of the segmentation masks.  Copy the "Output Path" string from the output directory into the "Output Path for Napari" text box above the Napari viewer button.  Click Lauch Napari Viewer button to load the processed images.

## Reference

This code was developed by Derek Thirstrup. Contact Derek at derekt@alleninstitute.org for questions or comments.
