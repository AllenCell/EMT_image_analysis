# Bio File Segmentation (BFS) Workflow Application

This repository contains a multi‑threaded, Streamlit‑based workflow pipeline for denoising, segmentation, and clipping of microscopy images used in the paper "A human induced pluripotent stem (hiPS) cell model for the holistic study of epithelial to mesenchymal transitions (EMTs)". The current BFS_v1 workflow supports OME‑Zarr file formats. It features both dynamic intensity normalization/denoising or constant intensity normalization/denoising options, segmentation using the cyto3 generalist Cellpose model or optional included pretrained model for generating instance nuclei segmentations, and adaptive post‑processing relative intensity pixel clipping of cellpose segmentation mask outputs. Processed results can be interactively visualized in Napari with the provided viewer launcher button at the bottom of the app.  Annotation of the segmentation mask outputs are also supported in the Napari viewer.

### Clone the Repository to Get Started

For the specific version used in the paper, you can clone this particular branch:

```
git clone -b instance_segmentation_workflow_gui https://github.com/AllenCell/EMT_image_analysis.git
```

```
cd path\to\cloned\repo\EMT_image_analysis\H2B_and_EOMES_instance_segmentation
```




### Data Availability

The raw data used in the paper is available through the Allen Cell BFF application. Generate a CSV file of S3 paths as described in the "Generating a List of S3 Paths Using the AICS BFF Application" section.

## Generating a List of S3 Paths Using the AICS BFF Application

The pipeline can accept a CSV file containing a list of S3 paths for raw images. To generate this CSV file, you can use the AICS BFF application. The AICS BFF (available at https://bff.allencell.org) is a tool that allows you to query and list files stored in the Allen Institute's S3 buckets. Follow these steps to generate your list:


1. **Generate Your CSV List:**

   Browse to the [Allen Cell BFF H2B Dataset](https://bff.allencell.org/app?c=Movie+ID%3A0.25%2CExperimental+Condition%3A0.25%2CGene%3A0.25%2CCell+Line%3A0.25&group=Gene&openFolder=%5B%22HIST1H2BJ%22%5D&source=%7B%22name%22%3A%22imaging_and_segmentation_data.csv+%288%2F15%2F2024+4%3A26%3A03+PM%29%22%2C%22type%22%3A%22csv%22%2C%22uri%22%3A%22https%3A%2F%2Fallencell.s3.amazonaws.com%2Faics%2Femt_timelapse_dataset%2Fmanifests%2Fimaging_and_segmentation_data.csv%3FversionId%3DWmTjARBNL4rNJhV4N7YFYr2dKHWlCHwc%22%7D&sourceMetadata=%7B%22name%22%3A%22Imaging_and_segmentation_data_column_description.csv+%288%2F15%2F2024+4%3A26%3A02+PM%29%22%2C%22type%22%3A%22csv%22%2C%22uri%22%3A%22https%3A%2F%2Fallencell.s3.amazonaws.com%2Faics%2Femt_timelapse_dataset%2Fmanifests%2FImaging_and_segmentation_data_column_description.csv%3FversionId%3D.bmbr.UUT06F9nupeuwxVBYuTMyKyYu6%22%7D) 
    
    Ensure that the CSV output includes a column (e.g. `"Raw Converted File Download"`) containing the S3 paths to the raw timelapse ome-zarr images.

2. **Use the CSV as Input for :**

   When launching the pipeline, select **"CSV Import"** as the parameter source and upload your CSV file. The pipeline will parse the S3 paths and use them for processing.

## Repository Structure

This repository is part of a larger codebase for EMT image analysis. The structure relevant to this workflow is:

```
EMT_image_analysis/
├── H2B_and_EOMES_instance_segmentation/                                # Github repo directory
│   ├── BFS_v1.py                                                       # Main application script
│   ├── README.md                                                       # This file
│   └── processing_parameters_used_for_paper.csv                        # Parameter file configured with settings used for paper
├── H2B_and_EOMES_instance_segmentation/models/                         # Github repo models directory
│   ├── cellpose_1733942372.5497868                                     # high performance generalist cyto3 model trained on intensity rescaled and denoised H2B 40xWI 3D dataset

```

## Installation

Ensure you have Python 3.12 installed. Install required dependencies by running:
#*Note:* CUDA GPU REQUIRED, make sure the proper CUDA drivers are installed.  Pick GPU version of torch which matches you CUDA version (https://pytorch.org/ 11.8, 12.4, or 12.6)

BFS segmentation workflow app install instructions:

1. conda create --name bfs python=3.12

2. conda activate bfs

3. python -m pip install cellpose

4. pip uninstall torch #removes cpu version of torch which is installed by default

5. pip install torch --index-url https://download.pytorch.org/whl/cu126

6. pip install "cellpose[gui]"

7. run "cellpose --Zstack". You should see the following output:

```
(bfs) C:\Users\derek>cellpose --Zstack

2025-02-25 09:19:24,152 [INFO] WRITING LOG OUTPUT TO C:\Users\derek\.cellpose\run.log
2025-02-25 09:19:24,153 [INFO]
cellpose version:       3.1.1.1
platform:               win32
python version:         3.12.9
torch version:          2.6.0+cu126
2025-02-25 09:19:25,604 [INFO] ** TORCH CUDA version installed and working. **
```

8. Install python package dependencies

'''pip install streamlit pandas numpy cupy-cuda12x scikit-image dask tqdm bioio==1.2.0 bioio-ome-zarr bioio-tifffile bioio-czi napari[all] matplotlib==3.10.0 tifffile==2025.1.10 PyOpenGL_accelerate
'''

9. cd to the directory path containing the python script.

```
cd C:\path\to\scripts\BFS_segmentation_workflow
```

## Usage

Start the processing pipeline http streamlit server app the with:

```
streamlit run BFS_v1.py --server.fileWatcherType none
```

![image](https://github.com/user-attachments/assets/e1e81325-03dd-4960-9f97-6f154b680332)


You can use the GUI to:

- The Parameter Source option at top allows for importing of a processing settings csv which is generated with every processing run so the same pipeline workflow settings can be replicated with different processing batches. 
- Set the input/output paths (or upload a CSV of S3 paths created with AICS BFF).
- Choose the processing mode.
- Configure denoising (select channels, choose between dynamic rescaling or constant denoising, and decide whether to save raw images).
- Set segmentation parameters (select segmentation type and specify pretrained model paths for nuclei and/or cell segmentation). Use either the "cyto3" model to reproduce paper segmentations or H2B trained model "cellpose_1733942372.5497868" file from the repo for the H2B nuclei raw data s3 paths from BFF appplication.
- Configure clipping parameters.
- Launch processing and view results in Napari using the provided viewer launcher.

## Reproducing Paper Results

To reproduce the results presented in our paper "A human induced pluripotent stem (hiPS) cell model for the holistic study of epithelial to mesenchymal transitions (EMTs)", follow these steps:


1. Select "CSV Inport" option to load the parameter file from the cloned repo directory:
   ```
   processing_parameters_used_for_paper.csv
   ```
    Can either browse to folder path for cloned repo or drag file to Upload Parameter CSV section to load processing parameters used for paper.  

2. Change Output Path to folder that you want to save the workflow processing results. 
    Scroll down to Clipping parameters section and confirm that base path is same specified in Output Path field.  This was built with a free form text box so that cliping step can be run on any previously processed datasets without having to run all steps over again. Ensure both checkboxes "Use existing denoised images for segmentation" and "Use existing segmentation mask images for clipping" are checked if only tuning of the clipping step is desired.

3. Run the BFS workflow application by clicking "Start Processing" to run the workflow with the exact parameters used in the paper.

The parameter file contains all settings needed to reproduce the segmentation results used in the publication "A human induced pluripotent stem (hiPS) cell model for the holistic study of epithelial to mesenchymal transitions (EMTs)".


## Troubleshooting

### Common Issues

1. **CUDA errors**: Ensure your NVIDIA drivers match the CUDA version you've installed with PyTorch
2. **Memory issues**: For large datasets, try processing fewer files at once or reduce the batch size parameter
3. **Import errors**: Verify all dependencies are installed correctly with the specified versions
4. **Output paths**: Verify output path is set to correctly and that clipping paths match specified output base path
5. **Excel issues**: "Timepoint Range" field is set to 1-2 so it processing first two timepoints and will get changed automatically to date datatype by excel so open in text editor to make configuration changes to the processing_parameters.csv

## Processing Pipeline Overview

The processing pipeline is organized into several steps:

1. **Input and Parameter Configuration**  
   - **User Interface:**  
     The Streamlit GUI lets you specify input and output directories, choose the processing mode (Denoising Only, Denoising + Segmentation, or Denoising + Segmentation + Clipping), and set detailed parameters for each processing step.  When setting up new workflow configure paths, processing modes, and parameters determined to be optimal for new datasets.  You may either enter all these parameters manually or import a preconfigured parameter csv to replicate a previous processing run output.
   - **Denoising Parameters:**  
     You specify which channel(s) to process via the "Denoising Channels" input (for example, `"1"` for H2B-EGFP nuclei segmentation). If "Save Raw Images" is checked it saves the raw offset‑corrected images in a dedicated "Raw" subfolder.  
   - **Segmentation Parameters:**  
     (v2 feature) Choose the segmentation type (Nuclei, Cell, or Nuclei + Cell). Different pretrained model paths can be provided for nuclei and cell segmentation. Segmentation is then applied (using multi‑threading) on the denoised images.
   - **Clipping Parameters:**  
     An adaptive clipping algorithm refines the segmentation masks by removing low‑intensity pixels on a per‑slice basis. Additional options allow you to fill small holes and apply size/intensity filtering. A summary CSV file with object properties is also generated.

2. **Denoising**  
   - **Dynamic or Constant Rescaling Options:**  
     Raw images are loaded, a constant camera offset is subtracted, and intensities are rescaled based on measured raw minimum and maximum values for dynamic mode or constant predetermined rescale values for the constant rescaling method.  This paper used constant rescale method.  The default parameters are set to same values used for segmentations using in the paper.

   - **Saving Raw Images:**  
     If enabled, the offset‑corrected raw images are saved in a "Raw" subfolder for reference.

3. **Segmentation**  
   - **Cellpose‑Based Segmentation:**  
     The denoised images were segmented using the Cellpose3 "cyto3" generalist model for this paper. The trained model in the model folder displays higher segmentation accuracy than default cyto3 model used to generate the paper segmentations. To use the "cellpose_1733942372.5497868" model to generate segmentations copy model file local path into the "Segmentation Pretrained Model" field in the GUI.
  
4. **Clipping**  
   - **Adaptive Clipping:**  
     The instance segmentation masks generated by cellpose are passed to an adaptive clipping algorithm developed to refine masks generated from 3D volumes with z sampling interval greater than optimal Nyquist sampling. This step removes low‑intensity pixels within the masks which fall below the specified relative per object XY and Z intensity thresholds. Additional optional intensity and object size post‑filtering and z slice wise hole filling can be optionally performed and a CSV summary of measured object properties is generate at the end of the workflow run.  When tuning clipping step turn off "Fill holes" checkbox since it can take awhile and is only need with high clipping thresholds so turn on only if holes are observed in the clipping segmentation mask outputs. 

5. **Visualization**  
   - **Napari Integration:**  
     Once processing is complete, you can launch a Napari viewer to load and inspect the segmentation results. Different processing outputs (Raw, Denoised, Segmentation, Clipped Masks) are loaded as separate layers for interactive 3D visualization and annotation of the segmentation masks.  Copy the "Output Path" string from the output directory into the "Output Path for Napari" text box above the Napari viewer button.  Click "Launch Napari Viewer button (all subfolders)" button to load the processed images.  Select any position that was processed using position drop down selection in lower left corner and click "Load Position" button. The Shuffle Colors button has to be clicked once before masks will display for Napari label layers. To annotate mask objects fill in annotator name first which is required to save annotation csv.  Select channel layer to be annotated and press 5 key which is shortcut to pick mode napari function.  To pan around the shortcut is 6 so toggle between 5 and 6 for viewing masks at different angles. Show selected checkbox allows viewing single masked objects.  The object id for that mask will be displayed in annotator object id text box. Make annoatation and click Record Annotation to store that annotation.  Once done annotation all objects click "Save Annoatations to CSV" button to save a record of all annotations to the output path specified in the GUI.


## Reference

This code was developed by Derek Thirstrup. Contact Derek at derekt@alleninstitute.org for questions or comments. 