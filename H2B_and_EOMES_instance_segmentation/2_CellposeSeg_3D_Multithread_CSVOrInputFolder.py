'''
The script facilitates batch processing of 3D image stacks for segmentation, leveraging the Cellpose deep learning model. It's particularly optimized for handling large datasets by employing multithreading to perform segmentation tasks in parallel. The script integrates the fire library to enable command-line argument passing for easy customization of input and output paths and the number of worker threads.

Detailed Steps and Function Descriptions

1. Segment a Single Image (segment_image function):
Purpose: Performs segmentation on a single image using specified Cellpose model parameters.

Parameters:
    model: The Cellpose model object configured for segmentation.
    img_path: Path to the input image file.
    output_dir: Directory where the segmentation masks are saved.
    
Other parameters like channels, diameter, flow_threshold, etc., are Cellpose-specific options that control the segmentation behavior.

Process:

Reads the image from img_path using tifffile.imread.
Invokes the Cellpose model's eval method on the image with the provided parameters to perform segmentation.

Saves the resulting mask to the output_dir.

2. Execute Segmentation in Parallel (segment_cells_in_parallel function):
Purpose: Orchestrates the parallel segmentation of all images in a given directory.

Parameters:
    input_path: Directory containing the input TIFF images.
    output_path: Directory where output masks are to be saved.
    workers: Number of worker threads to use for parallel processing.

Process:
    Ensures the output_path exists.
    Lists all TIFF images in input_path.
    Initializes the Cellpose model.
Creates a thread pool and dispatches segmentation tasks for each image to the pool.
Utilizes a progress bar to visually track task completion.

3. Main Function (main):

Purpose: Serves as the entry point for the script, enabling command-line execution.

Process:
Utilizes the fire.Fire function to parse command-line arguments for input_path, output_path, and workers.
Calls segment_cells_in_parallel with the parsed arguments to start the segmentation process.

Example Usage
The script is executed from the command line, allowing users to specify the input directory, output directory, and the number of workers. 

example command:


python CellposeSeg_3D_Multithread_CSVOrInputFolder.py --input_path="//allen/aics/microscopy/EOMES_Denoised/NewQCPositions/DenoisePerTimepoint" --output_path="D:\\CP3_EOMES_DenoisePerTimepoint_MultithreadedScript_CP_-2" --workers=16

'''

import os
import shutil
import numpy as np
import tifffile
from tqdm import tqdm
from cellpose import models
from concurrent.futures import ThreadPoolExecutor, as_completed
import fire
import csv
from cellpose.io import logger_setup
import time
import signal

# Function to handle termination signals
def handle_signal(signal_received, frame):
    print(f"Received signal: {signal_received}. Exiting gracefully.")
    exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# Function to read filenames from CSV
def read_filenames_from_csv(csv_path):
    start_time = time.time()
    filenames = []
    with open(csv_path, mode='r', newline='', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            filenames.append(row["Image Name"])
    print(f"Reading filenames from CSV took {time.time() - start_time:.2f} seconds")
    return filenames

# Define the function to segment a single image
def segment_image(model, img_path, output_dir, channels=[0, 0], diameter=60, flow_threshold=0.4, stitch_threshold=0.5, cellprob_threshold=-2.0, do_3D=False, anisotropy=5.87, z_axis=0, normalize=True, min_size=500):
    start_time = time.time()
    file = os.path.basename(img_path)
    output_name_3Dseg = file.split(".tif", 1)[0] + "_cp_masks.tif"
    output_path_3Dseg = os.path.join(output_dir, output_name_3Dseg)

    if os.path.exists(output_path_3Dseg):
        print(f"Skipping {file} as it is already processed.")
        return

    try:
        img = tifffile.imread(img_path)
        masks, flows, styles = model.eval([img], channels=channels, diameter=diameter, flow_threshold=flow_threshold, stitch_threshold=stitch_threshold, cellprob_threshold=cellprob_threshold, do_3D=do_3D, anisotropy=anisotropy, z_axis=z_axis, normalize=normalize, min_size=min_size)
        tifffile.imwrite(output_path_3Dseg, masks[0], compression='zlib')
    except Exception as e:
        print(f"Failed to evaluate model on {file}. Error: {e}")
    
    print(f"Processing {file} took {time.time() - start_time:.2f} seconds")

# Function to execute segmentation in parallel based on filenames from CSV or directory
def segment_cells_in_parallel(input_path, output_path, csv_path, workers):
    start_time = time.time()
    os.makedirs(output_path, exist_ok=True)

    if csv_path:
        csv_filenames = read_filenames_from_csv(csv_path)
        valid_filenames = [f"{name}.tif" for name in csv_filenames]
    else:
        valid_filenames = [f for f in os.listdir(input_path) if f.endswith('.tif')]

    logger_setup()
    model = models.CellposeModel(gpu=True, model_type='cyto3', diam_mean=60.0)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(segment_image, model, os.path.join(input_path, file), output_path) for file in valid_filenames]
        for future in tqdm(as_completed(futures), total=len(valid_filenames)):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing image: {e}")

    print(f"Total segmentation process took {time.time() - start_time:.2f} seconds")

# Main function to be executed with Fire for command-line argument parsing
def main(input_path, output_path, input_csv=None, workers=4):
    segment_cells_in_parallel(input_path, output_path, input_csv, workers)

if __name__ == "__main__":
    fire.Fire(main)
