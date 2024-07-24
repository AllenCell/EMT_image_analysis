'''

This script reads raw and mask TIFF images, filters objects based on size and intensity, clips pixels within the instance object masks of each object that are below the relative px intensity clipping threshold, optionally relabels the objects, removes small disconnected components, and saves the processed mask images. It also generates a summary CSV file with details about the processed objects.

Example usage:
    python ClipMaskBasedOnRelativeIntensityZSD20x_gpu.py --raw_dir /path/to/raw/images --mask_dir /path/to/mask/images --output_dir /path/to/output --scaling_factor 0.75 --pre_clipping_min_size 100 --post_clipping_min_size 100 --pre_clipping_min_mean_intensity 1 --post_clipping_min_integrated_intensity 1 --workers 8 --relabel False

Functions:
    handle_signal(signal, frame) -> None: Handles OS signals.
    clip_label_top(raw_img, mask_img, label, mean_intensity, scaling_factor, filename) -> np.array: Clips top intensity pixels of a given label.
    calculate_region_props(mask_img, raw_img, properties, filename) -> dict: Calculates region properties of the mask image.
    filter_objects(props, min_size, min_integrated_intensity, min_mean_intensity, filename) -> tuple: Filters objects based on size and intensity.
    remove_small_disconnected_objects(mask_img, props) -> np.array: Removes small disconnected components.
    process_single_tif(raw_path, mask_path, output_dir, scaling_factor, pre_clipping_min_size, post_clipping_min_size, pre_clipping_min_mean_intensity, post_clipping_min_integrated_intensity, relabel, csv_data) -> None: Processes a single TIFF file.
    process_directory(raw_dir, mask_dir, output_dir, scaling_factor, pre_clipping_min_size, post_clipping_min_size, pre_clipping_min_mean_intensity, post_clipping_min_integrated_intensity, workers, relabel) -> None: Processes all TIFF files in the specified directory.

Multithreading at the Image Level:

The process_directory function uses ThreadPoolExecutor to process multiple images concurrently. Each image is processed in a separate thread.
The executor.submit method is used to submit the process_single_tif function for each image.
Multithreading at the Object Level:

Inside the process_single_tif function, another ThreadPoolExecutor is used to process each label (object) concurrently.
The process_label function is responsible for clipping each label using the vectorized clip_label_top method.
Filtering and Region Properties Calculation:

The filter_objects function is used to filter objects and calculate their properties (volume and integrated intensity) on the whole image before the multithreaded clipping process.
Relabeling:

The relabel parameter controls whether the mask is relabeled to ensure sequential object IDs.
The relabel_sequential function is used if relabeling is enabled.

CSV Data:

Added a new_object_id column to the CSV to store the new object ID after relabeling.
If relabeling is not performed, this field is left blank.

# Using integrated intensity threshold
python ClipMaskBasedOnRelativeIntensityNikon.py --raw_dir="//path/to/raw_dir" --mask_dir="//path/to/mask_dir" --output_dir="//path/to/output_dir" --scaling_factor=0.75 --min_size=100 --min_integrated_intensity=1000 --workers=8 --relabel=False

# Using mean intensity threshold
python ClipMaskBasedOnRelativeIntensityNikon.py --raw_dir="//path/to/raw_dir" --mask_dir="//path/to/mask_dir" --output_dir="//path/to/output_dir" --scaling_factor=0.75 --min_size=100 --min_mean_intensity=10 --workers=8 --relabel=False

python ClipMaskBasedOnRelativeIntensityZSD20x_gpu.py --raw_dir="//allen/aics/microscopy/EOMES_Denoised/NewQCPositions/5827_P9_DenoisePerTimepoint" --mask_dir="//allen/aics/microscopy/EOMES_Denoised/NewQCPositions/5827_P9_CP3_SegBeforeClipping" --output_dir="//allen/aics/microscopy/EOMES_Denoised/NewQCPositions/Mask_CP_-2_0.75_MaskClipping --scaling_factor=0.75 --pre_clipping_min_size=1 --post_clipping_min_size=1 --pre_clipping_min_mean_intensity=1 --post_clipping_min_integrated_intensity=1 --workers=8 --relabel=False
'''

import os
import cupy as cp
import numpy as np
import tifffile as tiff
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from skimage.measure import label, regionprops_table
from skimage.segmentation import relabel_sequential
import pandas as pd
import signal
import gc

def handle_signal(signal, frame):
    """
    Handles OS signals for clean termination.
    
    Args:
        signal (int): The signal number.
        frame (frame object): The current stack frame.
    """
    print(f"Received signal: {signal}")

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def clip_label_top(raw_img, mask_img, label, mean_intensity, scaling_factor, filename):
    """
    Clips top intensity pixels of a given label in the mask image.

    Args:
        raw_img (cupy.ndarray): The raw image array.
        mask_img (cupy.ndarray): The mask image array.
        label (int): The label to be clipped.
        mean_intensity (float): The mean intensity of the label.
        scaling_factor (float): The scaling factor for clipping intensity.
        filename (str): The name of the file being processed.

    Returns:
        cupy.ndarray: The modified mask image.
    """
    try:
        print(f"Clipping label {label} in file {filename}...")
        label_mask = (mask_img == label)
        adjusted_mean_intensity = mean_intensity * scaling_factor

        intensity_mask = (raw_img < adjusted_mean_intensity) & label_mask

        mask_img[intensity_mask] = 0

        return mask_img
    except Exception as e:
        print(f"Error in clip_label_top for file {filename}, label {label}: {e}")
        raise

def calculate_region_props(mask_img, raw_img, properties, filename):
    """
    Calculates region properties of the mask image.

    Args:
        mask_img (cupy.ndarray): The mask image array.
        raw_img (cupy.ndarray): The raw image array.
        properties (list): List of properties to calculate.
        filename (str): The name of the file being processed.

    Returns:
        dict: A dictionary of calculated properties.
    """
    try:
        print(f"Calculating region properties for file {filename}...")
        mask_img_np = cp.asnumpy(mask_img)
        raw_img_np = cp.asnumpy(raw_img)
        labeled_array, _ = label(mask_img_np, return_num=True)
        props = regionprops_table(labeled_array, intensity_image=raw_img_np, properties=properties)
        return {k: cp.array(v) for k, v in props.items()}
    except Exception as e:
        print(f"Error in calculate_region_props for file {filename}: {e}")
        raise

def filter_objects(props, min_size, min_integrated_intensity=None, min_mean_intensity=None, filename=None):
    """
    Filters objects based on size and intensity.

    Args:
        props (dict): The properties of the regions.
        min_size (int): The minimum size of objects to keep.
        min_integrated_intensity (float, optional): The minimum integrated intensity of objects to keep.
        min_mean_intensity (float, optional): The minimum mean intensity of objects to keep.
        filename (str, optional): The name of the file being processed.

    Returns:
        tuple: Filtered labels and properties.
    """
    try:
        print(f"Filtering objects in file {filename}...")
        areas = props['area']
        mean_intensities = props['mean_intensity']
        integrated_intensities = mean_intensities * areas

        if min_integrated_intensity is not None:
            mask = (areas >= min_size) & (integrated_intensities >= min_integrated_intensity)
        elif min_mean_intensity is not None:
            mask = (areas >= min_size) & (mean_intensities >= min_mean_intensity)
        else:
            mask = (areas >= min_size)

        filtered_labels = props['label'][mask]
        filtered_props = {k: v[mask] for k, v in props.items()}

        print(f"Filtered {len(filtered_labels)} objects from {len(areas)} total objects in file {filename}.")
        return filtered_labels, filtered_props
    except Exception as e:
        print(f"Error in filter_objects for file {filename}: {e}")
        raise

def remove_small_disconnected_objects(mask_img, props):
    """
    Removes small disconnected components from the mask image.

    Args:
        mask_img (cupy.ndarray): The mask image array.
        props (dict): The properties of the regions.

    Returns:
        cupy.ndarray: The modified mask image.
    """
    try:
        print("Removing small disconnected objects...")
        unique_labels = np.unique(mask_img)
        new_mask = np.zeros_like(mask_img)

        for label in unique_labels:
            if label == 0:
                continue

            label_mask = (mask_img == label)
            sub_labels, sub_counts = np.unique(label_mask, return_counts=True)
            max_sub_label = sub_labels[np.argmax(sub_counts)]
            new_mask[label_mask & (sub_labels == max_sub_label)] = label

        return new_mask
    except Exception as e:
        print(f"Error in remove_small_disconnected_objects: {e}")
        raise

def process_single_tif(raw_path, mask_path, output_dir, scaling_factor, pre_clipping_min_size, post_clipping_min_size, pre_clipping_min_mean_intensity, post_clipping_min_integrated_intensity, relabel, csv_data):
    """
    Processes a single TIFF file, applying filters, clipping, and saving the results.

    Args:
        raw_path (str): Path to the raw image.
        mask_path (str): Path to the mask image.
        output_dir (str): Directory to save the output images.
        scaling_factor (float): Scaling factor for clipping intensity.
        pre_clipping_min_size (int): Minimum size of objects before clipping.
        post_clipping_min_size (int): Minimum size of objects after clipping.
        pre_clipping_min_mean_intensity (float): Minimum mean intensity of objects before clipping.
        post_clipping_min_integrated_intensity (float): Minimum integrated intensity of objects after clipping.
        relabel (bool): Whether to relabel the objects.
        csv_data (list): List to store the summary of processed objects.
    """
    mask_filename = os.path.basename(mask_path)
    output_file_path = os.path.join(output_dir, mask_filename)
    
    # Check if the output file already exists
    if os.path.exists(output_file_path):
        print(f"Output file already exists for {mask_filename}, skipping processing.")
        csv_data.append({
            "filename": mask_filename,
            "object_id": None,
            "volume": None,
            "integrated_intensity": None,
            "mean_intensity": None,
            "retained": 0,
            "new_object_id": None
        })
        return

    print(f"Processing: {mask_filename}")
    raw_img, mask_img, filtered_mask, modified_mask, post_filtered_mask = None, None, None, None, None
    try:
        print(f"Reading raw image from {raw_path}")
        raw_img = cp.array(tiff.imread(raw_path))
        print(f"Reading mask image from {mask_path}")
        mask_img = cp.array(tiff.imread(mask_path))

        # Pre-Clipping Filter
        properties = ['label', 'area', 'mean_intensity']
        props = calculate_region_props(mask_img, raw_img, properties, mask_filename)
        
        mean_area = cp.mean(props['area'])
        mean_intensity = cp.mean(props['mean_intensity'])
        print(f"Mean object size before pre-clipping filtering: {mean_area} in file {mask_filename}")
        print(f"Mean object intensity before pre-clipping filtering: {mean_intensity} in file {mask_filename}")

        filtered_labels, filtered_props = filter_objects(props, pre_clipping_min_size, min_mean_intensity=pre_clipping_min_mean_intensity, filename=mask_filename)

        mean_area_filtered = cp.mean(filtered_props['area']) if len(filtered_props['area']) > 0 else 0
        mean_intensity_filtered = cp.mean(filtered_props['mean_intensity']) if len(filtered_props['mean_intensity']) > 0 else 0
        print(f"Mean object size after pre-clipping filtering: {mean_area_filtered} in file {mask_filename}")
        print(f"Mean object intensity after pre-clipping filtering: {mean_intensity_filtered} in file {mask_filename}")
        
        filtered_mask = cp.zeros_like(mask_img)
        for label in filtered_labels:
            filtered_mask[mask_img == label] = label
        
        unique_labels = filtered_labels
        modified_mask = cp.zeros_like(mask_img)

        for label in unique_labels:
            label_pixels = raw_img[filtered_mask == label]
            if label_pixels.size == 0:
                continue
            result_mask = clip_label_top(raw_img, filtered_mask, label, label_pixels.mean(), scaling_factor, mask_filename)
            modified_mask[filtered_mask == label] = result_mask[filtered_mask == label]

        if relabel:
            print(f"Relabeling masks for file {mask_filename}...")
            modified_mask_np = cp.asnumpy(modified_mask)
            print(f"Shape of modified mask for relabeling: {modified_mask_np.shape}")
            print(f"Data type of modified mask for relabeling: {modified_mask_np.dtype}")
            modified_mask_np, forward_map, _ = relabel_sequential(modified_mask_np)
            print(f"Relabeling completed for file {mask_filename}")
            modified_mask = cp.array(modified_mask_np)
            new_ids = {int(old): int(new) for old, new in enumerate(forward_map) if old != 0}
            
            # Remove small disconnected objects
            modified_mask = remove_small_disconnected_objects(modified_mask, props)
        else:
            new_ids = {}

        # Post-Clipping Filter
        post_clipping_props = calculate_region_props(modified_mask, raw_img, properties, mask_filename)
        post_filtered_labels, post_filtered_props = filter_objects(post_clipping_props, post_clipping_min_size, min_integrated_intensity=post_clipping_min_integrated_intensity, filename=mask_filename)

        post_filtered_mask = cp.zeros_like(modified_mask)
        for label in post_filtered_labels:
            post_filtered_mask[modified_mask == label] = label

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save the output image immediately after relabeling
        print(f"Saving output for file {mask_filename} after relabeling and post-clipping filtering...")
        tiff.imwrite(output_file_path, cp.asnumpy(post_filtered_mask), photometric='minisblack', compression='zlib')
        print(f"Completed and saved: {mask_filename}")

        for label, volume, integrated_intensity, mean_intensity in zip(
            post_filtered_props['label'], post_filtered_props['area'], 
            post_filtered_props.get('mean_intensity', [None]*len(post_filtered_props['label'])),
            post_filtered_props.get('mean_intensity', [None]*len(post_filtered_props['label']))):

            integrated_intensity = mean_intensity * volume if mean_intensity is not None else None
            csv_data.append({
                "filename": mask_filename,
                "object_id": int(label),  # Convert to int
                "volume": int(volume),  # Convert to int
                "integrated_intensity": float(integrated_intensity),
                "mean_intensity": float(mean_intensity),
                "retained": int(cp.any(post_filtered_mask == label)),
                "new_object_id": new_ids.get(int(label), "")
            })

    except Exception as e:
        print(f"Error processing file {mask_filename}: {e}")
        csv_data.append({
            "filename": mask_filename,
            "object_id": None,
            "volume": None,
            "integrated_intensity": None,
            "mean_intensity": None,
            "retained": 0,
            "new_object_id": None
        })

    finally:
        # Free up memory
        if raw_img is not None:
            del raw_img
        if mask_img is not None:
            del mask_img
        if filtered_mask is not None:
            del filtered_mask
        if modified_mask is not None:
            del modified_mask
        if post_filtered_mask is not None:
            del post_filtered_mask
        gc.collect()

def process_directory(raw_dir, mask_dir, output_dir, scaling_factor=1.0, pre_clipping_min_size=100, post_clipping_min_size=100, pre_clipping_min_mean_intensity=1, post_clipping_min_integrated_intensity=100000, workers=8, relabel=False):
    """
    Processes all TIFF files in the specified directory, applying filters, clipping, and saving the results.

    Args:
        raw_dir (str): Directory containing raw images.
        mask_dir (str): Directory containing mask images.
        output_dir (str): Directory to save the output images.
        scaling_factor (float): Scaling factor for clipping intensity.
        pre_clipping_min_size (int): Minimum size of objects before clipping.
        post_clipping_min_size (int): Minimum size of objects after clipping.
        pre_clipping_min_mean_intensity (float): Minimum mean intensity of objects before clipping.
        post_clipping_min_integrated_intensity (float): Minimum integrated intensity of objects after clipping.
        workers (int): Number of worker threads to use.
        relabel (bool): Whether to relabel the objects.

    Returns:
        None
    """
    # Ensure output_dir does not include arguments by splitting it at space and taking the first part
    output_dir = output_dir.split()[0]

    print(f"Starting processing directory with raw_dir: {raw_dir}, mask_dir: {mask_dir}, output_dir: {output_dir}")
    try:
        raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.tif')]
    except Exception as e:
        print(f"Error listing files in raw_dir {raw_dir}: {e}")
        return

    mask_files = {}
    for f in raw_files:
        base_name = f.replace(".tif", "")
        mask_file_tif = os.path.join(mask_dir, base_name + "_cp_masks.tif")
        mask_file_tiff = os.path.join(mask_dir, base_name + "_cp_masks.tiff")
        if os.path.exists(mask_file_tif):
            mask_files[f] = mask_file_tif
        elif os.path.exists(mask_file_tiff):
            mask_files[f] = mask_file_tiff

    print(f"Found {len(raw_files)} raw files and {len(mask_files)} corresponding mask files.")
    csv_data = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for raw_file in tqdm(raw_files, desc="Processing Files"):
            mask_path = mask_files.get(raw_file)
            if mask_path is None:
                print(f"No corresponding mask file found for {raw_file}")
                csv_data.append({
                    "filename": raw_file,
                    "object_id": None,
                    "volume": None,
                    "integrated_intensity": None,
                    "mean_intensity": None,
                    "retained": 0,
                    "new_object_id": None
                })
                continue
            raw_path = os.path.join(raw_dir, raw_file)
            future = executor.submit(process_single_tif, raw_path, mask_path, output_dir, scaling_factor, pre_clipping_min_size, post_clipping_min_size, pre_clipping_min_mean_intensity, post_clipping_min_integrated_intensity, relabel, csv_data)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Files Completed"):
            try:
                future.result()
            except Exception as e:
                print(f"Error completing future for a file: {e}")

    csv_output_path = os.path.join(output_dir, "object_summary.csv")
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_output_path, index=False)
    print(f"CSV summary saved to {csv_output_path}")

if __name__ == '__main__':
    import fire
    fire.Fire(process_directory)
