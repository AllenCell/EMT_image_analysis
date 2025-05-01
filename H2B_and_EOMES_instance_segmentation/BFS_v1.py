# streamlit run BFS_v1.py --server.fileWatcherType none

import streamlit as st 
st.set_page_config(layout="wide")
import os
import time
import datetime
import numpy as np
import csv
from pathlib import Path
from tifffile import imwrite, imread
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import gc
import pandas as pd
import io as stio  # standard library io
import ast  # For literal evaluation of dictionary strings
import subprocess
import sys
import tempfile
import re

# Import Cellpose modules and additional readers.
from cellpose import io, denoise, transforms, models as cp_models
io.logger_setup()

# Additional bioio readers for TIFF, OME-Zarr, and CZI formats:
from bioio import BioImage
import bioio_ome_zarr
import bioio_tifffile
import bioio_czi

# For mask clipping method
import cupy as cp
import skimage.morphology
from skimage.measure import regionprops_table
from skimage.transform import resize  # used for downsampling segmentation masks

# -------------------- Shutdown Button --------------------
st.subheader("Click this button to shutdown streamlit GUI server so the process closes properly")
if st.button("Shutdown Server", key="shutdown_btn"):
    st.write("Shutting down the server...")
    import signal
    os.kill(os.getpid(), signal.SIGTERM)

# ==================== File Loading Utilities ====================
def load_image(image_path):
    """
    Loads an image file using BioImage with the appropriate reader based on the file extension.
    Supported formats: .zarr, .ome.zarr, .tif/.tiff, .czi.
    Also converts S3 paths to HTTPS URLs.
    """
    startload = time.time()
    if isinstance(image_path, str) and image_path.startswith("s3://"):
        try:
            _, rest = image_path.split("s3://", 1)
            bucket, key = rest.split("/", 1)
            # Sometimes the S3 path might require a second split:
            bucket, key = rest.split("/", 1)
            image_path = f"https://{bucket}.s3.amazonaws.com/{key}"
        except Exception as e:
            raise ValueError(f"Unable to parse the S3 path: {image_path}") from e
    ext = Path(image_path).suffix.lower()
    if ext in [".zarr", ".ome.zarr"]:
        reader = bioio_ome_zarr.Reader
    elif ext in [".tif", ".tiff"]:
        reader = bioio_tifffile.Reader
    elif ext == ".czi":
        reader = bioio_czi.Reader
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    bio_img = BioImage(image_path, reader=reader)
    print(f"[File Loader] Loaded file {image_path} with shape {bio_img.shape} in {time.time()-startload:.2f}s")
    try:
        print("Physical pixel sizes (Z, Y, X):", bio_img.physical_pixel_sizes)
    except AttributeError:
        print("Physical pixel sizes metadata not available for this image.")
    return bio_img

def get_image_paths(input_path):
    """
    Returns a list of image paths.
    """
    supported_exts = [".zarr", ".ome.zarr", ".tif", ".tiff", ".czi"]
    image_paths = []
    if input_path.lower().endswith('.csv'):
        with open(input_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            print("CSV headers:", reader.fieldnames)
            for row in reader:
                s3_path = None
                for key, value in row.items():
                    if key.strip().lower() == "raw converted file download":
                        if value is not None:
                            s3_path = value.strip()
                        break
                if s3_path:
                    image_paths.append(s3_path)
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for f in files:
                if any(f.lower().endswith(ext) for ext in supported_exts):
                    image_paths.append(os.path.join(root, f))
    else:
        if any(input_path.lower().endswith(ext) for ext in supported_exts):
            image_paths = [input_path]
        else:
            st.error("Input file format not supported.")
    return image_paths

# ==================== Denoising Utility Functions ====================
def ConvertFloatToUint16(img):
    """
    Converts a float32 image to uint16 by scaling intensities to 0-65535.
    """
    if img.dtype == np.uint16:
        return img
    minimum_val = img.min()
    maximum_val = img.max()
    scale = 1.0 if maximum_val == minimum_val else 65535.0 / (maximum_val - minimum_val)
    img = (img - minimum_val) * scale
    img[img < 0.0] = 0.0
    img[img > 65535] = 65535
    return img.astype(np.uint16)

def RescaleInputImage(image_offset_corrected, val1, val2, constant_mode=False):
    """
    In dynamic mode (constant_mode=False), rescales using the measured min and max.
    In constant mode (constant_mode=True), uses val1 as raw_max and val2 (p99) as the maximum for scaling.
    """
    if constant_mode:
        maximum_val = val1
        scale = 65535.0 / maximum_val
        image_rescaled = image_offset_corrected * scale
        image_rescaled = np.clip(image_rescaled, 0, 65535)
    else:
        minimum_val = image_offset_corrected.min()
        maximum_val = image_offset_corrected.max()
        scale = 1.0 if maximum_val == minimum_val else 65535.0 / (maximum_val - minimum_val)
        image_rescaled = (image_offset_corrected - minimum_val) * scale
    image_rescaled = np.clip(image_rescaled, 0, 65535)
    return image_rescaled.astype(np.uint16), scale

def save_image(image, output_path, convert=True):
    """
    Saves an image to the specified output path using tifffile.
    """
    image = np.squeeze(image)
    metadata = {'axes': 'ZYX'}
    if convert:
        image = ConvertFloatToUint16(image)
    imwrite(str(output_path), image, photometric='minisblack', metadata=metadata)

def process_timepoint(bio_img_or_path, t, image_name, channel, denoise_model, eval_params, output_path, input_scale_log, output_scale_log, save_raw=False, constant_mode=False, const_params=None):
    """
    Processes a single timepoint.
    In dynamic mode, computes metrics from the image.
    In constant mode, uses provided constants.
    Optionally saves the raw image in a "Raw" subfolder.
    Saves the denoised image with a _T{t}.tif suffix in the "Denoised" subfolder.
    """
    print(f"[Denoising] Processing timepoint {t} for {image_name}")
    total_start = time.time()
    bio_img = bio_img_or_path

    start = time.time()
    lazy_image = bio_img.get_image_dask_data("CZYX", T=t)
    image = lazy_image.compute()
    print(f"  Retrieved data in {time.time()-start:.2f}s")

    start = time.time()
    image = image[channel]
    print(f"  Selected channel in {time.time()-start:.2f}s")

    if save_raw:
        raw_folder = Path(output_path) / "Raw"
        raw_folder.mkdir(parents=True, exist_ok=True)
        raw_filename = raw_folder / f"{image_name}_T{t}_raw.tif"
        raw_to_save = ConvertFloatToUint16(image) if image.dtype != np.uint16 else image
        imwrite(str(raw_filename), raw_to_save, photometric='minisblack')
        print(f"  Saved raw image to {raw_filename}")

    camera_offset = 100
    start = time.time()
    if constant_mode:
        raw_max_val = const_params['raw_max']
        p99_const = const_params['p99']
    else:
        raw_max_val = image.max()
    image = image.astype(np.float32, copy=False)
    image -= camera_offset
    np.maximum(image, 0, out=image)
    print(f"  Offset correction in {time.time()-start:.2f}s")

    start = time.time()
    if constant_mode:
        image_rescaled, scale = RescaleInputImage(image, raw_max_val, p99_const, constant_mode=True)
    else:
        image_rescaled, scale = RescaleInputImage(image, image.min(), image.max(), constant_mode=False)
    p1 = np.percentile(image_rescaled, 1)
    if not constant_mode:
        p99 = np.percentile(image_rescaled, 99)
    else:
        p99 = p99_const
    print(f"  Rescaled & percentiles in {time.time()-start:.2f}s")

    input_scale_log.append([f"{image_name}_T{t}", scale, image.min(), image.max(), p1, p99])
    local_eval_params = eval_params.copy()
    local_eval_params['lowhigh'] = [p1, p99]
    
    start = time.time()
    denoised_image = denoise_model.eval(x=image_rescaled, channels=[0, 0],
                                        tile=False, normalize=local_eval_params)
    print(f"  Denoised in {time.time()-start:.2f}s")
    
    start = time.time()
    denoised_uint16 = ConvertFloatToUint16(denoised_image)
    print(f"  Converted to uint16 in {time.time()-start:.2f}s")
    output_raw_min = int(denoised_uint16.min())
    output_raw_max = int(denoised_uint16.max())
    p1_out = np.percentile(denoised_uint16, 1)
    p99_out = np.percentile(denoised_uint16, 99)
    output_scale_log.append([f"{image_name}_T{t}", scale, output_raw_min, output_raw_max, p1_out, p99_out])
    
    denoised_folder = Path(output_path) / "Denoised"
    denoised_folder.mkdir(parents=True, exist_ok=True)
    denoised_filename = denoised_folder / f"{image_name}_T{t}.tif"
    save_image(denoised_uint16, denoised_filename, convert=False)
    print(f"  Saved denoised image in {time.time()-start:.2f}s")
    print(f"Completed timepoint {t} for {image_name} in {time.time()-total_start:.2f}s")
    return (image_name, t, denoised_uint16)

def load_file_and_get_tasks(image_path, denoise_model, eval_params, channel, output_path, input_scale_log, output_scale_log, timepoint_range, save_raw, constant_mode, const_params):
    """
    Loads an image file and generates denoising tasks for each timepoint.
    """
    base_name = os.path.basename(image_path)
    image_name = ".".join(base_name.split('.')[:-1])
    print(f"[Denoising] Loading file {image_name}")
    start = time.time()
    bio_img = load_image(image_path)
    dim_shape = bio_img.shape
    scene_count = bio_img.scenes
    dim_order = bio_img.dims.order
    num_timepoints = dim_shape[0]
    print(f"  Retrieved shape {dim_shape} in {time.time()-start:.2f}s")
    print(f"  Retrieved scene count {scene_count} in {time.time()-start:.2f}s")
    print(f"  Retrieved dim order {dim_order} in {time.time()-start:.2f}s")
    if timepoint_range and timepoint_range.strip().lower() != "all":
        try:
            parts = timepoint_range.strip().split("-")
            start_tp = int(parts[0]) - 1
            end_tp = int(parts[1]) - 1 if len(parts) > 1 and parts[1] != "" else num_timepoints - 1
        except Exception as e:
            print(f"Error parsing timepoint range '{timepoint_range}': {e}. Processing all timepoints.")
            start_tp, end_tp = 0, num_timepoints - 1
    else:
        start_tp, end_tp = 0, num_timepoints - 1
    tasks = []
    for t in range(start_tp, min(end_tp + 1, num_timepoints)):
        tasks.append((bio_img, t, image_name, channel, denoise_model, eval_params,
                      output_path, input_scale_log, output_scale_log, save_raw))
    return tasks

def write_input_scale_log(output_path, input_scale_log):
    rescale_scale_dir = Path(output_path) / "DenoiseRescaleMetrics"
    rescale_scale_dir.mkdir(parents=True, exist_ok=True)
    csv_file = rescale_scale_dir / "input_image_rescale_values.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Rescale Factor", "Raw Min", "Raw Max", "Percentile 1", "Percentile 99"])
        writer.writerows(input_scale_log)

def write_output_scale_log(output_path, output_scale_log):
    rescale_scale_dir = Path(output_path) / "DenoiseRescaleMetrics"
    rescale_scale_dir.mkdir(parents=True, exist_ok=True)
    csv_file = rescale_scale_dir / "offset_corrected_image_rescale_metrics.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Rescale Factor", "Raw Min", "Raw Max", "Percentile 1", "Percentile 99"])
        writer.writerows(output_scale_log)

def run_denoising(input_path, output_path, max_workers, file_workers, channel, timepoint_range, save_raw):
    """
    Runs dynamic denoising in parallel and returns a dictionary mapping image names 
    to (stacked_volume, timepoints).
    """
    model_params = {
        'gpu': True,
        'model_type': 'denoise_cyto',
        'nchan': 1,
        'chan2': False,
    }
    eval_params = {
        'normalize': True,
        'norm3D': True,
    }
    denoise_model = denoise.DenoiseModel(**model_params)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    input_scale_log = []
    output_scale_log = []
    image_paths = get_image_paths(input_path)
    if not image_paths:
        st.error(f"No images found in {input_path}")
        return {}
    all_tasks = []
    print("[Denoising] Loading files and generating tasks...")
    file_futures = []
    with ThreadPoolExecutor(max_workers=file_workers) as file_executor:
        for image_path in image_paths:
            file_futures.append(file_executor.submit(load_file_and_get_tasks, image_path,
                                                       denoise_model, eval_params,
                                                       channel, str(output_path),
                                                       input_scale_log, output_scale_log,
                                                       timepoint_range, save_raw, False, None))
        for future in tqdm(as_completed(file_futures), total=len(file_futures), desc="Loading files"):
            try:
                tasks = future.result()
                all_tasks.extend(tasks)
            except Exception as e:
                print(f"Error loading file: {e}")
    print(f"[Denoising] Total timepoint tasks: {len(all_tasks)}")
    denoised_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as task_executor:
        futures = [task_executor.submit(process_timepoint, *task) for task in all_tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing timepoints"):
            try:
                result = future.result()
                denoised_results.append(result)
            except Exception as e:
                print(f"Error processing timepoint: {e}")
    write_input_scale_log(str(output_path), input_scale_log)
    write_output_scale_log(str(output_path), output_scale_log)
    print("[Denoising] Processing complete.")
    denoised_dict = {}
    for image_name, t, img in denoised_results:
        if image_name not in denoised_dict:
            denoised_dict[image_name] = []
        denoised_dict[image_name].append((t, img))
    for key in denoised_dict:
        denoised_dict[key].sort(key=lambda x: x[0])
        timepoints = [t for t, _ in denoised_dict[key]]
        stacked = np.stack([img for _, img in denoised_dict[key]], axis=0)
        denoised_dict[key] = (stacked, timepoints)
    return denoised_dict

# ==================== Constant Denoising Functions ====================
def ConvertFloatToUint16_constant(denoised_image, denoised_min, denoised_max):
    if denoised_min < 0:
        offset_val = -denoised_min
    else:
        offset_val = 0
    adjusted_denoised_min = denoised_min + offset_val
    adjusted_denoised_max = denoised_max + offset_val
    scale = 65535.0 / (adjusted_denoised_max - adjusted_denoised_min)
    denoised_image_scaled = (denoised_image + offset_val) * scale
    denoised_image_scaled = np.clip(denoised_image_scaled, 0, 65535)
    return denoised_image_scaled.astype(np.uint16)

def save_image_constant(image, output_path, denoised_min, denoised_max):
    image = np.squeeze(image)
    st.write(f"Saving image to: {output_path}")
    save_image(image, output_path, convert=True)

def process_image_constant_timepoint(bio_img, t, image_name, denoise_model, eval_params, const_params, scale_log, output_path, timepoint_range, save_raw):
    """
    Processes a single timepoint in constant mode.
    Loads the image for timepoint t using dask, applies constant rescaling using provided constants,
    denoises, converts the result to uint16, and saves the output with a _T{t}.tif suffix.
    Also saves the raw image in the "Raw" subfolder if save_raw is True.
    """
    T = bio_img.shape[0]
    if timepoint_range and timepoint_range.strip().lower() != "all":
        try:
            parts = timepoint_range.strip().split("-")
            start_tp = int(parts[0]) - 1
            end_tp = int(parts[1]) - 1 if len(parts) > 1 and parts[1] != "" else T - 1
        except Exception as e:
            start_tp, end_tp = 0, T - 1
    else:
        start_tp, end_tp = 0, T - 1
    if not (start_tp <= t <= end_tp):
        return None

    lazy_image = bio_img.get_image_dask_data("CZYX", T=t)
    image = lazy_image.compute()
    channel = get_param("Channel", 1)
    image = image[channel]
    if save_raw:
        raw_folder = Path(output_path) / "Raw"
        raw_folder.mkdir(parents=True, exist_ok=True)
        raw_filename = raw_folder / f"{image_name}_T{t}_raw.tif"
        raw_to_save = ConvertFloatToUint16(image) if image.dtype != np.uint16 else image
        imwrite(str(raw_filename), raw_to_save, photometric='minisblack')
        print(f"  Saved raw image to {raw_filename}")

    # Use offset from GUI (global variable "offset" is set in run_streamlit_app)
    image_offset_corrected = image.astype(np.float32) - offset
    image_offset_corrected[image_offset_corrected < 0] = 0
    raw_max_val = const_params['raw_max']
    p99 = const_params['p99']
    image_rescaled, scale = RescaleInputImage(image_offset_corrected, raw_max_val, p99, constant_mode=True)
    local_eval_params = eval_params.copy()
    local_eval_params['lowhigh'] = [0, p99]
    denoised_image = denoise_model.eval(x=image_rescaled, channels=[0, 0], tile=False, normalize=local_eval_params)
    denoised_uint16 = ConvertFloatToUint16(denoised_image)
    denoised_folder = Path(output_path) / "Denoised"
    denoised_folder.mkdir(parents=True, exist_ok=True)
    denoised_filename = denoised_folder / f"{image_name}_T{t}.tif"
    save_image(denoised_uint16, denoised_filename, convert=False)
    metrics = {
        'Image Name': f"{image_name}_T{t}",
        'Scale Factor': scale,
        'Raw Input Max': raw_max_val,
        'Percentile 99 Used': p99,
        'Min Intensity': denoised_uint16.min(),
        'Median Intensity': np.median(denoised_uint16),
        'Mean Intensity': np.mean(denoised_uint16),
        'Max Intensity': denoised_uint16.max(),
        'Max Intensity Count': np.sum(denoised_uint16 == 65535),
        'Percentile 99': np.percentile(denoised_uint16, 99)
    }
    scale_log.append(metrics)
    return (image_name, t, denoised_uint16)

def constant_denoise_directory(input_dir, output_dir, model_params, eval_params, max_workers, rescaling_method, const_params, timepoint_range, save_raw):
    image_output = Path(output_dir)
    denoised_out = image_output / "Denoised"
    denoised_out.mkdir(parents=True, exist_ok=True)
    denoise_model = denoise.DenoiseModel(**model_params)
    image_paths = get_image_paths(input_dir)
    scale_log = []
    tasks = []
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        image_name = os.path.splitext(base_name)[0]
        bio_img = load_image(image_path)
        T = bio_img.shape[0]
        if timepoint_range and timepoint_range.strip().lower() != "all":
            try:
                parts = timepoint_range.strip().split("-")
                start_tp = int(parts[0]) - 1
                end_tp = int(parts[1]) - 1 if len(parts) > 1 and parts[1] != "" else T - 1
            except Exception as e:
                start_tp, end_tp = 0, T - 1
        else:
            start_tp, end_tp = 0, T - 1
        for t in range(start_tp, min(end_tp + 1, T)):
            denoised_filename = denoised_out / f"{image_name}_T{t}.tif"
            if denoised_filename.exists():
                continue
            tasks.append((bio_img, t, image_name, denoise_model, eval_params, const_params, scale_log, str(image_output), timepoint_range, save_raw))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image_constant_timepoint, *task) for task in tasks]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res is not None:
                results.append(res)
    write_scale_log_constant(output_dir, scale_log)
    denoised_dict = {}
    for image_name, t, img in results:
        if image_name not in denoised_dict:
            denoised_dict[image_name] = []
        denoised_dict[image_name].append((t, img))
    for key in denoised_dict:
        denoised_dict[key].sort(key=lambda x: x[0])
        timepoints = [t for t, _ in denoised_dict[key]]
        stacked = np.stack([img for _, img in denoised_dict[key]], axis=0)
        denoised_dict[key] = (stacked, timepoints)
    return denoised_dict

def write_scale_log_constant(output_path, scale_log):
    scale_log_path = Path(output_path) / "output_intensity_metrics.csv"
    st.write(f"Writing scale log to: {scale_log_path}")
    with open(scale_log_path, 'w', newline='') as csvfile:
        fieldnames = ['Image Name','Scale Factor','Raw Input Max','Percentile 99 Used','Min Intensity','Median Intensity','Mean Intensity','Max Intensity','Max Intensity Count','Percentile 99']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in scale_log:
            writer.writerow(entry)

# ==================== Clipping Functions ====================
def adaptive_clip(raw_img, mask_img, label, mean_intensity, scaling_factor, slice_scaling_percentile, filename):
    try:
        # print(f"Clipping label {label} in file {filename}...")
        label_mask = (mask_img == label)
        adjusted_mean_intensity = mean_intensity * (scaling_factor / 100.0)
        
        intensity_mask = (raw_img < adjusted_mean_intensity) & label_mask
        mask_img[intensity_mask] = 0

        nucleus_pixels = raw_img[label_mask]
        median_intensity = cp.median(nucleus_pixels).item()

        slice_threshold = median_intensity * (slice_scaling_percentile / 100.0)

        slice_intensity_mask = (raw_img < slice_threshold) & label_mask
        mask_img[slice_intensity_mask] = 0

        # print(f"Median intensity for label {label} in file {filename}: {median_intensity}")
        # print(f"Slice threshold for label {label} in file {filename}: {slice_threshold}")

        return mask_img
    except Exception as e:
        print(f"Error in adaptive_clip for file {filename}, label {label}: {e}")
        raise

def calculate_region_props(mask_img, raw_img, properties, filename):
    try:
        # print(f"Calculating region properties for file {filename} using scikit-image on CPU...")
        mask_img_np = cp.asnumpy(mask_img)
        raw_img_np = cp.asnumpy(raw_img)
        props = regionprops_table(mask_img_np, intensity_image=raw_img_np, properties=properties)
        if len(props['label']) == 0:
            print(f"No objects found for file {filename} after property calculation.")
        return {k: v for k, v in props.items()}
    except Exception as e:
        print(f"Error in calculate_region_props for file {filename}: {e}")
        raise

def fill_holes_3d_gpu(mask, min_hole_diameter=64.0):
    """
    Fills small holes in each 2D slice of a 3D mask using skimage.morphology.remove_small_holes.
    """
    filled_mask = cp.copy(mask)
    radius = min_hole_diameter / 2.0
    min_hole_size = int(np.pi * (radius ** 2))
    # print(f"Using minimum hole size of {min_hole_size} pixels (from diameter {min_hole_diameter})")
    for z in range(mask.shape[0]):
        slice_mask = cp.asnumpy(mask[z])
        slice_labels = np.unique(slice_mask)[1:] if np.any(slice_mask) else []
        if len(slice_labels) == 0:
            continue
        filled_slice = slice_mask.copy()
        for label in slice_labels:
            binary_mask = slice_mask == label
            filled_binary = skimage.morphology.remove_small_holes(binary_mask, area_threshold=min_hole_size)
            filled_slice[filled_binary] = label
        filled_mask[z] = cp.array(filled_slice)
    return filled_mask

def process_single_tif(raw_path, mask_path, output_dir, scaling_factor, slice_scaling_percentile,
                       pre_clipping_min_size, post_clipping_min_size, pre_clipping_min_mean_intensity,
                       post_clipping_min_integrated_intensity, relabel, fill_holes, min_hole_diameter, skip_existing):
    """
    Processes a single raw/mask image pair.
    """
    mask_filename = os.path.basename(mask_path)
    output_file_path = os.path.join(output_dir, mask_filename)
    file_data = []
    if skip_existing and os.path.exists(output_file_path):
        print(f"Output file already exists for {mask_filename}, skipping.")
        return [{"filename": mask_filename, "object_id": None, "volume": None,
                 "integrated_intensity": None, "mean_intensity": None, "retained": 0, "new_object_id": None}]
    print(f"Processing: {mask_filename}")
    try:
        raw_img = cp.array(imread(raw_path))
        mask_img = cp.array(imread(mask_path))
        properties = ['label', 'area', 'mean_intensity']
        props = calculate_region_props(mask_img, raw_img, properties, mask_filename)
        if len(props['label']) == 0:
            print(f"No objects found for file {mask_filename}.")
            return file_data
        areas = cp.array(props['area'])
        mean_intensities = cp.array(props['mean_intensity'])
        labels = cp.array(props['label'])
        print(f"File {mask_filename}: {len(labels)} labels before pre-clipping filtering.")
        # print(f"Areas: {cp.asnumpy(areas)}")
        # print(f"Mean Intensities: {cp.asnumpy(mean_intensities)}")
        
        valid_indices = (areas >= pre_clipping_min_size) & (mean_intensities >= pre_clipping_min_mean_intensity)
        filtered_labels = labels[valid_indices]
        print(f"File {mask_filename}: {len(filtered_labels)} labels after pre-clipping filtering.")
        if len(filtered_labels) == 0:
            print(f"No valid labels remain for file {mask_filename}.")
            return file_data
        
        filtered_mask = cp.isin(mask_img, filtered_labels) * mask_img
        modified_mask = cp.copy(filtered_mask)

        for label in filtered_labels:
            # print(f"About to clip label {label} in file {mask_filename}")
            label_mask = (modified_mask == label)
            if cp.any(label_mask):
                computed_mean_intensity = cp.mean(raw_img[label_mask])
                # print(f"Label {label}: computed mean intensity = {computed_mean_intensity.item()}")
                modified_mask = adaptive_clip(raw_img, modified_mask, label, computed_mean_intensity,
                                               scaling_factor, slice_scaling_percentile, mask_filename)
            # else:
            #     print(f"Label {label} has no pixels in the mask!")

        props_post = calculate_region_props(modified_mask, raw_img, properties, mask_filename)
        post_areas = cp.array(props_post['area'])
        post_mean_intensities = cp.array(props_post['mean_intensity'])
        post_labels = cp.array(props_post['label'])
        post_integrated_intensity = post_mean_intensities * post_areas
        # print(f"File {mask_filename}: {len(post_labels)} labels after adaptive clipping (before post-filtering).")
        
        valid_post_indices = (post_areas >= post_clipping_min_size) & (post_integrated_intensity >= post_clipping_min_integrated_intensity)
        post_filtered_labels = post_labels[valid_post_indices]
        # print(f"File {mask_filename}: {len(post_filtered_labels)} labels after post-clipping filtering.")
        
        post_filtered_mask = cp.isin(modified_mask, post_filtered_labels) * modified_mask
        
        if fill_holes:
            # print(f"Filling holes slice-wise in mask for {mask_filename}...")
            post_filtered_mask = fill_holes_3d_gpu(post_filtered_mask, min_hole_diameter)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # print(f"Saving output file to {output_file_path}")
        imwrite(output_file_path, cp.asnumpy(post_filtered_mask), photometric='minisblack', compression='zlib')
        print(f"File saved successfully: {output_file_path}")
        
        for label in post_filtered_labels:
            file_data.append({
                "filename": mask_filename,
                "object_id": int(label),
                "volume": int(post_areas[post_labels == label][0]),
                "integrated_intensity": float(post_integrated_intensity[post_labels == label][0]),
                "mean_intensity": float(post_mean_intensities[post_labels == label][0]),
                "retained": 1,
                "new_object_id": None
            })
    except Exception as e:
        print(f"Error processing file {mask_filename}: {e}")
        file_data.append({
            "filename": mask_filename,
            "object_id": None,
            "volume": None,
            "integrated_intensity": None,
            "mean_intensity": None,
            "retained": 0,
            "new_object_id": None
        })
    finally:
        for var in ['raw_img', 'mask_img', 'filtered_mask', 'modified_mask', 'post_filtered_mask']:
            if var in locals():
                del locals()[var]
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    return file_data

def process_directory(raw_dir, mask_dir, output_dir, scaling_factor, slice_scaling_percentile,
                      pre_clipping_min_size, post_clipping_min_size, pre_clipping_min_mean_intensity,
                      post_clipping_min_integrated_intensity, workers, relabel, fill_holes, min_hole_diameter, skip_existing):
    """
    Processes a directory of raw images and their corresponding segmentation masks.
    """
    raw_dir = os.path.normpath(raw_dir)
    mask_dir = os.path.normpath(mask_dir)
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting clipping with raw_dir: {raw_dir}, mask_dir: {mask_dir}, output_dir: {output_dir}")
    try:
        raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.tif', '.tiff'))]
    except Exception as e:
        print(f"Error listing files in raw_dir {raw_dir}: {e}")
        return
    mask_files = {}
    for f in raw_files:
        base_name = f.replace(".tiff", "").replace(".tif", "")
        mask_file_tif = os.path.join(mask_dir, base_name + "_cp_masks.tif")
        mask_file_tiff = os.path.join(mask_dir, base_name + "_cp_masks.tiff")
        if os.path.exists(mask_file_tif):
            mask_files[f] = mask_file_tif
        elif os.path.exists(mask_file_tiff):
            mask_files[f] = mask_file_tiff
    print(f"Found {len(raw_files)} raw files and {len(mask_files)} corresponding mask files.")
    all_data = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for raw_file in tqdm(raw_files, desc="Processing Files"):
            mask_path = mask_files.get(raw_file)
            if mask_path is None:
                print(f"No corresponding mask file found for {raw_file}")
                all_data.append({
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
            future = executor.submit(
                process_single_tif,
                raw_path,
                mask_path,
                output_dir,
                scaling_factor,
                slice_scaling_percentile,
                pre_clipping_min_size,
                post_clipping_min_size,
                pre_clipping_min_mean_intensity,
                post_clipping_min_integrated_intensity,
                relabel,
                fill_holes,
                min_hole_diameter,
                skip_existing
            )
            futures.append(future)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Files Completed"):
            try:
                file_results = future.result()
                all_data.extend(file_results)
            except Exception as e:
                print(f"Error completing future for a file: {e}")
    csv_output_path = os.path.join(output_dir, "object_summary.csv")
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df = pd.DataFrame(all_data)
    df.to_csv(csv_output_path, index=False)
    print(f"CSV summary saved to {csv_output_path}")
    return csv_output_path

# ==================== Anisotropic Deblurring & Segmentation Functions ====================
def upsample_and_deblur_volume(volume, anisotropy, deblur_model_type="aniso_cyto2", diameter=30, z_axis=0, channels=[0,0]):
    """
    Upsamples an anisotropic 3D volume (ZYX) to approximate isotropic resolution and applies deblurring.
    """
    original_shape = volume.shape  # e.g., (Z, Y, X) or (Z, Y, X, 1)
    if len(original_shape) == 4 and original_shape[3] == 1:
        volume_proc = volume[..., 0]
        channel_present = True
    else:
        volume_proc = volume
        channel_present = False

    shape = volume_proc.shape
    new_shape = [int(shape[0] * anisotropy), shape[1], shape[2]]
    print(f"[DEBUG] Original volume shape: {shape}, new_shape: {new_shape}")
    vol_temp = transforms.resize_image(volume_proc.astype("float32").transpose(1, 0, 2),
                                         Ly=new_shape[0],
                                         Lx=new_shape[2],
                                         no_channels=True).transpose(1, 0, 2)
    print(f"[DEBUG] Shape after first resize: {vol_temp.shape}")
    vol_upsampled = transforms.resize_image(vol_temp,
                                              Ly=new_shape[1],
                                              Lx=new_shape[2],
                                              no_channels=True)
    print(f"[DEBUG] Shape after second resize (upsampled): {vol_upsampled.shape}")
    dn_model = denoise.DenoiseModel(model_type=deblur_model_type, gpu=True)
    img_iso1 = dn_model.eval(vol_upsampled.transpose(1, 0, 2),
                             diameter=diameter, z_axis=0, channels=channels)
    img_iso1 = img_iso1.squeeze().transpose(1, 0, 2)
    img_iso2 = dn_model.eval(vol_upsampled.transpose(2, 0, 1),
                             diameter=diameter, z_axis=0, channels=channels)
    img_iso2 = img_iso2.squeeze().transpose(1, 2, 0)
    img_iso = (img_iso1 + img_iso2) / 2.0
    print(f"[DEBUG] Deblurred volume shape before adding channel: {img_iso.shape}")
    if channel_present:
        img_iso = img_iso[..., np.newaxis]
        print(f"[DEBUG] Deblurred volume shape after adding channel: {img_iso.shape}")
    return img_iso

def segment_volume_from_array(model, volume, timepoints, image_name, output_path, 
                              channels=[0,0], diameter=60, flow_threshold=0.4, stitch_threshold=0.5,
                              cellprob_threshold=-2.0, do_3D=False, anisotropy=5.87,
                              z_axis=0, norm_params=None, min_size=500,
                              perform_anisotropic_deblur=False,
                              anisotropic_deblur_model="aniso_cyto2",
                              deblur_diameter=30,
                              save_upsampled=False):
    """
    Segments a 3D volume per timepoint using the Cellpose model.
    """
    T = volume.shape[0]
    for idx in range(T):
        tp = timepoints[idx]
        vol = volume[idx]  # (ZYX)
        print(f"[DEBUG] Segmenting {image_name} timepoint {tp}, volume shape: {vol.shape}")
        if perform_anisotropic_deblur:
            if vol.ndim == 4 and vol.shape[3] == 1:
                vol_proc = vol[..., 0]
            else:
                vol_proc = vol
            original_z = vol_proc.shape[0]
            print(f"[DEBUG] Original z-dimension: {original_z}")
            vol_deblur = upsample_and_deblur_volume(vol_proc, anisotropy, 
                                                    deblur_model_type=anisotropic_deblur_model, 
                                                    diameter=deblur_diameter, 
                                                    z_axis=z_axis, channels=channels)
            print(f"[DEBUG] Deblurred volume shape: {vol_deblur.shape}")
            if save_upsampled:
                upsample_folder = Path(output_path) / "Denoised_anisotropic_upsampled"
                upsample_folder.mkdir(parents=True, exist_ok=True)
                up_filename = upsample_folder / f"{image_name}_T{tp}_upsampled.tif"
                imwrite(str(up_filename), vol_deblur, compression='zlib')
                print(f"[DEBUG] Saved upsampled image: {up_filename}")
            if vol.ndim == 4 and vol.shape[3] == 1 and vol_deblur.ndim == 3:
                vol_deblur = vol_deblur[..., np.newaxis]
            vol = vol_deblur
        output_name = f"{image_name}_T{tp}_cp_masks.tif"
        # Use a new variable to avoid reassigning output_path.
        output_file_path = os.path.join(output_path, output_name)
        try:
            print(f"[DEBUG] Running segmentation for {image_name} timepoint {tp} on volume shape: {vol.shape}")
            masks, flows, styles = model.eval(
                [vol],
                channels=channels,
                diameter=diameter,
                flow_threshold=flow_threshold,
                stitch_threshold=stitch_threshold,
                cellprob_threshold=cellprob_threshold,
                do_3D=do_3D,
                anisotropy=anisotropy,
                z_axis=z_axis,
                normalize=(norm_params if norm_params is not None else False),
                min_size=min_size
            )
            mask_final = masks[0]
            print(f"[DEBUG] Mask shape before resizing: {mask_final.shape}")
            if perform_anisotropic_deblur:
                if mask_final.ndim == 3:
                    print(f"[DEBUG] Resizing mask from shape {mask_final.shape} back to original z-dim {original_z}")
                    mask_final = resize(mask_final,
                                        (original_z, mask_final.shape[1], mask_final.shape[2]),
                                        order=0,
                                        preserve_range=True,
                                        anti_aliasing=False).astype(mask_final.dtype)
                    print(f"[DEBUG] Mask shape after resizing: {mask_final.shape}")
                else:
                    print("[DEBUG] Skipping resize; mask is not 3D.")
            imwrite(output_file_path, mask_final, compression='zlib')
            print(f"[Segmentation] Processed {image_name} timepoint {tp}")
        except Exception as e:
            print(f"[Segmentation] Failed on {image_name} timepoint {tp}: {e}")

def run_segmentation_from_memory(denoised_dict, output_path, workers, seg_params):
    """
    Runs segmentation on in-memory denoised volumes.
    Saves segmentation masks to output_path/NucleiSegmentation.
    """
    seg_out = Path(output_path) / "NucleiSegmentation"
    seg_out.mkdir(parents=True, exist_ok=True)
    if seg_params['pretrained_model']:
        cp_model = cp_models.CellposeModel(
            gpu=True,
            pretrained_model=seg_params['pretrained_model'],
            model_type=None,
            pretrained_model_ortho=None,
            diam_mean=seg_params['diameter']
        )
    else:
        cp_model = cp_models.CellposeModel(
            gpu=True,
            model_type=seg_params['model_type'],
            diam_mean=seg_params['diameter']
        )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for image_name, (volume, timepoints) in denoised_dict.items():
            futures.append(executor.submit(
                segment_volume_from_array,
                cp_model,
                volume,
                timepoints,
                image_name,
                str(Path(output_path) / "NucleiSegmentation"),
                channels=seg_params['channels'],
                diameter=seg_params['diameter'],
                flow_threshold=seg_params['flow_threshold'],
                stitch_threshold=seg_params['stitch_threshold'],
                cellprob_threshold=seg_params['cellprob_threshold'],
                do_3D=seg_params['do_3D'],
                anisotropy=seg_params['anisotropy'],
                norm_params=seg_params['norm_params'],
                min_size=seg_params['min_size'],
                perform_anisotropic_deblur=seg_params.get('perform_anisotropic_deblur', False),
                anisotropic_deblur_model=seg_params.get('anisotropic_deblur_model', 'aniso_cyto2'),
                deblur_diameter=seg_params.get('deblur_diameter', 30),
                save_upsampled=seg_params.get('save_upsampled', False)
            ))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Segmenting images"):
            try:
                future.result()
            except Exception as e:
                print(f"[Segmentation] Error: {e}")
    print("[Segmentation] Processing complete.")

def load_denoised_images_from_disk(output_path):
    """
    Loads existing denoised images from output_path/Denoised.
    Returns a dictionary mapping image names to (stacked_volume, timepoints).
    """
    denoised_folder = Path(output_path) / "Denoised"
    if not denoised_folder.exists():
        raise ValueError(f"Denoised folder {denoised_folder} does not exist.")
    tif_files = list(denoised_folder.glob("*.tif"))
    results = {}
    for tif in tif_files:
        name = tif.stem  # e.g., "imageName_T0"
        if "_T" not in name:
            continue
        parts = name.rsplit("_T", 1)
        image_name = parts[0]
        try:
            timepoint = int(parts[1])
        except Exception as e:
            continue
        img = imread(str(tif))
        if image_name not in results:
            results[image_name] = []
        results[image_name].append((timepoint, img))
    denoised_dict = {}
    for image_name, lst in results.items():
        lst.sort(key=lambda x: x[0])
        timepoints = [t for t, _ in lst]
        stacked = np.stack([img for _, img in lst], axis=0)
        denoised_dict[image_name] = (stacked, timepoints)
    return denoised_dict

# ==================== Streamlit Interface ====================
def get_param(param_name, default):
    if param_source == "CSV Import" and param_name in params_csv_dict:
        val = params_csv_dict[param_name].strip()
        if isinstance(default, bool):
            return val.lower() in ("true", "1", "yes")
        if isinstance(default, dict):
            try:
                return ast.literal_eval(val)
            except Exception as e:
                return default
        try:
            return type(default)(val)
        except Exception as e:
            return default
    else:
        return default

def run_streamlit_app():
    st.title("BFS - Biofile Processing Script Workflow App")
    st.write("This application performs full workflow to process raw timelapse datasets using denoising, segmentation, and clipping to generate instance segmentation outputs.\nSupported formats: .zarr, .ome.zarr, .tif/.tiff, .czi.")

    global param_source, params_csv_dict, offset
    param_source = st.radio("Parameter Source", options=["Manual", "CSV Import"], key="param_source")
    params_csv_dict = {}
    if param_source == "CSV Import":
        param_file = st.file_uploader("Upload Parameter CSV", type=["csv"], key="param_csv_file")
        if param_file is not None:
            file_content = param_file.getvalue().decode("utf-8")
            csv_reader = csv.DictReader(stio.StringIO(file_content))
            for row in csv_reader:
                params_csv_dict[row["Parameter"]] = row["Value"]

    default_mode = get_param("Processing Mode", "Denoising Only")
    options = ["Denoising Only", "Denoising + Segmentation", "Denoising + Segmentation + Clipping"]
    process_mode = st.radio("Select Processing Mode", options=options, index=options.index(default_mode), key="process_mode")

    input_file_type = st.radio("Input File Type", options=["Single S3 Path", "CSV File of S3 Paths"], index=0, key="input_file_type")
    if input_file_type == "CSV File of S3 Paths":
        csv_input_file = st.file_uploader("Upload CSV file containing S3 Paths", type=["csv"], key="input_csv")
        if csv_input_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(csv_input_file.getvalue())
                tmp.flush()
                input_path_value = tmp.name
        else:
            st.error("Please upload a CSV file with S3 paths.")
            return
    else:
        input_path_value = st.text_input("Input Path", value=get_param("Input Path", "s3://allencell/aics/emt_timelapse_dataset/data/3500006256_11_raw_converted.ome.zarr"), key="input_path")

    output_path = st.text_input("Output Path", value=get_param("Output Path", "D:\\S3_Omezarr_DenoisePerTimepointScript1_Output"), key="output_path")
    timepoint_range = st.text_input("Timepoint Range (e.g., 1-2 or leave blank for all)", value=get_param("Timepoint Range", "1-2"), key="timepoint_range")

    with st.expander("Denoising Parameters", expanded=True):
        max_workers = st.number_input("Max Workers (Denoising)", min_value=1, value=get_param("Max Workers (Denoising)", 4), step=1, key="denoise_max_workers")
        file_workers = st.number_input("File Workers (Denoising)", min_value=1, value=get_param("File Workers (Denoising)", 2), step=1, key="denoise_file_workers")
        channel = st.number_input("Channel (0=Brightfield, 1=EGFP, 2=CollagenIV Ab)", min_value=0, value=get_param("Channel", 1), step=1, key="denoise_channel")
        offset = st.number_input("Enter black reference pixel intensity offset for image subtraction", value=100, step=1, key="offset")
        denoise_method = st.radio("Denoising Method", options=["Dynamic Rescaling", "Constant Rescaling"],
                                  index=["Dynamic Rescaling", "Constant Rescaling"].index(get_param("Denoising Method", "Dynamic Rescaling")), key="denoise_method")
        save_raw = st.checkbox("Save Raw Image", value=get_param("Save Raw Image", False), key="save_raw")
    if denoise_method == "Constant Rescaling":
        with st.expander("Constant Rescaling Parameters", expanded=True):
            raw_max = st.number_input("Raw Max", value=get_param("Raw Max", 2000), key="const_raw_max")
            p99 = st.number_input("99th Percentile", value=get_param("Percentile 99", 8000), key="const_p99")
            denoised_min = st.number_input("Denoised Min", value=get_param("Denoised Min", 0), key="const_denoised_min")
            denoised_max = st.number_input("Denoised Max", value=get_param("Denoised Max", 4), key="const_denoised_max")
            const_params = {
                'raw_max': raw_max,
                'p99': p99,
                'denoised_min': denoised_min,
                'denoised_max': denoised_max
            }
    else:
        const_params = {}

    if param_source == "CSV Import":
        seg_workers = get_param("Max Workers (Segmentation)", 4)
    else:
        seg_workers = st.number_input("Max Workers (Segmentation)", min_value=1, value=4, step=1, key="seg_max_workers")

    seg_params = {}
    use_existing_denoised = False
    use_existing_seg_masks = False
    if process_mode in ["Denoising + Segmentation", "Denoising + Segmentation + Clipping"]:
        with st.expander("Segmentation Parameters", expanded=True):
            seg_channels_str = get_param("Segmentation Channels", "[0, 0]").strip()
            if seg_channels_str.startswith("[") and seg_channels_str.endswith("]"):
                seg_channels_str = seg_channels_str[1:-1]
            seg_channels = [int(x.strip()) for x in seg_channels_str.split(",") if x.strip()]
            seg_diameter = st.number_input("Segmentation Diameter", min_value=1, value=get_param("Segmentation Diameter", 40), step=1, key="seg_diameter")
            seg_flow_threshold = st.number_input("Flow Threshold", value=get_param("Flow Threshold", 0.4), format="%.2f", key="seg_flow_threshold")
            seg_stitch_threshold = st.number_input("Stitch Threshold", value=get_param("Stitch Threshold", 0.5), format="%.2f", key="seg_stitch_threshold")
            seg_cellprob_threshold = st.number_input("Cell Probability Threshold", value=get_param("Cell Probability Threshold", -2.0), format="%.2f", key="seg_cellprob_threshold")
            seg_do_3D = st.checkbox("3D Processing", value=get_param("3D Processing", False), key="seg_do_3D")
            seg_anisotropy = st.number_input("Segmentation Anisotropy", value=get_param("Segmentation Anisotropy", 5.87), format="%.2f", key="seg_anisotropy")
            seg_model_type = st.text_input("Segmentation Model Type", value=get_param("Segmentation Model Type", "cyto3"), key="seg_model_type")
            seg_pretrained_model = st.text_input("Segmentation Pretrained Model", value=get_param("Segmentation Pretrained Model", r"C:\scripts\BFS_segmentation_workflow\models\cellpose_1733942372.5497868"), key="seg_pretrained_model")
            perform_aniso = st.checkbox("Perform anisotropic deblurring and upsampling", value=get_param("Perform anisotropic deblurring and upsampling", False), key="seg_perform_aniso")
            aniso_deblur_model = st.selectbox("Select anisotropic deblur model", options=["aniso_cyto2", "aniso_nuclei"],
                                              index=["aniso_cyto2", "aniso_nuclei"].index(get_param("Select anisotropic deblur model", "aniso_cyto2")), key="seg_aniso_deblur_model")
            deblur_diam = st.number_input("Deblur Diameter", value=get_param("Deblur Diameter", 30), step=1, key="seg_deblur_diam")
            save_upsampled = st.checkbox("Save deblurred/upsampled image", value=get_param("Save deblurred/upsampled image", False), key="seg_save_upsampled")
            norm_params = get_param("Normalization Parameters", {'lowhigh': None, 'percentile': [1.0, 99.0], 'normalize': True, 'norm3D': True, 'sharpen_radius': 0, 'smooth_radius': 0, 'tile_norm_blocksize': 0, 'tile_norm_smooth3D': 1, 'invert': False})
            seg_params = {
                "channels": seg_channels,
                "diameter": seg_diameter,
                "flow_threshold": seg_flow_threshold,
                "stitch_threshold": seg_stitch_threshold,
                "cellprob_threshold": seg_cellprob_threshold,
                "do_3D": seg_do_3D,
                "anisotropy": seg_anisotropy,
                "model_type": seg_model_type,
                "pretrained_model": seg_pretrained_model,
                "norm_params": norm_params,
                "perform_anisotropic_deblur": perform_aniso,
                "anisotropic_deblur_model": aniso_deblur_model,
                "deblur_diameter": deblur_diam,
                "save_upsampled": save_upsampled,
                "min_size": st.number_input("Min Size", value=get_param("Segmentation Min Size", 15), min_value=1, step=1, key="seg_min_size")
            }
            use_existing_denoised = st.checkbox("Use existing denoised images for segmentation", value=get_param("Use Existing Denoised for Segmentation", False), key="use_existing_denoised")
            use_existing_seg_masks = st.checkbox("Use existing segmentation mask images for clipping", value=get_param("Use Existing Segmentation Masks for Clipping", False), key="use_existing_seg_masks")
    else:
        process_mode = "Denoising Only"

    clip_params = {}
    if process_mode == "Denoising + Segmentation + Clipping":
        with st.expander("Clipping Parameters", expanded=True):
            clip_raw_dir = st.text_input("Clipping Raw Directory", value=get_param("Clipping Raw Directory", str(Path(output_path) / "Denoised")), help="Directory containing denoised images for clipping.", key="clip_raw_dir")
            clip_mask_dir = st.text_input("Clipping Mask Directory", value=get_param("Clipping Mask Directory", str(Path(output_path) / "NucleiSegmentation")), help="Directory containing segmentation masks.", key="clip_mask_dir")
            clip_output_dir = st.text_input("Clipping Output Directory", value=get_param("Clipping Output Directory", str(Path(output_path) / "ClippedMask")), help="Directory to save clipped masks.", key="clip_output_dir")
            clip_scaling_factor = st.number_input("XY Clipping Percentile", value=get_param("Clipping XY Percentile", 55.0), format="%.2f", key="clip_xy")
            clip_slice_percentile = st.number_input("Z Slice Clipping Percentile", value=get_param("Clipping Z Percentile", 65.0), format="%.2f", key="clip_z")
            clip_workers = st.number_input("Clipping Workers", min_value=1, value=get_param("Clipping Workers", 2), step=1, key="clip_workers")
            clip_relabel = st.checkbox("Relabel after clipping", value=get_param("Clipping Relabel", False), key="clip_relabel")
            clip_fill_holes = st.checkbox("Fill holes", value=get_param("Clipping Fill Holes", False), key="clip_fill_holes")
            clip_pre_min_size = st.number_input("Pre-clipping Min Size", value=get_param("Clipping Pre-min Size", 0), min_value=0, step=1, key="clip_pre_min")
            clip_post_min_size = st.number_input("Post-clipping Min Size", value=get_param("Clipping Post-min Size", 0), min_value=0, step=1, key="clip_post_min")
            clip_pre_min_mean = st.number_input("Pre-clipping Min Mean Intensity", value=get_param("Clipping Pre-min Mean Intensity", 0.0), format="%.1f", key="clip_pre_mean")
            clip_post_min_integrated_intensity = st.number_input("Post-clipping Min Integrated Intensity", value=get_param("Clipping Post-min Integrated Intensity", 0.0), format="%.1f", key="clip_post_int")
            clip_min_hole_diameter = st.number_input("Clipping Min Hole Diameter", value=get_param("Clipping Min Hole Diameter", 64.0), format="%.1f", key="clip_min_diam")
            skip_existing = st.checkbox("Skip processing if output file exists", value=get_param("Skip Existing Clipping Files", False), key="clip_skip")
            clip_params = {
                "raw_dir": clip_raw_dir,
                "mask_dir": clip_mask_dir,
                "output_dir": clip_output_dir,
                "scaling_factor": clip_scaling_factor,
                "slice_scaling_percentile": clip_slice_percentile,
                "workers": clip_workers,
                "relabel": clip_relabel,
                "fill_holes": clip_fill_holes,
                "pre_clipping_min_size": clip_pre_min_size,
                "post_clipping_min_size": clip_post_min_size,
                "pre_clipping_min_mean_intensity": clip_pre_min_mean,
                "post_clipping_min_integrated_intensity": clip_post_min_integrated_intensity,
                "min_hole_diameter": clip_min_hole_diameter,
                "skip_existing": skip_existing
            }

    if st.button("Run Processing", key="run_processing"):
        if not input_path_value or not output_path:
            st.error("Please provide both input and output paths.")
        else:
            st.info("Starting processing...")
            with st.spinner("Processing... This may take a while."):
                try:
                    # ------------------ Step 1: Denoising ------------------
                    if use_existing_denoised:
                        st.info("Using existing denoised images...")
                        denoised_dict = load_denoised_images_from_disk(output_path)
                    else:
                        if denoise_method == "Dynamic Rescaling":
                            denoised_dict = run_denoising(input_path_value, output_path, int(max_workers),
                                                          int(file_workers), int(channel), timepoint_range, save_raw)
                        else:
                            denoised_dict = constant_denoise_directory(input_path_value, output_path, 
                                                                       {'gpu': True, 'model_type': 'denoise_cyto', 'nchan': 1, 'chan2': False},
                                                                       {'normalize': True, 'norm3D': True},
                                                                       int(max_workers), "constant", const_params, timepoint_range, save_raw)
                    
                    # ------------------ Step 2: Segmentation ------------------
                    if process_mode in ["Denoising + Segmentation", "Denoising + Segmentation + Clipping"]:
                        if use_existing_denoised:
                            denoised_dict = load_denoised_images_from_disk(output_path)
                        if use_existing_seg_masks:
                            st.info("Using existing segmentation mask images; skipping segmentation.")
                        else:
                            run_segmentation_from_memory(denoised_dict, output_path, int(seg_workers), seg_params)
                    
                    # ------------------ Step 3: Clipping ------------------
                    if process_mode == "Denoising + Segmentation + Clipping":
                        process_directory(raw_dir=str(Path(output_path) / "Denoised"),
                                          mask_dir=clip_params["mask_dir"],
                                          output_dir=clip_params["output_dir"],
                                          scaling_factor=clip_params["scaling_factor"],
                                          slice_scaling_percentile=clip_params["slice_scaling_percentile"],
                                          pre_clipping_min_size=clip_params["pre_clipping_min_size"],
                                          post_clipping_min_size=clip_params["post_clipping_min_size"],
                                          pre_clipping_min_mean_intensity=clip_params["pre_clipping_min_mean_intensity"],
                                          post_clipping_min_integrated_intensity=clip_params["post_clipping_min_integrated_intensity"],
                                          workers=clip_params["workers"],
                                          relabel=clip_params["relabel"],
                                          fill_holes=clip_params["fill_holes"],
                                          min_hole_diameter=clip_params["min_hole_diameter"],
                                          skip_existing=clip_params["skip_existing"])
                    st.success("Processing completed successfully!")
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    if seg_params.get("pretrained_model", "").strip():
                        seg_model_display = ""
                    else:
                        seg_model_display = seg_params.get("model_type", "")
                    params = {
                        "Input Path": input_path_value,
                        "Output Path": output_path,
                        "Timepoint Range": timepoint_range,
                        "Max Workers (Denoising)": max_workers,
                        "File Workers (Denoising)": file_workers,
                        "Channel": channel,
                        "Processing Mode": process_mode,
                        "Denoising Method": denoise_method,
                    }
                    if process_mode in ["Denoising + Segmentation", "Denoising + Segmentation + Clipping"]:
                        params.update({
                            "Segmentation Channels": seg_params["channels"],
                            "Segmentation Diameter": seg_params["diameter"],
                            "Segmentation Flow Threshold": seg_params["flow_threshold"],
                            "Segmentation Stitch Threshold": seg_params["stitch_threshold"],
                            "Segmentation CellProb Threshold": seg_params["cellprob_threshold"],
                            "Segmentation 3D": seg_params["do_3D"],
                            "Segmentation Anisotropy": seg_params["anisotropy"],
                            "Segmentation Min Size": seg_params["min_size"],
                            "Segmentation Model Type": seg_model_display,
                            "Segmentation Pretrained Model": seg_params["pretrained_model"],
                            "Normalization Parameters": seg_params["norm_params"],
                            "Perform Anisotropic Deblur": seg_params["perform_anisotropic_deblur"],
                            "Anisotropic Deblur Model": seg_params["anisotropic_deblur_model"],
                            "Deblur Diameter": seg_params["deblur_diameter"],
                            "Save Upsampled": seg_params["save_upsampled"],
                            "Use Existing Denoised for Segmentation": use_existing_denoised,
                            "Use Existing Segmentation Masks for Clipping": use_existing_seg_masks,
                        })
                    if process_mode == "Denoising + Segmentation + Clipping":
                        params.update({
                            "Clipping Raw Directory": str(Path(output_path) / "Denoised"),
                            "Clipping Mask Directory": clip_params["mask_dir"],
                            "Clipping Output Directory": clip_params["output_dir"],
                            "Clipping XY Percentile": clip_params["scaling_factor"],
                            "Clipping Z Percentile": clip_params["slice_scaling_percentile"],
                            "Clipping Workers": clip_params["workers"],
                            "Clipping Relabel": clip_params["relabel"],
                            "Clipping Fill Holes": clip_params["fill_holes"],
                            "Clipping Pre-min Size": clip_params["pre_clipping_min_size"],
                            "Clipping Post-min Size": clip_params["post_clipping_min_size"],
                            "Clipping Pre-min Mean Intensity": clip_params["pre_clipping_min_mean_intensity"],
                            "Clipping Post-min Integrated Intensity": clip_params["post_clipping_min_integrated_intensity"],
                            "Clipping Min Hole Diameter": clip_params["min_hole_diameter"],
                            "Skip Existing Clipping Files": clip_params["skip_existing"],
                        })
                    params_csv = Path(output_path) / f"processing_parameters_{timestamp}.csv"
                    with open(params_csv, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["Parameter", "Value"])
                        for key, value in params.items():
                            writer.writerow([key, value])
                    st.success(f"Parameters saved to {params_csv}")
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # Streamlit UI to Launch Napari
    st.subheader("Load Processed Results in Napari")
    st.write("Click the button below after processing to launch Napari and load all subfolders (e.g., Raw, Denoised, NucleiSegmentation, ClippedMask) as separate layers.")
    #output_path_str = st.text_input("Output Path for Napari", value="D:\\S3_Omezarr_DenoisePerTimepointScript1_Output", key="napari_output_path")
    output_path_str = st.text_input("Output Path for Napari", value=output_path, key="napari_output_path")
    if st.button("Launch Napari Viewer (All Subfolders)", key="launch_napari_btn"):
        launch_napari_all_subfolders(Path(output_path_str))

# ==================== Napari Launcher Functions (Module Level) ====================

from napari.utils.colormaps import DirectLabelColormap
from matplotlib import colormaps

def launch_napari_all_subfolders(output_path):
    # Ensure output_path is converted to a string for subprocess
    subprocess.Popen([sys.executable, __file__, "viewer", str(output_path)])

def get_unique_positions(directory):
    """
    Scans the given directory and extracts unique positions from filenames.
    """
    positions = set()
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(('.tif', '.tiff')):
                m = re.match(r'^(.*?)_T', entry.name)
                if m:
                    positions.add(m.group(1))
                else:
                    positions.add(os.path.splitext(entry.name)[0])
    sorted_positions = sorted(positions)
    print(f"[INFO] Found {len(sorted_positions)} unique positions: {sorted_positions}")
    return sorted_positions

def load_layer_images(folder_path, position_id):
    """
    Loads images from the specified folder whose filenames start with the given position_id.
    """
    folder = Path(folder_path)
    if not folder.exists():
        return None
    matching_files = [f for f in os.listdir(folder)
                      if f.startswith(position_id) and f.lower().endswith(('.tif', '.tiff'))]
    def extract_timepoint(fname):
        m = re.search(r'_T(\d+)', fname)
        return int(m.group(1)) if m else 0
    matching_files.sort(key=extract_timepoint)
    images = []
    for f in matching_files:
        img = imread(str(folder / f))
        images.append(img)
    if images:
        return np.stack(images, axis=0) if len(images) > 1 else images[0]
    return None

def update_layers(output_path, position_id, viewer, generate_multiscale):
    """
    Updates viewer layers based on the selected position.
    Converts labels layers to an integer type.
    """
    target_layers = ["Denoised", "NucleiSegmentation", "ClippedMask", "Raw"]
    for layer_name in target_layers:
        folder = Path(output_path) / layer_name
        data = load_layer_images(folder, position_id)
        if data is None:
            print(f"No data found for layer '{layer_name}' at position '{position_id}'.")
            continue
        if data.ndim == 4:
            axes = ("Time", "Z", "Y", "X")
            scale = (1, 2.0, 0.274, 0.274)
        elif data.ndim == 3:
            axes = ("Z", "Y", "X")
            scale = (2.0, 0.274, 0.274)
        else:
            axes = None
            scale = None
        if generate_multiscale is not None:
            pyramid = generate_multiscale(data, downscale=2)
            multiscale = True
            data_to_use = pyramid
        else:
            multiscale = False
            data_to_use = data
        # If this is a labels layer, cast data to an integer type.
        if layer_name.lower() in {"clippedmask", "nucleisegmentation", "nuclei_segmentation"}:
            if data_to_use.dtype.kind != 'i':
                data_to_use = data_to_use.astype(np.int32)
        layer = viewer.layers[layer_name]
        layer.data = data_to_use
        layer.axis_labels = axes
        layer.scale = scale
        layer.multiscale = multiscale
        layer._update_thumbnail = lambda: None
        if layer_name.lower() in {"clippedmask", "nucleisegmentation", "nuclei_segmentation"}:
            layer.blending = 'additive'
            layer.depiction = 'volume'
            layer.multiscale = multiscale
            layer.opacity = 0.5
            layer.rendering = 'translucent'
        else:
            lower_folder = layer_name.lower().strip()
            if "raw" in lower_folder:
                contrast_limits = [100, 2000]
            elif "denoised" in lower_folder:
                contrast_limits = [1500, 65535]
            else:
                contrast_limits = [0, 65535]
            layer.blending = 'additive'
            layer.contrast_limits = contrast_limits
            layer.depiction = 'volume'
            layer.multiscale = multiscale
            layer.rendering = 'mip'
        print(f"Layer '{layer_name}' updated for position '{position_id}'.")

def napari_viewer_process(output_path):
    """
    Launches Napari in 3D mode and loads all subfolders as layers.
    Converts labels layers to integer type.
    """
    import napari
    try:
        from napari.utils.multiscale import generate_multiscale
    except ImportError:
        generate_multiscale = None

    viewer = napari.Viewer(ndisplay=3)
    target_layers = ["Raw", "Denoised", "NucleiSegmentation", "ClippedMask"]
    layer_dict = {}
    for layer_name in target_layers:
        folder = Path(output_path) / layer_name
        data = None
        if folder.exists():
            positions_in_folder = get_unique_positions(str(folder))
            if positions_in_folder:
                init_pos = positions_in_folder[0]
                data = load_layer_images(folder, init_pos)
        if data is None:
            data = np.zeros((1,1,1))
        if layer_name.lower() in {"clippedmask", "nucleisegmentation"}:
            # Ensure the data is of integer type
            if data.dtype.kind != 'i':
                data = data.astype(np.int32)
            axes = ("Time", "Z", "Y", "X") if data.ndim == 4 else ("Z", "Y", "X")
            scale = (1, 2.0, 0.274, 0.274) if data.ndim == 4 else (2.0, 0.274, 0.274)
            if generate_multiscale is not None:
                pyramid = generate_multiscale(data, downscale=2)
                multiscale = True
                data_to_use = pyramid
            else:
                multiscale = False
                data_to_use = data
            layer = viewer.add_labels(
                data_to_use,
                name=layer_name,
                colormap=DirectLabelColormap(colors=[tuple(colormaps['turbo'](i)) for i in np.linspace(0,1,256)]),
                axis_labels=axes,
                blending='additive',
                depiction='volume',
                multiscale=multiscale,
                opacity=0.5,
                rendering='translucent',
                scale=scale,
                visible=True,
            )
        else:
            axes = ("Time", "Z", "Y", "X") if data.ndim == 4 else ("Z", "Y", "X")
            scale = (1, 2.0, 0.274, 0.274) if data.ndim == 4 else (2.0, 0.274, 0.274)
            if generate_multiscale is not None:
                pyramid = generate_multiscale(data, downscale=2)
                multiscale = True
                data_to_use = pyramid
            else:
                multiscale = False
                data_to_use = data
            layer = viewer.add_image(
                data_to_use,
                name=layer_name,
                colormap='gray',
                axis_labels=axes,
                blending='additive',
                contrast_limits=[100,2000] if "raw" in layer_name.lower() else ([1500,65535] if "denoised" in layer_name.lower() else [0,65535]),
                depiction='volume',
                multiscale=multiscale,
                scale=scale,
                rendering='mip',
                visible=True,
            )
        layer._update_thumbnail = lambda: None
        layer_dict[layer_name] = layer

    denoised_dir = os.path.join(output_path, "Denoised")
    positions = get_unique_positions(denoised_dir) if os.path.exists(denoised_dir) else []

    from magicgui import magicgui
    @magicgui(call_button="Load Position", position={"widget_type": "ComboBox", "choices": positions})
    def position_list(position: str):
        update_layers(output_path, position, viewer, generate_multiscale)
        print(f"Position '{position}' loaded.")

    annotations = []
    @magicgui(call_button="Record Annotation", layout='vertical')
    def annotation_widget(
        annotator: str,
        object_id: str,
        fail_fov: bool,
        good: bool,
        merge: bool,
        split: bool,
        grows_into_noise: bool,
        comments: str
    ):
        if not annotator:
            print("Annotator name is required.")
            return
        if not object_id:
            print("Object ID is required.")
            return
        pos = position_list.position.value if hasattr(position_list, "position") else "Unknown"
        current_timepoint = viewer.dims.current_step[0]
        z_position = viewer.dims.current_step[1]
        annotations.append([
            pos,
            object_id,
            annotator,
            'Yes' if fail_fov else 'No',
            'Yes' if good else 'No',
            'Yes' if merge else 'No',
            'Yes' if split else 'No',
            'Yes' if grows_into_noise else 'No',
            comments,
            current_timepoint,
            z_position,
            "",
            "",
            viewer.layers.selection.active.name if viewer.layers.selection.active else 'None'
        ])
        annotation_widget.object_id.value = ""
        annotation_widget.fail_fov.value = False
        annotation_widget.good.value = False
        annotation_widget.merge.value = False
        annotation_widget.split.value = False
        annotation_widget.grows_into_noise.value = False
        annotation_widget.comments.value = ""
        print("Annotation recorded.")

    @magicgui(call_button="Save Annotations to CSV")
    def save_annotations_button():
        if annotations:
            annotator = annotations[-1][2]
            os.makedirs(output_path, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_path, f"annotations_{annotator}_{timestamp}.csv")
            headers = [
                'Filename', 'Object ID', 'Annotator', 'Fail FOV', 'Good', 'Merge',
                'Split', 'Grows into Noise', 'Comments', 'Timepoint', 'Z Position',
                'Max 3D Object ID', 'Max 2D Object ID', 'Selected Layer'
            ]
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(annotations)
            print(f"Annotations saved to CSV in {output_path}")
        else:
            print("No annotations to save.")

    from napari.layers import Labels
    def update_object_id(layer):
        object_id = layer.selected_label
        annotation_widget.object_id.value = str(object_id)
    for layer in viewer.layers:
        if isinstance(layer, Labels):
            layer.events.selected_label.connect(lambda event, layer=layer: update_object_id(layer))

    viewer.window.add_dock_widget(position_list, area='left')
    viewer.window.add_dock_widget(annotation_widget, area='right')
    viewer.window.add_dock_widget(save_annotations_button, area='right')

    if positions:
        init_pos = positions[0]
        update_layers(output_path, init_pos, viewer, generate_multiscale)
        position_list.position.value = init_pos

    napari.run()

# ==================== Main ====================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "viewer":
        # In viewer mode, the second argument is the output path.
        napari_viewer_process(sys.argv[2])
    else:
        run_streamlit_app()
