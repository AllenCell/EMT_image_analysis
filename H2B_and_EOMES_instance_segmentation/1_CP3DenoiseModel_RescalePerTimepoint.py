"""
Multithreaded Cellpose3 Image Denoising Script

This script denoises images in a specified input directory and saves the processed images into an output directory.
It leverages the Cellpose3 DenoiseModel for denoising and uses multithreading to process multiple images concurrently,
significantly speeding up the processing time for large sets of images.  The script also converts the denoised image float32 datatype returned from DenoiseModel class to Uint16 format before saving.

This script is designed to be used with command-line arguments specifying the number of parallel workers, input path, and output path directories.

Dependencies:
- numpy
- tifffile
- cellpose
- concurrent.futures
- fire
- tqdm
- pathlib
- glob

Functions:
- ConvertFloatToUint16: Converts floating point image arrays to uint16, scaling intensity values.
- load_image: Loads an image from a given path into a numpy array.
- save_image: Saves an image numpy array to a specified path, converting to uint16 if necessary.
- process_image: Processes a single image by loading, denoising, and saving it.
- denoise_directory: Denoises all images in the specified input directory and saves them to the output directory.
- main: The main function that initializes model parameters and starts the denoising process.

Usage:
To run the script, you need to have Python 3.11 installed on your system along with the required dependencies listed above.

Install dependencies in conda env:
pip install numpy tifffile cellpose==3.0.8 fire tqdm pathlib glob


python CP3DenoiseModel_Multithreaded_tqdm_GlobalPercentile.py --input_path="//allen/aics/microscopy/H2B_Denoised/NewQCPositions/RawNewPositions/Z_Stacks"  --output_path="//allen/aics/microscopy/H2B_Denoised/NewQCPositions/RawNewPositions/DenoisePerTimepoint" --max_workers=8

"""
import os
import numpy as np
import fire
import csv
from pathlib import Path
from tifffile import imread, imwrite
from glob import glob
from cellpose.denoise import DenoiseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def load_image(image_path):
    return imread(image_path)

def save_image(image, output_path, image_name):
    image = np.squeeze(image)
    metadata = {'axes': 'ZYX'}
    image_uint16 = ConvertFloatToUint16(image)
    imwrite(str(output_path), image_uint16, photometric='minisblack', metadata=metadata)

def ConvertFloatToUint16(img):
    minimum_val = img.min()
    maximum_val = img.max()
    scale = 65535.0 / (maximum_val - minimum_val)
    img = (img - minimum_val) * scale
    img[img < 0.0] = 0.0
    img[img > 65535] = 65535
    
    return img.astype(np.uint16)

def RescaleInputImage(image_offset_corrected, raw_min, raw_max):
    minimum_val = image_offset_corrected.min()
    maximum_val = image_offset_corrected.max()
    scale = 65535.0 / (maximum_val - minimum_val)
    image_rescaled = (image_offset_corrected - minimum_val) * scale
    image_rescaled[image_rescaled < 0.0] = 0.0
    image_rescaled[image_rescaled > 65535] = 65535
    
    return image_rescaled.astype(np.uint16), scale

def process_image(image_path, output_dir, denoise_model, eval_params, z_axis=None, scale_log=None):
    base_name = os.path.basename(image_path)
    image_name = ".".join(base_name.split('.')[:-1])
    print(f"Processing {image_name}")
    image = load_image(image_path)
    
    raw_min = image.min()
    raw_max = image.max()
    
    camera_offset = 100

    image_offset_corrected = image.astype(np.float32) - camera_offset
    image_offset_corrected[image_offset_corrected < 0] = 0

    image_rescaled, scale = RescaleInputImage(image_offset_corrected, raw_min, raw_max)

    percentile_1 = np.percentile(image_rescaled, 1)
    percentile_99 = np.percentile(image_rescaled, 99)

    local_eval_params = eval_params.copy()
    local_eval_params['lowhigh'] = [percentile_1, percentile_99]

    denoised_image = denoise_model.eval(x=image_rescaled, channels=[0, 0], tile=False, z_axis=z_axis, normalize=local_eval_params)

    denoised_filename = Path(output_dir) / f"{image_name}.tif"
    save_image(denoised_image, denoised_filename, image_name)

    scale_log.append([image_name, scale, raw_min, raw_max, percentile_1, percentile_99])

def write_scale_log(output_dir, scale_log):
    rescale_scale_dir = Path(output_dir) / "RescaleScale"
    rescale_scale_dir.mkdir(parents=True, exist_ok=True)
    csv_file = rescale_scale_dir / "scale_values.csv"
    
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Rescale Factor", "Raw Min", "Raw Max", "Percentile 1", "Percentile 99"])
        writer.writerows(scale_log)

def denoise_directory(input_dir, output_dir, model_params, eval_params, max_workers=4):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scale_log = []
    
    z_axis = model_params.pop('z_axis', None)
    denoise_model = DenoiseModel(**model_params)
    image_paths = glob(os.path.join(input_dir, '*.tiff'))  # Ensure correct file extension

    if not image_paths:
        print(f"No images found in {input_dir}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [executor.submit(process_image, image_path, output_dir, denoise_model, eval_params, z_axis, scale_log) for image_path in image_paths]
        for task in tqdm(as_completed(tasks), total=len(tasks)):
            _ = task.result()

    write_scale_log(output_dir, scale_log)

def main(input_path, output_path, max_workers=4, **kwargs):
    model_params = {
        'gpu': True,
        'model_type': 'denoise_cyto',
        'nchan': 1,
        'chan2': False,
        'z_axis': 0,
    }

    eval_params = {
        'normalize': True,
        'norm3D': True,
    }

    denoise_directory(input_path, output_path, model_params, eval_params, max_workers)

if __name__ == "__main__":
    fire.Fire(main)
