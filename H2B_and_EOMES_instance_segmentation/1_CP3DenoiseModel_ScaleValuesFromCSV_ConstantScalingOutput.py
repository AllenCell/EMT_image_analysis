

#python denoise_ScaleValuesFromCSV_ConstantDenoiseMinMaxScalingOutput.py --input_path="//allen/aics/microscopy/H2B_Denoised/Raw/Missing"  --output_path="//allen/aics/microscopy/H2B_Denoised/Raw/Missing/CP3_Denoised_PerStructureNormCSV_MaxInt_2000_99P_8000_FixedRescale" --max_workers=8

#python denoise_ScaleValuesFromCSV_ConstantDenoiseMinMaxScalingOutput.py --input_path="//allen/aics/microscopy/EOMES_Denoised/Raw/Missing"  --output_path="//allen/aics/microscopy/EOMES_Denoised/Raw/Missing/Denoised_SeqRescale_PerStructureNormCSV_MaxInt_240_99P_21140" --max_workers=8


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
    print(f"Loading image: {image_path}")
    return imread(image_path)

def save_image(image, output_path, denoised_min, denoised_max):
    image = np.squeeze(image)
    image_uint16 = ConvertFloatToUint16(image, denoised_min, denoised_max)
    print(f"Saving image to: {output_path}")
    imwrite(output_path, image_uint16, photometric='minisblack')

def load_image_attributes(output_path):
    attributes_path = Path(output_path) / "input_scale_values.csv"
    print(f"Loading image attributes from: {attributes_path}")
    image_attributes = {}
    with open(attributes_path, 'r', encoding='utf-8-sig') as infile:
        dict_reader = csv.DictReader(infile)
        for row in dict_reader:
            image_name = row['Image Name'].strip()
            raw_max = float(row['Raw Max'])
            p99 = float(row['Percentile 99'])
            denoised_min = float(row['Denoised Min'])
            denoised_max = float(row['Denoised Max'])
            image_attributes[image_name] = (raw_max, p99, denoised_min, denoised_max)
    return image_attributes


def ConvertFloatToUint16(denoised_image, denoised_min, denoised_max):
    """Convert denoised float image to uint16, adjusting scale based on denoised min and max values from csv."""
    if denoised_min < 0:
        offset = -denoised_min
    else:
        offset = 0

    # Adjust minimum and maximum values based on the offset
    adjusted_denoised_min = denoised_min + offset
    adjusted_denoised_max = denoised_max + offset

    # Calculate scale and apply to denoised_image
    scale = 65535.0 / (adjusted_denoised_max - adjusted_denoised_min)
    denoised_image_scaled = (denoised_image + offset) * scale

    # Ensure values are within uint16 range
    denoised_image_scaled = np.clip(denoised_image_scaled, 0, 65535)
    return denoised_image_scaled.astype(np.uint16)
  

def rescale_input_image(image_offset_corrected, raw_max, p99=None):
    maximum_val = raw_max
    scale = 65535.0 / maximum_val
    image_rescaled = image_offset_corrected * scale
    image_rescaled = np.clip(image_rescaled, 0, 65535)
    return image_rescaled.astype(np.uint16), scale

def process_image(image_path, output_path, denoise_model, eval_params, image_attributes, scale_log):
    base_name = os.path.basename(image_path)
    image_name = ".".join(base_name.split('.')[:-1])
    print(f"Processing {image_name}")

    if image_name not in image_attributes:
        print(f"Warning: {image_name} does not have attributes specified in 'input_scale_values.csv'. Skipping.")
        return

    raw_max, p99, denoised_min, denoised_max = image_attributes[image_name]
    image = load_image(image_path)
    image_offset_corrected = image.astype(np.float32) - 100
    image_offset_corrected[image_offset_corrected < 0] = 0

    image_rescaled, scale = rescale_input_image(image_offset_corrected, raw_max, p99)

    local_eval_params = eval_params.copy()
    local_eval_params['lowhigh'] = [0, p99]
    denoised_image = denoise_model.eval(x=image_rescaled, channels=[0, 0], tile=False, z_axis=0, normalize=local_eval_params)

    denoised_filename = Path(output_path) / f"{image_name}.tif"
    save_image(denoised_image, denoised_filename, denoised_min, denoised_max)

    # Calculate metrics directly from denoised_image to trend with seg accuracy 
    min_intensity = denoised_image.min()
    median_intensity = np.median(denoised_image)
    mean_intensity = np.mean(denoised_image)
    max_intensity = denoised_image.max()
    max_intensity_count = np.sum(denoised_image == 65535)
    percentile_99 = np.percentile(denoised_image, 99)

    scale_log.append({
        'Image Name': image_name,
        'Scale Factor': scale,
        'Raw Input Max': raw_max,
        'Percentile 99 Used': p99,
        'Min Intensity': min_intensity,
        'Median Intensity': median_intensity,
        'Mean Intensity': mean_intensity,
        'Max Intensity': max_intensity,
        'Max Intensity Count': max_intensity_count,
        'Percentile 99': percentile_99
    })

def write_scale_log(output_path, scale_log):
    scale_log_path = Path(output_path) / "output_intensity_metrics.csv"
    print(f"Writing scale log to: {scale_log_path}")
    with open(scale_log_path, 'w', newline='') as csvfile:
        fieldnames = ['Image Name', 'Scale Factor', 'Raw Input Max', 'Percentile 99 Used', 'Min Intensity', 'Median Intensity', 'Mean Intensity', 'Max Intensity', 'Max Intensity Count', 'Percentile 99']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in scale_log:
            writer.writerow(entry)

def denoise_directory(input_dir, output_path, model_params, eval_params, max_workers=4):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    image_attributes = load_image_attributes(output_path)
    z_axis = model_params.pop('z_axis', None)
    denoise_model = DenoiseModel(**model_params)
    image_paths = glob(os.path.join(input_dir, '*.tif*'))
    scale_log = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, image_path, output_path, denoise_model, eval_params, image_attributes, scale_log) for image_path in image_paths]
        for future in tqdm(as_completed(futures), total=len(image_paths)):
            _ = future.result()

    write_scale_log(output_path, scale_log)

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

