import argparse
import glob
import os
import numpy as np
from skimage.filters import threshold_otsu
import tifffile

from bioio import BioImage
import logging

def MyconvertFloatToChar(img: np.ndarray) -> np.ndarray:
    minimum_val = img.min()
    maximum_val = img.max()
    if (maximum_val - minimum_val) == 0.0:
        scale = 1.0
    else:
        scale = 255.0 / (maximum_val - minimum_val)
    img = (img - minimum_val) * scale
    img[img < 0.0] = 0.0
    img[img > 255] = 255
    return img.astype(np.uint8)

def setup_logging(scene: str) -> None:
    """Set up logging for a specific scene."""
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s [%(levelname)s] [Scene {scene}] %(message)s',
        handlers=[
            logging.FileHandler('progress.log', mode='a'),
            logging.StreamHandler()
        ]
    )

def process_image(filename: str, path_512: str, path_128: str, target_dir: str, scene: str) -> str:
    try:
        imgname = os.path.basename(filename)
        target_filename = os.path.join(target_dir, scene, imgname)

        reader = BioImage(filename)
        struct_img0 = reader.data[0, 0]
        struct_img1 = MyconvertFloatToChar(struct_img0)
        thre = threshold_otsu(struct_img1)
        bw = struct_img1 > thre

        img_512_path = os.path.join(path_512, imgname)
        img_512 = BioImage(img_512_path)
        struct_img0_512 = img_512.data[0, 0]
        struct_img1_512 = MyconvertFloatToChar(struct_img0_512)
        thre_512 = threshold_otsu(struct_img1_512)
        bw_512 = struct_img1_512 > thre_512

        img_128_path = os.path.join(path_128, imgname)
        img_128 = BioImage(img_128_path)
        struct_img0_128 = img_128.data[0, 0]
        struct_img1_128 = MyconvertFloatToChar(struct_img0_128)
        thre_128 = threshold_otsu(struct_img1_128)
        bw_128 = struct_img1_128 > thre_128

        bw_all = bw + bw_128 + bw_512

        out = bw_all.astype(np.uint8)
        out[out > 0] = 255
        tifffile.imwrite(target_filename, out)

        return f"Processed {imgname} successfully"
    except Exception as e:
        return f"Error processing {imgname}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Process images for a specific scene")
    parser.add_argument('--scene', required=True, help="Scene name (e.g., P11-B9)")
    parser.add_argument('--target-dir', required=True, help="Target base directory")
    parser.add_argument('--path-512', required=True, help="Path to scale3 images")
    parser.add_argument('--path-128', required=True, help="Path to scale1 images")
    parser.add_argument('--path-scale2', required=True, help="Path to scale2 images")
    args = parser.parse_args()

    setup_logging(args.scene)
    logging.info(f"Starting processing for scene {args.scene}")

    dir_path = os.path.join(args.path_scale2, f'*{args.scene}*tif')
    files = glob.glob(dir_path)
    total_files = len(files)
    logging.info(f"Found {total_files} .tif files to process")

    results = []
    for i, filename in enumerate(files, 1):
        result = process_image(filename, args.path_512, args.path_128, args.target_dir, args.scene)
        results.append(result)
        logging.info(f"Processed file {i}/{total_files}: {result}")

    logging.info(f"Completed processing for scene {args.scene}")

if __name__ == '__main__':
    main()
