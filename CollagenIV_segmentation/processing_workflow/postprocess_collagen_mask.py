import os
import argparse
from tqdm import tqdm
import skimage
import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from pathlib import Path
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback


def morphology_ops(seg_slice: np.ndarray):
    seg_slice = skimage.morphology.remove_small_objects(seg_slice, min_size=25)
    seg_slice = skimage.morphology.dilation(seg_slice, footprint=skimage.morphology.disk(4))
    seg_slice = skimage.morphology.binary_closing(seg_slice, footprint=skimage.morphology.disk(4))
    seg_slice = skimage.morphology.remove_small_holes(seg_slice, area_threshold=50000)
    return seg_slice

def background_subtracted_segmentation(seg_fn, output, threshold=0.25):
    '''
    Processes the segmentation probability mask to keep only the largest connected component in the segmentation mask
    '''
    output_save_name = f"{seg_fn.stem}_seg_collagen.tiff"
    if (output / output_save_name).exists():
        pred = BioImage(output / output_save_name).data.squeeze()
        if np.any(pred):
            return

    pred = BioImage(seg_fn).data.squeeze()
    pred_thresh = pred> threshold*255
    
    with Pool(8) as p:
        pred_thresh = np.stack(
            p.map(morphology_ops, [pred_thresh[i,...] for i in range(pred_thresh.shape[0])]),
            axis=0
        )

    pred_thresh = skimage.measure.label(pred_thresh)
    # only keep largest object
    lumen_sizes = [prop.area for prop in skimage.measure.regionprops(pred_thresh)]
    if len(lumen_sizes) == 0:
        tempelate_background = np.zeros_like(pred_thresh)
    else:
        tempelate_background = np.where(
            pred_thresh == (np.argmax(lumen_sizes)+1), 
            pred, 0)

    # output_save_name = f"{seg_fn.stem}_seg_collagen.tiff"        
    OmeTiffWriter().save(tempelate_background, output / output_save_name, dim_order="ZYX")

    return

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--segmentation_dir', type=str, required=False, 
                    help="Directory with the segmentation files ouput by cyto-dl that need postprocessing")
parser.add_argument('-o', '--output_dir', type=str, required=True,
                    help="Directory to save post-processed images")

if __name__ == "__main__":
    args = parser.parse_args()
    
    seg_fns = [fn for fn in Path(args.segmentation_dir).iterdir() if fn.suffix in ['.tif', '.tiff']]
    output = Path(args.output_dir)
    output.mkdir(exist_ok=True, parents=True)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(background_subtracted_segmentation, fn, output) for fn in seg_fns]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Segmentations"):
            try:
                _ = future.result()
            except Exception as e:
                print(f"Error processing timelapse: {e}")
                print(traceback.format_exc())





