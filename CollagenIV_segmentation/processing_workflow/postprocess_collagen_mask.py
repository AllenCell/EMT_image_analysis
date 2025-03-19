import os
import argparse
from tqdm import tqdm
import skimage
import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from pathlib import Path


def background_subtracted_segmentation(pred, threshold=0.25):
    '''
    Processes the segmentation probability mask to keep only the largest connected component in the segmentation mask
    '''
    thresh = 255*threshold
    binary = pred> thresh
    for slice in range(np.shape(binary)[0]):
            binary[slice,:,:] = skimage.morphology.remove_small_objects(binary[slice,:,:], min_size=25)
            binary[slice,:,:] = skimage.morphology.dilation(binary[slice,:,:], footprint=skimage.morphology.disk(4))
            binary[slice,:,:] = skimage.morphology.binary_closing(binary[slice,:,:], footprint=skimage.morphology.disk(4))
            binary[slice,:,:] = skimage.morphology.remove_small_holes(binary[slice,:,:], area_threshold=50000)
    labeled_lumen = skimage.measure.label(binary)
    # only keep largest object
    lumen_sizes = [np.sum(labeled_lumen==i) for i in np.unique(labeled_lumen)[1:]]
    final_lumen = labeled_lumen == (np.argmax(lumen_sizes)+1)
    tempelate_background = final_lumen*pred

    return tempelate_background

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


    for fn in tqdm(seg_fns, total=seg_fns):
        seg = BioImage(fn).data.squeeze()
        seg = background_subtracted_segmentation(seg)

        output_save_name = f"{fn.name}_seg_collagen.tiff"        
        OmeTiffWriter().save(seg, output / output_save_name, dim_order="ZYX")





