import os
from aicsimageio.writers import OmeTiffWriter
import os
import pandas as pd
import time
import argparse
from aicsimageio import AICSImage
from tqdm import tqdm
import skimage
import numpy as np

def get_fms_id(raw_log_file):
    fms_id = os.path.basename(raw_log_file).split("_input.txt", 1)[0].split("fmsid_", 1)[1]
    return fms_id


def background_subtracted_segmentation(pred, threshold=0.25):
    '''
    Processes the segmentation probability mask to keep only the largest connected component in the segmentation mask
    '''
    tempelate_background = np.zeros_like(pred)
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

def get_matching_filenames(fms_id, path):
    matching_filenames = [os.path.join(path,f) for f in os.listdir(path) if fms_id in f]
    return matching_filenames

parser = argparse.ArgumentParser()
parser.add_argument('--raw_log_file', type=str, required=False, help="dummy file of raw_log for snakemake for input. This is just used to determine metadata of file to run")
parser.add_argument('--output_log_file', type=str, required=False, help="dummy file dir raw_log for snakemake for output. This is just used to determine metadata of file to run")
parser.add_argument('--output_parent_dir', type=str, help="Where the parent dir for output is saved")
parser.add_argument('--output_segmentation_dir', type=str, help="Where the cytoDl segmentation masks directly are saved")




if __name__ == "__main__":
    args = parser.parse_args()
    fms_id = get_fms_id(args.raw_log_file)
    segmentation_filenames = get_matching_filenames(fms_id, args.output_segmentation_dir)


    output_fmsdir = os.path.join(args.output_parent_dir, fms_id)
    if not os.path.exists(output_fmsdir):
        os.mkdir(output_fmsdir)

    output_fms_savedir = os.path.join(output_fmsdir, "basement_membrane_segmentation_test")

    if not os.path.exists(output_fms_savedir):
        os.mkdir(output_fms_savedir)

    for i in tqdm(range(len(segmentation_filenames))):
        img = AICSImage(segmentation_filenames[i])
        dat = img.data[0,0,:,:,:]
        final_seg = background_subtracted_segmentation(dat)

        # save name zfill to 4 digits
        output_save_name = f"fms_id_{fms_id}_seg_collagen_tp_{str(i).zfill(4)}.tiff"        
        OmeTiffWriter.save(final_seg, os.path.join(output_fms_savedir, output_save_name), dim_order="ZYX", channel_names=["Collagen"])
        print(f"Saved tiff_{i}.tiff")


    with open(args.output_log_file, 'w') as f:
        f.write("done")




