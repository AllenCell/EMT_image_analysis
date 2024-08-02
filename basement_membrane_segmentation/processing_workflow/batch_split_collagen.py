import os
from aicsimageio.writers import OmeTiffWriter
import os
import pandas as pd
import time
import argparse
from aicsimageio import AICSImage
from tqdm import tqdm


def get_dataset_path(raw_log_file, manifest):
    fms_id = os.path.basename(raw_log_file).split("_input.txt", 1)[0].split("fmsid_", 1)[1]
    filedir = manifest[manifest["file_id"] == fms_id]["file_path"].values[0]
    return filedir, fms_id


parser = argparse.ArgumentParser()
parser.add_argument('--raw_log_file', type=str, required=False, help="dummy file of raw_log for input for snakemake. This is just used to determine metadata of file to run")
parser.add_argument('--output_log_file', type=str, required=False, help="dummy file of raw_log for output for snakemake. This is just used to determine metadata of file to run")
parser.add_argument('--output_parent_dir', type=str, help="parent dir output directory")
parser.add_argument('--manifest_dir', type=str, help="fms manifest")




if __name__ == "__main__":
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest_dir)
    data_path, fms_id = get_dataset_path(args.raw_log_file, manifest)
    img = AICSImage(data_path)
    channel_dim = img.channel_names.index("AF660")
    timepoints = img.dims['T'][0]

    output_fmsdir = os.path.join(args.output_parent_dir, fms_id)

    if not os.path.exists(output_fmsdir):
        os.mkdir(output_fmsdir)

    output_fms_savedir = os.path.join(output_fmsdir, "raw_collagen")
    if not os.path.exists(output_fms_savedir):
        os.mkdir(output_fms_savedir)
    # For each timepoint, load image and save as tiff
    for i in tqdm(range(timepoints)):
        dat = img.get_image_dask_data("CZYX", T=int(i))
        raw_img = dat.compute()[channel_dim,:,:,:]
        # save name zfill to 4 digits
        output_save_name = f"fms_id_{fms_id}_raw_collagen_tp_{str(i).zfill(4)}.tiff"        
        OmeTiffWriter.save(raw_img, os.path.join(output_fms_savedir, output_save_name), dim_order="ZYX", channel_names=["Collagen"])
        print(f"Saved tiff_{i}.tiff")


    with open(args.output_log_file, 'w') as f:
        f.write("done")




    
