import argparse
import numpy as np

from bioio import BioImage
from locking_csv_writer import append_to_csv_manifest
from pathlib import Path
from skimage.filters import threshold_otsu
from tifffile import imwrite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--csvmanifest", required=False)
    args = parser.parse_args()
    output_dir = Path(args.outputdir)
    csv_manifest = Path(args.csvmanifest) if args.csvmanifest else None
    # directories containing segmentations for different patch sizes
    path_128 = output_dir / "runtime_data/infer_movie_multiscale_patch1/seg"
    path_256 = output_dir / "runtime_data/infer_movie_multiscale_patch2/seg"
    path_512 = output_dir / "runtime_data/infer_movie_multiscale_patch3/seg"
    # final location for merged masks
    merged_dir = output_dir / "output"
    merged_dir.mkdir(parents=True, exist_ok=True)

    for file in path_128.glob('*.tif'):
        target_file = merged_dir / file.name
        print(f"generating {target_file}")

        bw_128 = create_bw_image(file)
        bw_256 = create_bw_image(path_256 / file.name)
        bw_512 = create_bw_image(path_512 / file.name)

        bw_all = bw_128 + bw_256 + bw_512

        out = bw_all.astype(np.uint8)
        out[out > 0] = 255
        imwrite(target_file, out)
        if csv_manifest:
            # name of the output directory is the name of the source file being put through ACM
            append_to_csv_manifest(csv_manifest, [target_file.absolute(), target_file.name, output_dir.name])


def create_bw_image(img_path):
    img = BioImage(img_path).data[0]
    struct_img = normalize_0_to_255(img[0,:,:,:])
    threshold = threshold_otsu(struct_img)
    return struct_img > threshold


def normalize_0_to_255(img):
    minimum_val = img.min()
    maximum_val = img.max()
    if (maximum_val - minimum_val) == 0.0:
        # this case ends up with all 0 in the output
        scale = 1.0
    else:
        scale = 255.0 / (maximum_val - minimum_val)
    img = (img - minimum_val) * scale
    # preventing rounding errors due to float arithmetic? - Danny 12/02/2025
    img[img < 0.0] = 0.0
    img[img > 255] = 255
    return img.astype(np.uint8)


if __name__ == "__main__":
    main()
