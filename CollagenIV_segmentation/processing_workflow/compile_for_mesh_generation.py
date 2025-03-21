import numpy as np
import pandas as pd
import dask.array as da
from pathlib import Path

from bioio import BioImage
from bioio.writers import OmeTiffWriter
from tqdm import tqdm
from argparse import ArgumentParser


def compile(
        cyto_csv: Path | str,
        seg_dir: Path | str,
        output_dir: Path | str,
    ):
    #read cytodl csv
    df_cyto = pd.read_csv(cyto_csv)
    
    # ensure seg_dir and output_dir are paths
    if isinstance(seg_dir,str):
        seg_dir = Path(seg_dir)
    if isinstance(output_dir,str):
        output_dir = Path(output_dir)

    #make output folder
    output_dir.mkdir(exist_ok=True, parents=True)

    def sort_tp(fn: Path):
        return int(fn.name.split('_T_')[1].split('_')[0])

    df_mesh = []
    for _, czi in tqdm(df_cyto.iterrows()):
        base_fn = Path(czi['path']).stem
        movie_id = base_fn + f'_scene_{scene}_T_{start:04d}-{end:04d}'
        scene = czi['scene']
        start = int(czi['start'])
        end = int(czi['stop'])
        out_fn = output_dir / (movie_id + '_basement_membrane_segmentation.tif')

        scene_fns = [fn for fn in seg_dir.glob(f'{base_fn}*_scene_{scene}*.tif')]
        scene_fns.sort(key=sort_tp)

        img_stack = []
        for fn in tqdm(scene_fns):
            img_stack.append(BioImage(fn).get_image_dask_data('ZYX',T=0,C=0))

        img_stack = da.stack(img_stack, axis=0).compute()
        
        OmeTiffWriter().save(
            img_stack,
            out_fn,
            dim_order='TZYX'
        )

        row = {
            'Movie Unique ID': movie_id,
            'CollagenIV Segmentation Probability File Download': out_fn,
            'Image Size T': img_stack.shape[0]
        }
        df_mesh.append(row)
    df_mesh = pd.DataFrame(df_mesh, index=None)
    df_mesh.to_csv(output_dir / 'mesh_generation_manifest.csv')

parser = ArgumentParser()
parser.add_argument('--segmentation_manifest', '-m', type=str, required=True,
                    help='Path to csv manifest used for running segmentations')
parser.add_argument('--segmentation_directory', '-s', type=str, required=True,
                    help='Path to directory with post-processed segmentations')
parser.add_argument('--output_directory', '-o', type=str, required=True,
                    help='Where to save compiled segmentations and mesh generation manifest')

if __name__ == "__main__":
    args = parser.parse_args()
    
    compile(
        args.segmentation_manifest, 
        args.segmentation_directory, 
        args.output_directory
    )