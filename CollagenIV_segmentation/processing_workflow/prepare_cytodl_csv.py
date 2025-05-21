import pandas as pd
import numpy as np

from bioio import BioImage
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Union

def generate_csv(
        img_path: str | Path,
        channel: int = 2,
        scenes: List[str] | None = None,
        start_tp: int = 0,
        end_tp: int = -1,
        step:int = 1 
    ):
    '''
    Generate a dataframe for the given czi that is compatible as an input for
     cytodl.datamodules.multidim_image.MultiDimImageDataset.

    ---Parameters---
        img_path: str | Path
            Path to czi file
        channel: int
            Channel to segment (Default 2)
        scenes: List[str] | None
            Scenes from czi file to segment. If none is provided all scenes
             will be used. (Default None)
        start_tp: int
            Starting timepoint to begin segmentation. (Default 0)
        end_tp: int
            Last timepoint for segmentation. If value is -1 segmetnations 
             will be processed to last timepoint in each scene. (Default -1)
        step: int
            Interval between timepoints that are segmented. (Default 1)
            
    '''
    # if isinstance(img_path, str):
    #     img_path = Path(img_path)
    
    #load image
    img = BioImage(img_path)

    # set scenes to all present or make sure all specified scenes exist
    if scenes is None:
        scenes = img.scenes
    else:
        for scn in scenes:
            assert scn in img.scenes, f"Invalid scene: scene with name \"{scn}\" not found in the czi file"

    assert start_tp >= 0, "Invalid start: start_tp < 0"

    df_scenes = []
    print(scenes)
    for scene in scenes:
        #set img scene
        img.set_scene(scene)

        #check that start timepoint is valid
        assert start_tp < img.shape[0], f"Invalid start for scene {scene}: start_tp is large than length of sequence ({img.shape[0]})"

        #set end timepoint to length of sequence or provided value
        if end_tp < 0:
            end_tp_scene = img.shape[0]-1
        else:
            end_tp_scene = min([end_tp, img.shape[0]-1])

        # assert valid channel for scene
        assert channel < img.shape[1], f"Invalid channel for scene {scene}: {channel} > {img.shape[1]-1}"

        df_scenes.append(
            {
                'path': img_path,
                'channel': channel,
                'scene': scene,
                'start': start_tp,
                'end': end_tp_scene,
                'step': step
            }
        )

    return pd.DataFrame(df_scenes)


# CLI inputs
parser = ArgumentParser()
parser.add_argument('--czi', type=str, default=None,
                    help='Source czi file for all scenes')
parser.add_argument('--manifest', type=str, default=None,
                    help='Source czi file for all scenes')

parser.add_argument('--output_dir', type=str, required=True,
                    help='Directory to save csv. Name will be same as czi')
parser.add_argument('--channel', type=int, default=2,
                    help='Channel to segment. (default 2)')
parser.add_argument('--scenes', nargs='+', default=None,
                    help='List of scenes to use, separated by spaces. Use all scenes if not included (default None). ex: --scenes P1-D1 P2-D2 P3-D3')
parser.add_argument('--start', type=int, default=0,
                    help='Start timepoint')
parser.add_argument('--end', type=int, default=-1,
                    help='End timepoint. If -1, all timepoints from start. (default -1)')
parser.add_argument('--step', type=int, default=1,
                    help='Interval between timepoints segmented. (default 1)')

if __name__ == "__main__":
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(exist_ok=True, parents=True)

    if args.czi is not None:
        df_csv = generate_csv(
            img_path=args.czi,
            channel=args.channel,
            scenes=args.scenes,
            start_tp=args.start,
            end_tp=args.end,
            step=args.step
        )
    elif args.manifest is not None:
        df_manifest = pd.read_csv(args.manifest, index_col=None)
        df_csv = []
        for _, row in df_manifest.iterrows():
            fn = row['Raw Converted File Download'].replace('\\','/')
            df_csv.append(
                generate_csv(
                    img_path=fn,
                    channel=args.channel,
                    scenes=args.scenes,
                    start_tp=args.start,
                    end_tp=args.end,
                    step=args.step
                )
            )
        df_csv = pd.concat(df_csv, ignore_index=True)
    else:
        SyntaxError('Must provide either a source file or a manifest of source files')

    df_csv.to_csv(output / 'segmentation_manifest.csv', index=False)

    