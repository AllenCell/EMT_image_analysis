from bioio import BioImage
from bioio.writers import OmeTiffWriter
import bioio_ome_tiff
import bioio_ome_zarr

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List
from tqdm import tqdm

import argparse

class OmeZarrImporter():
    """
    Class for importing OME-Zarr files from the s3 bucket as OME-Tiff files to the local file system. 
    """
    def __init__(
            self, 
            local_path: str,
        ):
        """
        parameters:
            local_path: str
                The path to the local file system where the OME-Tiff files will be saved.
        """
        self.local_path = local_path
        Path(self.local_path).mkdir(parents=True, exist_ok=True)


    def import_from_dataframe(self, 
            csv_path: str, 
            file_columns: List[str] = [
                "Raw Converted URL",
                "All Cells Mask URL",
                "EOMES Nuclear Segmentation URL",
                "H2B Nuclear Segmentation URL",
                "CollagenIV Segmentation Probability URL"
            ]
        ):
        """
        Imports files from a dataframe with paths to the s3 bucket to the local file system. 
        
        parameters:
            csv_path: str
                The path to a csv file with paths to the OME-Zarr files in the s3 bucket.
            file_columns: list
                The columns in the csv file that contain the paths to the OME-Zarr files.
        """
        df = pd.read_csv(csv_path, columns=file_columns)
        files = []
        for col in file_columns:
            files.extend(df[col].unique().tolist())
        for file in tqdm(files):
            self.import_ome_zarr(file)
        return

    def import_ome_zarr(self, remote_path):
        """
        Imports an OME-Zarr file from the s3 bucket as an OME-Tiff file to the local file system. 
        
        parameters:
            remote_path: str
                The path to the OME-Zarr file in the s3 bucket. Must begin with \"https://allencell\".
        """
        self.check_path(remote_path)
        image = BioImage(remote_path, reader=bioio_ome_tiff.OmeTiffReader)

        fname = Path(remote_path).name.replace(".ome.zarr", ".ome.tiff")
        import_path = Path(self.local_path) / fname

        image.save(import_path)
        return 
    
    def check_path(self, file_path: str):
        """
        Checks if the path begins with \"https://allencell\" and ends with \".ome.zarr\".
        """

        if not file_path.startswith("https://allencell"):
            raise ValueError("The path must start with \"https://\"")
        if not file_path.endswith(".ome.zarr"):
            raise ValueError("The path must end with \".ome.zarr\"")
        
        return
    
    
if __name__ == "__main__":
    # CLI arguments
    parser = argparse.ArgumentParser(description="Import OME-Zarr files from the s3 bucket as OME-Tiff files to the local file system.")
    
    parser.add_argument("--local_path", type=str, help="The path to the local file system where the OME-Tiff files will be saved.")
    parser.add_argument("--file_path", type=str, default=None, help="The path to the OME-Zarr file in the s3 bucket.")
    parser.add_argument("--csv_path", type=str, default=None, help="The path to a csv file with paths to the OME-Zarr files in the s3 bucket.")
    parser.add_argument("--file_columns", type=str, nargs="+", default=["Raw Converted URL", "All Cells Mask URL", "EOMES Nuclear Segmentation URL", "H2B Nuclear Segmentation URL", "CollagenIV Segmentation Probability URL"], help="The columns in the csv file that contain the paths to the OME-Zarr files.")

    args = parser.parse_args()

    # Check arguments
    if args.file_path is None and args.csv_path is None:
        raise ValueError("Please provide either a file path or a csv path.")
    if args.file_path is not None and args.csv_path is not None:
        raise ValueError("Please provide either a file path or a csv path, not both.")
    
    # Import OME-Zarr files
    if args.file_path is not None:
        importer = OmeZarrImporter(args.local_path)
        importer.import_ome_zarr(args.file_path)
    else:
        importer = OmeZarrImporter(args.local_path)
        importer.import_from_dataframe(args.csv_path, args.file_columns)