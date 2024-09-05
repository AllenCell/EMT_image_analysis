# S3 Data Bucket OME-Zarr Importer

The class provided here can be used to import data on our S3 bucket to you local file system. This can be done on a single-file basis or in large batches using a csv manifest.

## Command-Line Usage

```bash
# create virtual environment
python -m venv importData-env

# activate env and install requirements
./meshgen-env/bin/activate

cd EMT_image_analysis/Data_access
pip install .

# run file importer
python Ome_zarr_importer.py \
    --local_path \      #location in local directory to save files
    # Only provide --file_path or --csv_path, not both
    --file_path  \      #s3 file path to download
    --csv_path \        #csv file with paths s3 files to download
    --file_columns \    #[Optional]list of column names where file paths are provided. (only needed for csv input)
                        #Default is ["Raw Converted URL", "All Cells Mask URL", "EOMES Nuclear Segmentation URL", "H2B Nuclear Segmentation URL", "CollagenIV Segmentation Probability URL"]

# this example command will download the entire EMT dataset to your Downloads folder 
python Ome_zarr_importer.py \
    --local_path ~/Downloads/EMT_data \
    --csv_path "Raw Converted URL" "All Cells Mask URL" "EOMES Nuclear Segmentation URL" "H2B Nuclear Segmentation URL" "CollagenIV Segmentation Probability URL"
```

## Programatic Usage

The file `Ome_zarr_importer.py` contains a class called `OmeZarrImporter()` which can be used programatically to download the data in your own code. 

Please consult the documentation within the source code to understand its usage and methods.

## File Paths to Use

When selecting files to download make sure that the paths begin with `http://allencell` and end with `.ome.zarr` as the filetype. These are the paths which the code can access and download for you.

**Correct Filepath**
`https://allencell.s3.amazonaws.com/aics/emt_timelapse_dataset/data/3500006062_2_raw_converted.ome.zarr`

**Incorrect Filepaths**
`s3://allencell/aics/emt_timelapse_dataset/data/3500006062_2_raw_converted.ome.zarr`
`https://volumeviewer.allencell.org/viewer?url=https%3A%2F%2Fallencell.s3.amazonaws.com%2Faics%2Femt_timelapse_dataset%2Fdat`