#!/bin/bash
# args:
#   $1: output directory
#   $2: source file path
#   $3: source file name
#   $4: csv manifest path (may be accessed by multiple processes)
#   $5: brightfield channel

set -e
source .venv/bin/activate

# note that throughout this script, quotes are used extensively to avoid issues with spaces/special chars in file paths

# create output dir structure
mkdir -p "$1"
mkdir "$1/runtime_data"
mkdir "$1/runtime_data/cyto_dl_logs"
mkdir "$1/output"

# generate csv to use as input for cyto-dl
echo "movie_path,bf_channel" > "$1/runtime_data/predict.csv"
echo '"'"$2"'",'"$5" >> "$1/runtime_data/predict.csv"

export CYTODL_CONFIG_PATH="$PWD/configs"
# required for apptainer to use GCC
if [ "$LD_LIBRARY_PATH" == "/.singularity.d/libs" ]; then
    export LIBRARY_PATH="$LD_LIBRARY_PATH"
fi

# write the source file row to the csv manifest
python locking_manifest_writer.py "$4" --filepath "$2" --filename "$3" --parentfilename "" 

# this hydra override syntax is a little crazy:
# in order to pass special characters to a hydra CLI override, you need syntax like 'paths.data_dir="some spaces(!)"'
# the string (possibly) containing special characters is an arg to this script, and args can't be used within single quotes
# so, we concatenate by ending the single quote string, inserting a double quote string with the arg, then continuing the single quote string
python -m cyto_dl.eval hydra/job_logging=disabled experiment=im2im/eval_scale1.yaml 'paths.data_dir="'"$1"'/runtime_data"' 'paths.log_dir="'"$1"'/runtime_data/cyto_dl_logs"'
python -m cyto_dl.eval hydra/job_logging=disabled experiment=im2im/eval_scale2.yaml 'paths.data_dir="'"$1"'/runtime_data"' 'paths.log_dir="'"$1"'/runtime_data/cyto_dl_logs"'
python -m cyto_dl.eval hydra/job_logging=disabled experiment=im2im/eval_scale3.yaml 'paths.data_dir="'"$1"'/runtime_data"' 'paths.log_dir="'"$1"'/runtime_data/cyto_dl_logs"'

python ColonyMask_merging_thresholding_rearranging.py --outputdir="$1" --csvmanifest="$4"
