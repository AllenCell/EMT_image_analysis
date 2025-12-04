#!/bin/bash
#SBATCH --partition aics_gpu
#SBATCH --gres gpu:v100:1
#SBATCH --time 150
#SBATCH --mem 64Gb

source .venv/bin/activate
# first arg is the output directory, second arg is the source file path, third arg is source file name, fourth arg is csv manifest path
# note that quotes are used extensively to avoid issues with spaces/special chars in file paths
mkdir -p "$1"
mkdir "$1/runtime_data"
mkdir "$1/runtime_data/cyto_dl_logs"
mkdir "$1/output"

# generate csv to use as input
python locking_csv_writer.py "$1/runtime_data/predict.csv" count movie_path bf_channel
python locking_csv_writer.py "$1/runtime_data/predict.csv" 0 "$2" 0

export CYTODL_CONFIG_PATH=$PWD/configs
export HOME=/home/daniel.saelid

# write the source file row to the csv manifest
python locking_csv_writer.py "$4" "$2" "$3" "" 

# this hydra override syntax is a little crazy:
# in order to pass special characters to a hydra CLI override, you need syntax like 'paths.data_dir="some spaces(!)"'
# the string (possibly) containing special characters is an arg to this script, and args can't be used within single quotes
# so, we concatenate by ending the single quote string, inserting a double quote string with the arg, then continuing the single quote string
python -m cyto_dl.eval experiment=im2im/eval_scale1.yaml 'paths.data_dir="'"$1"'/runtime_data"' 'paths.log_dir="'"$1"'/runtime_data/cyto_dl_logs"'
python -m cyto_dl.eval experiment=im2im/eval_scale2.yaml 'paths.data_dir="'"$1"'/runtime_data"' 'paths.log_dir="'"$1"'/runtime_data/cyto_dl_logs"'
python -m cyto_dl.eval experiment=im2im/eval_scale3.yaml 'paths.data_dir="'"$1"'/runtime_data"' 'paths.log_dir="'"$1"'/runtime_data/cyto_dl_logs"'

python ColonyMask_merging_thresholding_rearranging.py --outputdir="$1"
