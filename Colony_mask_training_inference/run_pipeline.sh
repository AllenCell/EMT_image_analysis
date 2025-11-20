#!/bin/bash

source .venv/bin/activate
# first arg is the output directory, second arg is the source file path
mkdir -p $1
mkdir $1/runtime_data
mkdir $1/runtime_data/cyto_dl_logs
mkdir $1/output

# generate csv to use as input
echo "count,movie_path,bf_channel" > $1/runtime_data/predict.csv
echo "0,$2,0" >> $1/runtime_data/predict.csv

export CYTODL_CONFIG_PATH=$PWD/configs
python -m cyto_dl.eval experiment=im2im/eval_scale1.yaml paths.data_dir=$1/runtime_data paths.log_dir=$1/runtime_data/cyto_dl_logs
python -m cyto_dl.eval experiment=im2im/eval_scale2.yaml paths.data_dir=$1/runtime_data paths.log_dir=$1/runtime_data/cyto_dl_logs
python -m cyto_dl.eval experiment=im2im/eval_scale3.yaml paths.data_dir=$1/runtime_data paths.log_dir=$1/runtime_data/cyto_dl_logs

python ColonyMask_merging_thresholding_rearranging.py --outputdir=$1