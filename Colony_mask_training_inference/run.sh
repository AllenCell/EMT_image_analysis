#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Makes pipelines return the exit code of the last failing command

DEVICE_ID="add/your/GPU/UUID"
CONFIG_PATH="$PWD/configs"
EVAL_SCRIPT="../../cyto-dl/cyto_dl/eval.py"

# List of experiments to run sequentially
EXPERIMENTS=( 
    #"im2im/yamls_generated/eval_scale2_7450_p10.yaml"
    #"im2im/yamls_generated/eval_scale2_7450_p11.yaml"
    #"im2im/yamls_generated/eval_scale2_7450_p12.yaml"
    #"im2im/yamls_generated/eval_scale2_7450_p13.yaml"
)

echo "Using device: $DEVICE_ID"
echo "Using config path: $CONFIG_PATH"

for EXP in "${EXPERIMENTS[@]}"; do
    echo "Starting job for experiment: $EXP"

    CUDA_VISIBLE_DEVICES=$DEVICE_ID CYTODL_CONFIG_PATH=$CONFIG_PATH \
        python $EVAL_SCRIPT experiment="$EXP"

    echo "Finished job for experiment: $EXP"
    wait
done

echo "All jobs completed!"
