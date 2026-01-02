#!/bin/bash
#SBATCH --partition aics_gpu
#SBATCH --gres gpu:v100:1
#SBATCH --time 150
#SBATCH --mem 64Gb

# this script should be used with sbatch (it is a workaround for the slurm API being unable to request GPUs)

# first arg is the output directory, second arg is the source file path, third arg is source file name, fourth arg is csv manifest path
# we mount /allen to give VAST access, and mount $TMPDIR (the real path for it) to allow pytorch to store JIT compiled executables

apptainer run --cwd /usr/src/acm --mount type=bind,src=/allen,dst=/allen --mount type=bind,src=$(realpath $TMPDIR),dst=$(realpath $TMPDIR) --env TMPDIR=$(realpath $TMPDIR) --nv example-hpc-image_0.0.12.sif "$1" "$2" "$3" "$4"