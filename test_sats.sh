#!/bin/sh
#SBATCH -p gtx1080
#SBATCH --time 24:00:00

source ../setup.sh

data_dir=$1

for dataset_path in data_dir/*; do
    dataset=$(basename $dataset_path)
    python samples/sat/sat.py test --dataset $dataset_path --weights last --results results/$dataset
