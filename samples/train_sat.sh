#!/bin/sh
#SBATCH -p gtx1080
#SBATCH --time 24:00:00

source ../setup.sh

data_dir=$1

python samples/sat/sat.py --dataset=$data_dir --weights=coco
