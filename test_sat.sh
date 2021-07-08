#!/bin/sh
#SBATCH -p gtx1080
#SBATCH --time 24:00:00

source ../setup.sh

data_dir=$1

cd samples/sat
python sat.py test --dataset $data_dir --weights last
cd ../../
