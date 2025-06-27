#!/bin/bash

#SBATCH -A ees240018p
#SBATCH --job-name="regridding"
#SBATCH -o "outputs/regridding.%j.%N.out"
#SBATCH -p RM-512 #could do RM-512, RM-shared, RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --export=ALL
#SBATCH -t 24:00:00 # max of 48 hours for GPU
#SBATCH --mem=300G
#SBATCH --no-requeue

module purge

source /jet/home/gmooers/miniconda3/bin/activate torchenv

cd /ocean/projects/ees240018p/gmooers/CAM_NN/NN_For_CAM/Regridder/

python3 new_regridder_terrain_info.py