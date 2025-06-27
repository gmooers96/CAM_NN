#!/bin/bash

#SBATCH -A ees220005p
#SBATCH --job-name="cam_dense_1"
#SBATCH -o "outputs/cam_dense_1.%j.%N.out"
#SBATCH -p RM-512 #could do RM-512, RM-shared, RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --export=ALL
#SBATCH -t 48:00:00 # max of 48 hours for GPU
#SBATCH --mem=492G
#SBATCH --no-requeue

module purge

source /jet/home/gmooers/miniconda3/bin/activate torchenv

cd /ocean/projects/ees240018p/gmooers/CAM_NN/NN_For_CAM/Training/

python3 neural_network_training_baseline.py /ocean/projects/ees240018p/gmooers/CAM_NN/NN_For_CAM/Training/New_Configs/cam_config_1.yaml
