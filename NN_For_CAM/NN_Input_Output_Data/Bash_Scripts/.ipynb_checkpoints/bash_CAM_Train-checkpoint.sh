#!/bin/bash

#SBATCH -A ees240018p
#SBATCH --job-name="CAM_Train"
#SBATCH -o "outputs/CAM_Train.%j.%N.out"
#SBATCH -p RM-512 #could do RM-512, RM-shared, EM, RM
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH -t 12:00:00 # max of 48 hours for GPU
#SBATCH --mem=492G
#SBATCH --no-requeue


module purge

source /jet/home/gmooers/miniconda3/bin/activate torchenv

cd /ocean/projects/ees240018p/gmooers/CAM_NN/NN_For_CAM/NN_Input_Output_Data/

python3 input_output_data_preparation.py /ocean/projects/ees240018p/gmooers/CAM_NN/NN_For_CAM/NN_Input_Output_Data/configs/config_cam_1.yaml

