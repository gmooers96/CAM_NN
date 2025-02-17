#!/bin/bash

#SBATCH -A ees220005p
#SBATCH --job-name="original_CAM_trial_10"
#SBATCH -o "outputs/original_CAM_trial_10.%j.%N.out"
#SBATCH -p RM-512 #could do RM-512, RM-shared, RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --export=ALL
#SBATCH -t 72:00:00 # max of 48 hours for GPU
#SBATCH --mem=492G
#SBATCH --no-requeue

module purge

source /jet/home/gmooers/miniconda3/bin/activate torchenv

cd /ocean/projects/ees240018p/gmooers/Githubs/Neural_nework_parameterization/NN_training/src/

python3 neural_network_training_Original_JN.py /ocean/projects/ees240018p/gmooers/Githubs/Neural_nework_parameterization/NN_training/run_training/Improved_run_Experiments/Config_Files/New_Configs/config_10_original_CAM.yaml