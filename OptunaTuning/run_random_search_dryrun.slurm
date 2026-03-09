#!/usr/bin/env bash

#SBATCH --job-name="sutran_random_search_dryrun"
#SBATCH --cluster="wice"
#SBATCH --partition="gpu_h100"
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type="BEGIN,END,FAIL"
#SBATCH --mail-user="bertrand.dequeker@student.kuleuven.be"
#SBATCH --output="/data/leuven/383/vsc38329/%x.o%A"
#SBATCH --error="/data/leuven/383/vsc38329/%x.e%A"

# Load conda
source /data/leuven/383/vsc38329/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate sutran_env

# Navigate to project
cd $VSC_SCRATCH/SuTraN_Bert

# DRY RUN: Small n_trials and n_epochs just to verify everything works
echo "Starting DRY RUN with n_trials=2, n_epochs=2..."
python -m OptunaTuning.random_search

echo "DRY RUN COMPLETE"
