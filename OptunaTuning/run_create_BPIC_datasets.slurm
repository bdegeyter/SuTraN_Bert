#!/usr/bin/env bash

#SBATCH --job-name="create_toy_dataset"
#SBATCH --cluster="genius"
#SBATCH --partition="gpu_v100"
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type="BEGIN,END,FAIL"
#SBATCH --mail-user="bert.degeyter@student.kuleuven.be"
#SBATCH --output="/data/leuven/383/vsc38329/%x.o%A"
#SBATCH --error="/data/leuven/383/vsc38329/%x.e%A"

# Load conda
source /data/leuven/383/vsc38329/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate sutran_env

# Navigate to project
cd $VSC_SCRATCH/SuTraN_Bert

# Create datasets
echo "Creating BPIC17_DR dataset..."
python create_BPIC17_DR_data.py

echo "Creating BPIC19 dataset..."
python create_BPIC19_data.py

echo "Dataset creation complete!"
