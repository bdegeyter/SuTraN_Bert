#!/usr/bin/env bash

#SBATCH --job-name="sutran_tpe_search"
#SBATCH --cluster="genius"
#SBATCH --partition="gpu_v100"
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=16G
#SBATCH --time=24:30:00
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

# TPE + Hyperband: single-objective normalized composite, 24h timeout, resumable
# Requires random search results to exist (for normalization constants).
# Re-running this script after interruption resumes from where it left off.
echo "Starting TPE + Hyperband search..."
python -m OptunaTuning.tpe_search

echo "TPE search complete!"