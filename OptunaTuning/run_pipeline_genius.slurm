#!/usr/bin/env bash

#SBATCH --job-name="sutran_hpo_pipeline"
#SBATCH --cluster="genius"
#SBATCH --partition="gpu_v100"
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=16G
#SBATCH --time=40:00:00
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

# Full HPO pipeline: Phase 1 (random search) -> Phase 2 (TPE + Hyperband)
# Both phases are resumable. Re-running after interruption resumes from where it left off.
echo "Starting HPO pipeline..."
python -m OptunaTuning.run_pipeline

echo "Pipeline complete!"
