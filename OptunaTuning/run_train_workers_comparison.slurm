#!/usr/bin/env bash

#SBATCH --job-name="sutran_workers_comparison"
#SBATCH --cluster="genius"
#SBATCH --partition="gpu_v100"
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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

# Test 1: Without workers (baseline) - 5 epochs for quick comparison
echo "=========================================="
echo "TEST 1: Training WITHOUT workers (5 epochs)"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "from TRAIN_EVAL_SUTRAN_DA import train_eval; train_eval('BPIC_19', use_workers=False, num_epochs=5)"

echo ""
echo "=========================================="
echo "TEST 1 COMPLETE - check epoch times above"
echo "=========================================="
echo ""

# Test 2: With workers (optimized) - 5 epochs for quick comparison
echo "=========================================="
echo "TEST 2: Training WITH workers (num_workers=4, 5 epochs)"
echo "=========================================="
python -c "from TRAIN_EVAL_SUTRAN_DA import train_eval; train_eval('BPIC_19', use_workers=True, num_epochs=5)"

echo ""
echo "=========================================="
echo "TEST 2 COMPLETE - check epoch times above"
echo "=========================================="
echo ""
echo "COMPARISON COMPLETE - Compare epoch times from both tests"
