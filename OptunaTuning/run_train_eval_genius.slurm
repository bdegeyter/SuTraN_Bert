#!/usr/bin/env bash

#SBATCH --job-name="sutran_train_eval"
#SBATCH --cluster="genius"
#SBATCH --partition="gpu_v100"
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account="lp_lirisnlp"
#SBATCH --mem=16G
#SBATCH --time=12:00:00
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

# Final train + test evaluation using best HPs from TPE.
# Trains from scratch with fixed HPs, applies early stopping + rank-sum
# epoch selection, then evaluates on the test set.
# Results saved to: OptunaTuning/eval_results/BPIC_17_DR/tpe_best/
echo "Starting final train + test evaluation..."
python -c "
import pickle
from OptunaTuning.train_and_evaluate import train_and_evaluate

with open('OptunaTuning/results/tpe_BPIC_17_DR/tpe_best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

print('Best params:', best_params)

results = train_and_evaluate(
    params=best_params,
    log_name='BPIC_17_DR',
    n_epochs=200,
    seed=42,
    run_name='tpe_best',
    patience=24,
)
"

echo "Train + eval complete!"
