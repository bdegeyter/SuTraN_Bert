"""
Configuration for Optuna Hyperparameter Tuning
===============================================

This file defines:
  1. SEARCH_SPACE  - the ranges/choices for each hyperparameter
  2. FIXED_PARAMS  - model settings we don't tune (kept at paper defaults)
  3. TRAINING_CONFIG - how many trials / epochs / etc.

All 15 tunable hyperparameters and their ranges are defined in one place.
Changing a range here is all you need to do to update the search.
"""

# ──────────────────────────────────────────────────────────────────────
# 1. SEARCH SPACE
# ──────────────────────────────────────────────────────────────────────
#
# Each entry describes one hyperparameter. The "type" field tells the
# objective function which Optuna suggest_* method to use:
#
#   "categorical"  → trial.suggest_categorical(name, choices)
#   "int"          → trial.suggest_int(name, low, high)
#   "float"        → trial.suggest_float(name, low, high)
#   "log_float"    → trial.suggest_float(name, low, high, log=True)
#
# d_model values are all divisible by every num_heads choice (2, 4, 8),
# so any combination is valid — no need for constraint checking.

SEARCH_SPACE = {
    "d_model": {
        "type": "categorical",
        "choices": [32, 48, 64, 96, 128, 192, 256],
    },
    "d_ff_multiplier": {
        # d_ff = d_model * d_ff_multiplier.
        # Using a multiplier instead of absolute values keeps d_ff
        # proportional to d_model (which is standard practice).
        "type": "categorical",
        "choices": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    },
    "num_prefix_encoder_layers": {
        "type": "int",
        "low": 1,
        "high": 6,
    },
    "num_decoder_layers": {
        "type": "int",
        "low": 1,
        "high": 6,
    },
    "num_heads": {
        "type": "categorical",
        "choices": [2, 4, 8],
    },
    "learning_rate": {
        # Log-uniform: samples are spread evenly on a log scale.
        # This means 1e-5 → 1e-4 gets as many samples as 1e-3 → 1e-2,
        # which is what you want for learning rates.
        "type": "log_float",
        "low": 1e-5,
        "high": 1e-2,
    },
    "weight_decay": {
        "type": "log_float",
        "low": 1e-6,
        "high": 1e-2,
    },
    "dropout": {
        "type": "float",
        "low": 0.0,
        "high": 0.5,
    },
    "lr_schedule": {
        # exponential = original paper (gamma decay each epoch).
        # cosine      = cosine annealing to 0 over all epochs.
        # reduce_on_plateau = halve LR when val_loss stalls for 3 epochs.
        "type": "categorical",
        "choices": ["exponential", "cosine", "reduce_on_plateau"],
    },
    "layer_norm_position": {
        # post_ln = original Transformer (Vaswani et al., 2017).
        # pre_ln  = Pre-LN variant (Xiong et al., 2020); LayerNorm
        #           before each sublayer, final LayerNorm after last block.
        "type": "categorical",
        "choices": ["post_ln", "pre_ln"],
    },
}


# ──────────────────────────────────────────────────────────────────────
# 2. FIXED MODEL PARAMETERS (not tuned)
# ──────────────────────────────────────────────────────────────────────
# These match the paper defaults. We keep them here so there's one
# single source of truth instead of magic numbers scattered around.

FIXED_MODEL_PARAMS = {
    "layernorm_embeds": True,
    "outcome_bool": False,
    "remaining_runtime_head": True,  # Always True per the paper
}


# ──────────────────────────────────────────────────────────────────────
# 3. FIXED TRAINING PARAMETERS (not tuned — but still configurable)
# ──────────────────────────────────────────────────────────────────────

FIXED_TRAINING_PARAMS = {
    # ExponentialLR gamma — decays the LR by this factor each epoch
    "lr_decay_factor": 0.96,
    # Gradient clipping max norm
    "max_norm": 2.0,
    # How often to print batch-level training stats
    "batch_interval": 800,
    # Batch size for validation inference (large = faster, uses more VRAM)
    "val_batch_size": 4096,
}


# ──────────────────────────────────────────────────────────────────────
# 4. RANDOM SEARCH CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

RANDOM_SEARCH_CONFIG = {
    "n_trials": 20,
    "n_epochs": 15,
    "seed": 42,
}


# ──────────────────────────────────────────────────────────────────────
# 5. TPE + HYPERBAND CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
# Single-objective TPE with HyperbandPruner.
#
# Composite objective (minimized):
#   composite = MAE_rrt_stand / μ_rrt  +  (1 - DL_similarity) / μ_dl
#
# where μ_rrt and μ_dl are the means of each metric from Phase 1
# (random search), so both terms contribute ~1.0 on average.
#
# Individual metrics stored as user_attrs for post-hoc rank-sum
# model selection (as in the base SuTraN paper).
#
# Hyperband settings justified by Phase 1 random search:
#   - min_resource=5: rank ordering was stable from epoch 5 onward
#   - max_resource=100: matches realistic full training (early stopping
#     in the base model consistently triggers at epoch ≤100)
#   - reduction_factor=3: standard default (Li et al., 2018, JMLR)
#     produces rungs at epochs 5, 15, 45, 100

TPE_CONFIG = {
    "n_trials": 150,          # upper bound; also capped by timeout
    "timeout": 72000,          # 20 hours in seconds
    "max_epochs": 50,          # Hyperband max_resource
    "min_resource": 5,         # Hyperband min_resource (grace period)
    "reduction_factor": 3,     # Hyperband η
    "seed": 42,
    "n_startup_trials": 5,     # random exploration before TPE kicks in
}


# ──────────────────────────────────────────────────────────────────────
# 6. DATASET CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

DATASET_CONFIG = {
    # Name of the preprocessed event log folder.
    # Must match the log_name used in Preprocessing/from_log_to_tensors.py
    "log_name": "BPIC_17_DR",
}
