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
        "choices": [2, 3, 4],
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
    "activation": {
        # Activation function for the feed-forward sublayers.
        # relu = original paper, gelu/silu = popular modern alternatives.
        "type": "categorical",
        "choices": ["relu", "gelu", "silu"],
    },
    "optimizer": {
        # AdamW = original paper default. NAdam = Adam + Nesterov momentum,
        # often converges faster on transformer architectures.
        "type": "categorical",
        "choices": ["adamw", "nadam"],
    },
    "lr_schedule": {
        # exponential = original paper (gamma decay each epoch).
        # cosine      = cosine annealing to 0 over all epochs.
        # reduce_on_plateau = halve LR when val_loss stalls for 3 epochs.
        "type": "categorical",
        "choices": ["exponential", "cosine", "reduce_on_plateau"],
    },
    "batch_size": {
        # Training batch size. Larger = faster epochs but more VRAM.
        "type": "categorical",
        "choices": [64, 128, 256, 512],
    },
    "weight_ttne": {
        # Weight of the time-till-next-event MAE loss relative to the
        # cross-entropy activity loss (fixed at 1.0). Log-uniform so
        # that 0.2→1.0 gets as many samples as 1.0→5.0.
        "type": "log_float",
        "low": 0.2,
        "high": 5.0,
    },
    "weight_rrt": {
        # Weight of the remaining-runtime MAE loss relative to the
        # cross-entropy activity loss (fixed at 1.0).
        "type": "log_float",
        "low": 0.2,
        "high": 5.0,
    },
    "label_smoothing": {
        # Smoothing factor for the activity suffix CE loss.
        # 0.0 = hard one-hot targets (paper default).
        # 0.1 = common starting point in NLP / sequence tasks.
        # Softens targets to (1-ε, ε/(K-1), ...), reducing overconfidence.
        "type": "float",
        "low": 0.0,
        "high": 0.3,
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
# 5. DATASET CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

DATASET_CONFIG = {
    # Name of the preprocessed event log folder.
    # Must match the log_name used in Preprocessing/from_log_to_tensors.py
    "log_name": "BPIC_17_DR",
}
