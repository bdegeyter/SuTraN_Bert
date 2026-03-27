"""
Utility functions for Optuna Hyperparameter Tuning
====================================================

This module contains all the "plumbing" that connects Optuna to the
existing SuTraN codebase:

  - load_data()         → loads preprocessed tensors + metadata
  - set_seed()          → reproducible seeding across all libraries
  - sample_hyperparams()→ asks Optuna for one set of HP values
  - create_model()      → instantiates SuTraN from an HP dict
  - create_optimizer()  → wraps model params in AdamW or NAdam
  - create_scheduler()  → wraps optimizer in one of three LR schedules

Each function is small and does exactly one thing. The objective
function (in objective.py) calls them in sequence.
"""

import os
import random
import pickle
import math
import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube

from OptunaTuning.config import SEARCH_SPACE, FIXED_MODEL_PARAMS, FIXED_TRAINING_PARAMS


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_data(log_name):
    """Load all preprocessed data needed for training and validation.

    This replicates the data-loading logic from TRAIN_EVAL_SUTRAN_DA.py,
    but returns everything in a single dictionary so it can be passed
    around easily.

    Parameters
    ----------
    log_name : str
        Name of the event log folder (e.g. "BPIC_17_DR"). Must contain
        the pickle files and tensor datasets produced by the
        preprocessing pipeline.

    Returns
    -------
    data : dict
        Keys include:
        - "num_activities", "cardinality_list_prefix", etc. (metadata)
        - "mean_std_ttne", "mean_std_tsp", "mean_std_tss", "mean_std_rrt"
        - "train_dataset" (TensorDataset), "val_dataset" (raw tuple)
        - "num_categoricals_pref"
    """
    def _load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Metadata pickles ──
    cardinality_dict = _load_pickle(
        os.path.join(log_name, f"{log_name}_cardin_dict.pkl")
    )
    num_activities = cardinality_dict["concept:name"] + 2

    cardinality_list_prefix = _load_pickle(
        os.path.join(log_name, f"{log_name}_cardin_list_prefix.pkl")
    )
    cardinality_list_suffix = _load_pickle(
        os.path.join(log_name, f"{log_name}_cardin_list_suffix.pkl")
    )
    num_cols_dict = _load_pickle(
        os.path.join(log_name, f"{log_name}_num_cols_dict.pkl")
    )
    cat_cols_dict = _load_pickle(
        os.path.join(log_name, f"{log_name}_cat_cols_dict.pkl")
    )
    train_means_dict = _load_pickle(
        os.path.join(log_name, f"{log_name}_train_means_dict.pkl")
    )
    train_std_dict = _load_pickle(
        os.path.join(log_name, f"{log_name}_train_std_dict.pkl")
    )

    # ── Standardization statistics ──
    mean_std_ttne = [
        train_means_dict["timeLabel_df"][0],
        train_std_dict["timeLabel_df"][0],
    ]
    mean_std_tsp = [
        train_means_dict["suffix_df"][1],
        train_std_dict["suffix_df"][1],
    ]
    mean_std_tss = [
        train_means_dict["suffix_df"][0],
        train_std_dict["suffix_df"][0],
    ]
    mean_std_rrt = [
        train_means_dict["timeLabel_df"][1],
        train_std_dict["timeLabel_df"][1],
    ]

    num_numericals_pref = len(num_cols_dict["prefix_df"])
    num_categoricals_pref = len(cat_cols_dict["prefix_df"])

    # ── Tensor datasets ──
    train_tensors = torch.load(
        os.path.join(log_name, "train_tensordataset.pt")
    )
    val_dataset = torch.load(
        os.path.join(log_name, "val_tensordataset.pt")
    )

    # The training set needs to be a TensorDataset for DataLoader
    train_dataset = torch.utils.data.TensorDataset(*train_tensors)

    return {
        "num_activities": num_activities,
        "cardinality_list_prefix": cardinality_list_prefix,
        "cardinality_list_suffix": cardinality_list_suffix,
        "num_numericals_pref": num_numericals_pref,
        "num_categoricals_pref": num_categoricals_pref,
        "mean_std_ttne": mean_std_ttne,
        "mean_std_tsp": mean_std_tsp,
        "mean_std_tss": mean_std_tss,
        "mean_std_rrt": mean_std_rrt,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
    }


# ──────────────────────────────────────────────────────────────────────
# Seeding
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed):
    """Set random seeds for full reproducibility.

    Covers: Python stdlib, NumPy, PyTorch CPU, PyTorch CUDA.
    Also enables deterministic cuDNN (slightly slower but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
# Hyperparameter sampling
# ──────────────────────────────────────────────────────────────────────

def sample_hyperparams(trial):
    """Use an Optuna trial to sample one complete HP configuration.

    Reads the SEARCH_SPACE from config.py and calls the appropriate
    trial.suggest_* method for each hyperparameter.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The current Optuna trial object.

    Returns
    -------
    params : dict
        A flat dictionary with all sampled hyperparameter values.
        Includes a derived "d_ff" key (= d_model * d_ff_multiplier).
    """
    params = {}

    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif spec["type"] == "log_float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=True
            )

    # Derived parameter: absolute d_ff from multiplier
    params["d_ff"] = int(params["d_model"] * params["d_ff_multiplier"])

    return params


# ──────────────────────────────────────────────────────────────────────
# Model creation
# ──────────────────────────────────────────────────────────────────────

def create_model(params, data):
    """Instantiate a SuTraN model from hyperparameters + data metadata.

    Parameters
    ----------
    params : dict
        Hyperparameters (from sample_hyperparams). Must contain:
        d_model, d_ff, num_prefix_encoder_layers, num_decoder_layers,
        num_heads, dropout.
    data : dict
        Data metadata (from load_data). Must contain:
        num_activities, cardinality_list_prefix, num_numericals_pref.

    Returns
    -------
    model : SuTraN
        Initialized model, moved to GPU if available.
    """
    from SuTraN.SuTraN import SuTraN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SuTraN(
        num_activities=data["num_activities"],
        d_model=params["d_model"],
        cardinality_categoricals_pref=data["cardinality_list_prefix"],
        num_numericals_pref=data["num_numericals_pref"],
        num_prefix_encoder_layers=params["num_prefix_encoder_layers"],
        num_decoder_layers=params["num_decoder_layers"],
        num_heads=params["num_heads"],
        d_ff=params["d_ff"],
        dropout=params["dropout"],
        remaining_runtime_head=FIXED_MODEL_PARAMS["remaining_runtime_head"],
        layernorm_embeds=FIXED_MODEL_PARAMS["layernorm_embeds"],
        outcome_bool=FIXED_MODEL_PARAMS["outcome_bool"],
        pre_ln=params.get("layer_norm_position", "post_ln") == "pre_ln",
    )

    model.to(device)
    return model


# ──────────────────────────────────────────────────────────────────────
# Optimizer creation
# ──────────────────────────────────────────────────────────────────────

def create_optimizer(model, params):
    """Create an AdamW optimizer with the sampled learning rate and weight decay.

    Parameters
    ----------
    model : SuTraN
        The initialized model whose parameters will be optimized.
    params : dict
        Must contain "learning_rate" and "weight_decay".

    Returns
    -------
    optimizer : torch.optim.AdamW
    """
    return torch.optim.AdamW(
        params=model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )


# ──────────────────────────────────────────────────────────────────────
# Learning rate scheduler creation
# ──────────────────────────────────────────────────────────────────────

def create_scheduler(optimizer, params, n_epochs):
    """Create a learning rate scheduler based on the sampled schedule type.

    Three options:
      - "exponential"        : multiply LR by gamma each epoch (paper default)
      - "cosine"             : cosine annealing from initial LR down to 0
      - "reduce_on_plateau"  : halve LR when val_loss stalls for 3 epochs

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    params : dict
        Must contain "lr_schedule" (or defaults to "exponential").
    n_epochs : int
        Total number of training epochs (needed for cosine annealing).

    Returns
    -------
    scheduler : torch.optim.lr_scheduler._LRScheduler
    """
    schedule = params.get("lr_schedule", "exponential")

    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=n_epochs,
        )
    elif schedule == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
    else:  # "exponential" (default)
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=FIXED_TRAINING_PARAMS["lr_decay_factor"],
        )


# ──────────────────────────────────────────────────────────────────────
# Latin Hypercube Sampling
# ──────────────────────────────────────────────────────────────────────

def generate_lhs_configurations(n_trials, seed=42):
    """Generate n_trials HP configurations using Latin Hypercube Sampling.

    LHS divides each parameter's range into n_trials equal-probability
    bins and guarantees that each bin is sampled exactly once. This gives
    much better coverage of the search space than plain random sampling,
    especially when n_trials is small relative to the number of parameters.

    How it works:
      1. scipy generates an (n_trials × n_params) matrix of values in
         [0, 1], with LHS guarantees on uniformity.
      2. We map each [0, 1] value to the actual parameter range, respecting
         the type (categorical → pick from list, int → round, log_float →
         log-scale mapping, etc.).

    Parameters
    ----------
    n_trials : int
        Number of configurations to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    configurations : list of dict
        Each dict maps parameter names to concrete values that can
        be passed to study.enqueue_trial().
    """
    param_names = list(SEARCH_SPACE.keys())
    n_params = len(param_names)

    # Generate LHS samples: shape (n_trials, n_params), values in [0, 1)
    sampler = LatinHypercube(d=n_params, seed=seed)
    unit_samples = sampler.random(n=n_trials)  # (n_trials, n_params)

    configurations = []
    for i in range(n_trials):
        params = {}
        for j, name in enumerate(param_names):
            spec = SEARCH_SPACE[name]
            u = unit_samples[i, j]  # value in [0, 1)

            if spec["type"] == "categorical":
                # Map [0, 1) → index into the choices list
                idx = int(u * len(spec["choices"]))
                idx = min(idx, len(spec["choices"]) - 1)  # safety clamp
                params[name] = spec["choices"][idx]

            elif spec["type"] == "int":
                # Map [0, 1) → integer in [low, high]
                low, high = spec["low"], spec["high"]
                params[name] = int(round(low + u * (high - low)))

            elif spec["type"] == "float":
                # Map [0, 1) → float in [low, high]
                low, high = spec["low"], spec["high"]
                params[name] = low + u * (high - low)

            elif spec["type"] == "log_float":
                # Map [0, 1) → float in [low, high] on log scale.
                # u=0 → low, u=1 → high, uniform in log-space between.
                log_low = math.log(spec["low"])
                log_high = math.log(spec["high"])
                params[name] = math.exp(log_low + u * (log_high - log_low))

        configurations.append(params)

    return configurations


# ──────────────────────────────────────────────────────────────────────
# Results CSV export
# ──────────────────────────────────────────────────────────────────────

def save_results_csv(results_collector, save_path):
    """Flatten per-trial per-epoch results into a CSV file.

    Each row is one (trial, epoch) pair with all hyperparameters and
    metrics as columns.  Useful for quick inspection in Excel / VS Code
    without needing to deserialize a pickle.

    Parameters
    ----------
    results_collector : dict
        {trial_number: {"params": {...}, "epochs": [{...}, ...]}}.
    save_path : str
        Destination CSV path.
    """
    import csv

    # Collect all rows first so we can derive the full set of column names
    rows = []
    for trial_num, trial_data in results_collector.items():
        for epoch_data in trial_data["epochs"]:
            row = {"trial_id": trial_num, **trial_data["params"], **epoch_data}
            rows.append(row)

    if not rows:
        print(f"  No results to save to CSV.")
        return

    fieldnames = list(rows[0].keys())
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Results CSV saved to {save_path}")
