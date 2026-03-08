"""
Optuna Objective Function for SuTraN
======================================

This is the core module that connects Optuna to SuTraN's training loop.

It defines a single function — create_objective() — which returns a
closure that Optuna calls once per trial. Each call:

  1. Samples hyperparameters      (via utils.sample_hyperparams)
  2. Seeds everything             (via utils.set_seed)
  3. Builds model/optimizer/sched (via utils.create_model etc.)
  4. Trains for N epochs, calling the EXISTING train_epoch + inference_loop
  5. Returns the best validation score to Optuna

We do NOT rewrite training or inference logic — we call the exact same
functions that TRAIN_EVAL_SUTRAN_DA.py uses.
"""

import torch
from torch.utils.data import DataLoader

from SuTraN.train_procedure import train_epoch
from SuTraN.train_utils import MultiOutputLoss
from SuTraN.inference_procedure import inference_loop

from OptunaTuning.config import (
    FIXED_MODEL_PARAMS,
    FIXED_TRAINING_PARAMS,
    RANDOM_SEARCH_CONFIG,
)
from OptunaTuning.utils import (
    set_seed,
    sample_hyperparams,
    create_model,
    create_optimizer,
    create_scheduler,
)


# ──────────────────────────────────────────────────────────────────────
# Validation helper
# ──────────────────────────────────────────────────────────────────────

def validate(model, val_dataset, data):
    """Run inference on the validation set and return all metrics.

    Calls the existing inference_loop from SuTraN/inference_procedure.py
    and unpacks the results into a readable dictionary.

    Parameters
    ----------
    model : SuTraN
        Must be in eval mode (model.eval() called before this).
    val_dataset : tuple of torch.Tensor
        The raw validation tensors (NOT a TensorDataset).
    data : dict
        From load_data(). Needs mean_std_* and num_categoricals_pref.

    Returns
    -------
    metrics : dict
        All validation metrics, plus a composite "val_loss" used as
        the Optuna objective.
    """
    remaining_runtime_head = FIXED_MODEL_PARAMS["remaining_runtime_head"]
    outcome_bool = FIXED_MODEL_PARAMS["outcome_bool"]

    inf_results = inference_loop(
        model=model,
        inference_dataset=val_dataset,
        remaining_runtime_head=remaining_runtime_head,
        outcome_bool=outcome_bool,
        num_categoricals_pref=data["num_categoricals_pref"],
        mean_std_ttne=data["mean_std_ttne"],
        mean_std_tsp=data["mean_std_tsp"],
        mean_std_tss=data["mean_std_tss"],
        mean_std_rrt=data["mean_std_rrt"],
        results_path=None,  # Don't save predictions to disk
        val_batch_size=FIXED_TRAINING_PARAMS["val_batch_size"],
    )

    # Unpack the flat list returned by inference_loop.
    # The order is fixed — see SuTraN/inference_procedure.py.
    metrics = {
        "MAE_ttne_stand": inf_results[0],
        "MAE_ttne_minutes": inf_results[1],
        "DL_similarity": inf_results[2],
        "perc_too_early": inf_results[3],
        "perc_too_late": inf_results[4],
        "perc_correct": inf_results[5],
        "mean_absolute_length_diff": inf_results[6],
        "mean_too_early": inf_results[7],
        "mean_too_late": inf_results[8],
    }

    # RRT metrics (remaining_runtime_head is always True)
    metrics["MAE_rrt_stand"] = inf_results[9]
    metrics["MAE_rrt_minutes"] = inf_results[10]

    # ── Composite objective ──
    # Sum of the two standardized MAEs. Both are on the same scale,
    # so an unweighted sum is fair. We log all metrics anyway, so
    # DL similarity can be inspected after the fact.
    metrics["val_loss"] = metrics["MAE_ttne_stand"] + metrics["MAE_rrt_stand"]

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Objective factory
# ──────────────────────────────────────────────────────────────────────

def create_objective(data, n_epochs, results_collector=None):
    """Create an Optuna objective function (a closure).

    Why a closure? The objective needs access to shared data (the
    dataset, the number of epochs, the results collector) but Optuna's
    study.optimize() only passes a single `trial` argument. A closure
    captures the rest.

    Parameters
    ----------
    data : dict
        From load_data(). Contains datasets + metadata.
    n_epochs : int
        How many epochs to train each trial for.
    results_collector : dict or None
        If provided, per-trial per-epoch metrics are stored here.
        Key = trial number, value = list of per-epoch metric dicts.
        Useful for plotting learning curves after the search.

    Returns
    -------
    objective : callable
        A function with signature objective(trial) -> float that
        Optuna will call once per trial.
    """
    remaining_runtime_head = FIXED_MODEL_PARAMS["remaining_runtime_head"]
    outcome_bool = FIXED_MODEL_PARAMS["outcome_bool"]

    def objective(trial):
        """Single Optuna trial: sample HPs → train → validate → return score."""

        # ── 1. Sample hyperparameters ──
        params = sample_hyperparams(trial)

        # ── 2. Seed for reproducibility ──
        # Each trial gets a different but deterministic seed.
        trial_seed = RANDOM_SEARCH_CONFIG["seed"] + trial.number
        set_seed(trial_seed)

        # ── 3. Build model, optimizer, scheduler, loss function ──
        model = create_model(params, data)
        optimizer = create_optimizer(model, params)
        scheduler = create_scheduler(optimizer, params, n_epochs)
        num_classes = data["num_activities"]
        loss_weights = (
            1.0,  # CE weight is fixed at 1.0 (anchor)
            params.get("weight_ttne", 1.0),
            params.get("weight_rrt", 1.0),
        )
        label_smoothing = params.get("label_smoothing", 0.0)
        loss_fn = MultiOutputLoss(
            num_classes, remaining_runtime_head, outcome_bool,
            weights=loss_weights, label_smoothing=label_smoothing,
        )

        # ── 4. Create training DataLoader ──
        train_loader = DataLoader(
            data["train_dataset"],
            batch_size=params.get("batch_size", 128),
            shuffle=True,
            pin_memory=True,
        )

        # ── 5. Epoch loop ──
        best_val_loss = float("inf")

        for epoch in range(n_epochs):
            # Seed the epoch (same as original code — controls shuffle order)
            torch.manual_seed(epoch)

            # --- Train one epoch ---
            model.train()
            model, optimizer, _last_running_avgs = train_epoch(
                model=model,
                training_loader=train_loader,
                remaining_runtime_head=remaining_runtime_head,
                outcome_bool=outcome_bool,
                optimizer=optimizer,
                loss_fn=loss_fn,
                batch_interval=FIXED_TRAINING_PARAMS["batch_interval"],
                epoch_number=epoch,
                max_norm=FIXED_TRAINING_PARAMS["max_norm"],
            )

            # --- Validate ---
            model.eval()
            val_metrics = validate(model, data["val_dataset"], data)
            val_loss = val_metrics["val_loss"]

            # Track the best
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # --- Collect results (optional, for analysis later) ---
            if results_collector is not None:
                if trial.number not in results_collector:
                    results_collector[trial.number] = {
                        "params": params,
                        "epochs": [],
                    }
                results_collector[trial.number]["epochs"].append(
                    {
                        "epoch": epoch,
                        "train_loss": _last_running_avgs[0],
                        **val_metrics,
                    }
                )

            # --- Report to Optuna (used by pruners in Phase 2) ---
            # Even though we don't prune in Phase 1 (random search),
            # we still report. It's harmless now and means the same
            # objective function works for both phases.
            trial.report(val_loss, epoch)

            # Check if the trial should be pruned (killed early).
            # In Phase 1 this always returns False (no pruner).
            # In Phase 2 (Hyperband) this can return True.
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()

            # Step the learning rate scheduler.
            # ReduceLROnPlateau needs the metric; others don't.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            print(
                f"  Trial {trial.number} | Epoch {epoch}/{n_epochs-1} | "
                f"val_loss={val_loss:.4f} | best={best_val_loss:.4f} | "
                f"MAE_ttne={val_metrics['MAE_ttne_minutes']:.2f}min | "
                f"MAE_rrt={val_metrics['MAE_rrt_minutes']:.2f}min | "
                f"DL_sim={val_metrics['DL_similarity']:.4f}"
            )

        # ── 6. Return the best validation loss across all epochs ──
        # Optuna minimizes this value.
        return best_val_loss

    return objective
