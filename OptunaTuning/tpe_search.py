"""
TPE + Hyperband Hyperparameter Search
======================================

Entry point for running single-objective TPE with Hyperband pruning.

Objective (minimized):
  composite = MAE_rrt_stand / μ_rrt  +  (1 - DL_similarity) / μ_dl

where μ_rrt and μ_dl are the mean values of each metric computed from
a prior random search on the same dataset. This normalization ensures
both metrics contribute equally to the composite, preventing one from
dominating TPE's surrogate model and Hyperband's pruning decisions.

Individual metrics are stored as trial user_attrs for post-hoc
rank-sum analysis (matching the SuTraN paper's selection criterion).

Note: Optuna does not support trial.report() / trial.should_prune()
for multi-objective studies, so we use single-objective to enable
Hyperband pruning.

Pruning:
  HyperbandPruner with min_resource=5, max_resource=100, eta=3.
  Rungs are at epochs 5, 15, 45, 100.

Sampler:
  TPESampler with n_startup_trials random exploration before the
  surrogate model guides search.

Resumability:
  The study is stored in a SQLite database. Interrupted runs can be
  resumed by re-running this script — already-completed trials are
  preserved and skipped.

Usage (from the project root):
    python -m OptunaTuning.tpe_search

Post-run analysis:
  - The SQLite DB contains all trial data and can be loaded with
    optuna.load_study().
  - A CSV with per-trial per-epoch metrics is saved for inspection.
  - select_best_trial() applies the paper's rank-sum criterion
    across completed trials to pick the best configuration.
"""

import csv
import os
import time
import pickle

import optuna

from OptunaTuning.config import TPE_CONFIG, DATASET_CONFIG, FIXED_MODEL_PARAMS, FIXED_TRAINING_PARAMS
from OptunaTuning.utils import load_data, save_results_csv


# ──────────────────────────────────────────────────────────────────────
# Normalization constants from prior random search
# ──────────────────────────────────────────────────────────────────────

def compute_normalization_constants(log_name):
    """Compute mean MAE_rrt_stand and mean (1 - DL_similarity) from
    the random search results CSV for the given dataset.

    These means are used to normalize the composite scalar so that
    both metrics contribute equally.

    Falls back to mu=1.0 for both metrics if no CSV is found (e.g. when
    skipping the random search phase). The 1.0 fallback introduces a
    slight bias toward MAE_rrt (which typically has a higher mean than
    1 - DL_similarity), but the final rank-sum model selection is
    completely unaffected since it uses raw per-metric user_attrs.

    Parameters
    ----------
    log_name : str
        Dataset name (e.g. "BPIC_17_DR").

    Returns
    -------
    mu_rrt : float
    mu_dl : float
    """
    csv_path = os.path.join(
        "OptunaTuning", "results", f"random_search_{log_name}",
        "random_search_results.csv",
    )
    if not os.path.exists(csv_path):
        print(
            f"  [INFO] No random search CSV found for '{log_name}'. "
            f"Using mu_rrt=1.0, mu_dl=1.0 (unnormalized composite). "
            f"Final rank-sum selection is unaffected."
        )
        return 1.0, 1.0

    rrt_vals, dl_vals = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rrt_vals.append(float(row["MAE_rrt_stand"]))
            dl_vals.append(1.0 - float(row["DL_similarity"]))

    if not rrt_vals:
        print(f"  [WARN] Random search CSV at {csv_path} is empty. Using mu=1.0 fallback.")
        return 1.0, 1.0

    mu_rrt = sum(rrt_vals) / len(rrt_vals)
    mu_dl = sum(dl_vals) / len(dl_vals)

    return mu_rrt, mu_dl


# ──────────────────────────────────────────────────────────────────────
# Post-hoc rank-sum analysis
# ──────────────────────────────────────────────────────────────────────

def select_best_trial(completed_trials):
    """Select the best configuration using the paper's rank-sum criterion.

    Ranks all completed trials by MAE_rrt_stand and by (1 - DL_similarity)
    independently, then picks the trial with the lowest rank-sum.
    Individual metric values are read from trial.user_attrs.

    Parameters
    ----------
    completed_trials : list of optuna.trial.FrozenTrial

    Returns
    -------
    best_trial : optuna.trial.FrozenTrial
    rank_df : list of dict
        All trials with their rank-sum scores, sorted best-first.
    """
    import math

    candidates = []
    for t in completed_trials:
        mae_rrt = t.user_attrs.get("best_MAE_rrt_stand", float("nan"))
        dl_sim = t.user_attrs.get("best_DL_similarity", float("nan"))
        if math.isnan(mae_rrt) or math.isnan(dl_sim):
            continue
        one_minus_dl = 1.0 - dl_sim
        candidates.append({"trial": t, "mae_rrt": mae_rrt, "one_minus_dl": one_minus_dl, "dl_sim": dl_sim})

    if not candidates:
        raise ValueError("No valid (non-NaN) completed trials found.")

    n = len(candidates)
    sorted_by_rrt = sorted(range(n), key=lambda i: candidates[i]["mae_rrt"])
    sorted_by_dl  = sorted(range(n), key=lambda i: candidates[i]["one_minus_dl"])

    rrt_ranks = [0] * n
    dl_ranks  = [0] * n
    for rank, idx in enumerate(sorted_by_rrt):
        rrt_ranks[idx] = rank + 1
    for rank, idx in enumerate(sorted_by_dl):
        dl_ranks[idx] = rank + 1

    for i, c in enumerate(candidates):
        c["rrt_rank"]  = rrt_ranks[i]
        c["dl_rank"]   = dl_ranks[i]
        c["rank_sum"]  = rrt_ranks[i] + dl_ranks[i]

    candidates.sort(key=lambda c: c["rank_sum"])
    best = candidates[0]

    rank_df = [
        {
            "trial_number": c["trial"].number,
            "MAE_rrt_stand": c["mae_rrt"],
            "1_minus_DL": c["one_minus_dl"],
            "DL_similarity": c["dl_sim"],
            "rrt_rank": c["rrt_rank"],
            "dl_rank": c["dl_rank"],
            "rank_sum": c["rank_sum"],
        }
        for c in candidates
    ]

    return best["trial"], rank_df


def save_ranked_csv(rank_df, save_path):
    """Write the ranked trial analysis to a CSV file."""
    import csv
    if not rank_df:
        return
    fieldnames = list(rank_df[0].keys())
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rank_df)
    print(f"  Ranked trials CSV saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Single-objective objective with Hyperband pruning
# ──────────────────────────────────────────────────────────────────────

def create_tpe_objective(data, max_epochs, mu_rrt, mu_dl, results_collector=None):
    """Create the objective function for single-objective TPE + Hyperband.

    Minimizes composite = MAE_rrt_stand / mu_rrt + (1 - DL_similarity) / mu_dl.
    The normalization constants mu_rrt and mu_dl are computed from a prior
    random search, so both terms contribute ~1.0 on average.

    Individual metrics are stored as trial.user_attrs for post-hoc
    rank-sum analysis.

    Parameters
    ----------
    data : dict
        From load_data().
    max_epochs : int
        Maximum training epochs (= Hyperband max_resource).
    mu_rrt : float
        Mean MAE_rrt_stand from random search (normalization constant).
    mu_dl : float
        Mean (1 - DL_similarity) from random search (normalization constant).
    results_collector : dict or None
        If provided, per-trial per-epoch metrics are stored here.

    Returns
    -------
    objective : callable
        Signature: objective(trial) -> float
    """
    # Import here to avoid circular imports
    from OptunaTuning.config import FIXED_MODEL_PARAMS, FIXED_TRAINING_PARAMS, TPE_CONFIG
    from OptunaTuning.utils import set_seed, sample_hyperparams, create_model, create_optimizer, create_scheduler
    from OptunaTuning.objective import validate
    from SuTraN.train_procedure import train_epoch
    from SuTraN.train_utils import MultiOutputLoss
    import torch
    from torch.utils.data import DataLoader

    remaining_runtime_head = FIXED_MODEL_PARAMS["remaining_runtime_head"]
    outcome_bool = FIXED_MODEL_PARAMS["outcome_bool"]

    def objective(trial):
        """Single TPE trial: sample, train, validate, return composite scalar."""

        # ── 1. Sample hyperparameters ──
        params = sample_hyperparams(trial)

        # ── 2. Seed ──
        trial_seed = TPE_CONFIG["seed"] + trial.number
        set_seed(trial_seed)

        # ── 3. Build model / optimizer / scheduler / loss ──
        model = create_model(params, data)
        optimizer = create_optimizer(model, params)
        scheduler = create_scheduler(optimizer, params, max_epochs)
        num_classes = data["num_activities"]
        loss_fn = MultiOutputLoss(num_classes, remaining_runtime_head, outcome_bool)

        # ── 4. DataLoader ──
        train_loader = DataLoader(
            data["train_dataset"],
            batch_size=512,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2,
        )

        # ── 5. Epoch loop ──
        best_composite = float("inf")
        best_mae_rrt   = float("inf")
        best_dl_sim    = -float("inf")
        trial_start    = time.time()

        for epoch in range(max_epochs):
            torch.manual_seed(epoch)

            # Train
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

            # Validate
            model.eval()
            val_metrics = validate(model, data["val_dataset"], data)
            mae_rrt  = val_metrics["MAE_rrt_stand"]
            dl_sim   = val_metrics["DL_similarity"]

            composite = mae_rrt / mu_rrt + (1.0 - dl_sim) / mu_dl

            # Track best composite across epochs
            if composite < best_composite:
                best_composite = composite
                best_mae_rrt   = mae_rrt
                best_dl_sim    = dl_sim

            # Report to Hyperband pruner
            trial.report(composite, epoch)

            # Collect per-epoch data
            if results_collector is not None:
                if trial.number not in results_collector:
                    results_collector[trial.number] = {
                        "params": params,
                        "epochs": [],
                        "wall_clock_start": trial_start,
                    }
                results_collector[trial.number]["epochs"].append(
                    {
                        "epoch": epoch,
                        "train_loss": _last_running_avgs[0],
                        "wall_clock_seconds": time.time() - trial_start,
                        **val_metrics,
                    }
                )

            # Check Hyperband pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Step LR scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(composite)
            else:
                scheduler.step()

            print(
                f"  Trial {trial.number} | Epoch {epoch}/{max_epochs - 1} | "
                f"MAE_rrt={mae_rrt:.4f} | DL_sim={dl_sim:.4f} | "
                f"composite={composite:.4f}"
            )

        # ── 6. Store individual metrics as user_attrs for post-hoc analysis ──
        trial.set_user_attr("best_MAE_rrt_stand", best_mae_rrt)
        trial.set_user_attr("best_DL_similarity", best_dl_sim)

        # Return composite scalar (minimized by TPE)
        return best_composite

    return objective


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    # ── Configuration ──
    log_name      = DATASET_CONFIG["log_name"]
    n_trials      = TPE_CONFIG["n_trials"]
    timeout       = TPE_CONFIG["timeout"]
    max_epochs    = TPE_CONFIG["max_epochs"]
    min_resource  = TPE_CONFIG["min_resource"]
    reduction_factor = TPE_CONFIG["reduction_factor"]
    seed          = TPE_CONFIG["seed"]

    # ── Output directory (dataset-specific) ──
    results_dir = os.path.join("OptunaTuning", "results", f"tpe_{log_name}")
    os.makedirs(results_dir, exist_ok=True)

    # ── Load data ──
    print(f"Loading data for '{log_name}'...")
    data = load_data(log_name)
    print(
        f"  {data['num_activities']} activities, "
        f"{len(data['train_dataset'])} training instances"
    )

    # ── Normalization constants from random search ──
    mu_rrt, mu_dl = compute_normalization_constants(log_name)
    print(f"  Normalization constants: mu_rrt={mu_rrt:.4f}, mu_dl={mu_dl:.4f}")

    # ── Build study ──
    #
    # Single-objective: minimize composite = MAE_rrt_stand/mu_rrt + (1-DL_sim)/mu_dl
    # Individual metrics stored as user_attrs for rank-sum analysis.
    #
    # HyperbandPruner: uses composite from trial.report().
    #   min_resource  = 5   -> no pruning before epoch 5
    #   max_resource  = 100 -> matches realistic full training length
    #   reduction_factor = 3 -> rungs at 5, 15, 45, 100 (Li et al. 2018)
    #
    # storage: SQLite for resumability.
    storage_path = os.path.join(results_dir, "tpe_study.db")
    storage_url  = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=f"sutran_tpe_{log_name}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=TPE_CONFIG["n_startup_trials"],
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=min_resource,
            max_resource=max_epochs,
            reduction_factor=reduction_factor,
        ),
        storage=storage_url,
        load_if_exists=True,
    )

    # ── Resumability check ──
    n_complete_before = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    n_pruned_before = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ])

    if n_complete_before > 0:
        print(
            f"  Resuming: {n_complete_before} complete, {n_pruned_before} pruned, "
            f"remaining trials capped by timeout={timeout // 3600}h."
        )
    else:
        print(
            f"  Starting fresh: up to {n_trials} trials, "
            f"max {max_epochs} epochs each, timeout={timeout // 3600}h."
        )

    # ── Create objective + results collector ──
    results_collector = {}
    objective = create_tpe_objective(data, max_epochs, mu_rrt, mu_dl, results_collector)

    # ── Run ──
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True,   # free GPU memory between trials
    )

    # ── Save per-epoch results ──
    pkl_path = os.path.join(results_dir, "tpe_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results_collector, f)
    print(f"  Per-epoch results saved to {pkl_path}")

    csv_path = os.path.join(results_dir, "tpe_results.csv")
    save_results_csv(results_collector, csv_path)

    # ── Post-hoc rank-sum analysis ──
    print("\n" + "=" * 60)
    print("TPE SEARCH COMPLETE")
    print("=" * 60)

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    n_pruned_total = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ])
    print(
        f"  Total trials: {len(study.trials)}  |  "
        f"Complete: {len(completed_trials)}  |  "
        f"Pruned: {n_pruned_total}"
    )

    if completed_trials:
        # Best by composite (Optuna's best_trial)
        opt_best = study.best_trial
        print(f"\n  Best composite trial: Trial {opt_best.number}")
        print(f"    composite      = {opt_best.value:.4f}")
        print(f"    MAE_rrt_stand  = {opt_best.user_attrs.get('best_MAE_rrt_stand', 'N/A')}")
        print(f"    DL_similarity  = {opt_best.user_attrs.get('best_DL_similarity', 'N/A')}")

        # Rank-sum analysis across all completed trials
        best_trial, rank_df = select_best_trial(completed_trials)

        print(f"\n  Best trial (rank-sum criterion): Trial {best_trial.number}")
        print(f"    MAE_rrt_stand  = {best_trial.user_attrs.get('best_MAE_rrt_stand', 'N/A')}")
        print(f"    DL_similarity  = {best_trial.user_attrs.get('best_DL_similarity', 'N/A')}")
        print("    Hyperparameters:")
        for k, v in best_trial.params.items():
            print(f"      {k}: {v}")

        print(f"\n  All completed trials (rank-sum order):")
        for row in rank_df:
            print(
                f"    Trial {row['trial_number']:>3d}  "
                f"MAE_rrt={row['MAE_rrt_stand']:.4f}  "
                f"DL_sim={row['DL_similarity']:.4f}  "
                f"rank_sum={row['rank_sum']}"
            )

        ranked_csv = os.path.join(results_dir, "tpe_ranked_trials.csv")
        save_ranked_csv(rank_df, ranked_csv)

        # Save the best configuration as a pickle for easy loading
        best_params_path = os.path.join(results_dir, "tpe_best_params.pkl")
        with open(best_params_path, "wb") as f:
            pickle.dump(best_trial.params, f)
        print(f"\n  Best params saved to {best_params_path}")

    return study


if __name__ == "__main__":
    main()
