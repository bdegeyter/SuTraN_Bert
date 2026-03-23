"""
Train & Evaluate with Fixed Hyperparameters
=============================================

Run a complete train → validate → test pipeline for a specific set of
hyperparameters on a specific dataset.  No Optuna involved — this is
for producing final thesis numbers after HPO is done.

Usage examples (from project root):

  # With the best params found by TPE:
  python -m OptunaTuning.train_and_evaluate

  # Or import and call programmatically:
  from OptunaTuning.train_and_evaluate import train_and_evaluate
  results = train_and_evaluate(
      params={"d_model": 96, ...},
      log_name="BPIC_19",
      n_epochs=100,
      seed=42,
  )

Results are saved to OptunaTuning/eval_results/<log_name>/<run_name>/
which is deliberately separate from the HPO results directory.
"""

import os
import time
import pickle
import csv
import torch
from torch.utils.data import DataLoader

from SuTraN.train_procedure import train_epoch
from SuTraN.train_utils import MultiOutputLoss
from SuTraN.inference_procedure import inference_loop

from OptunaTuning.config import FIXED_MODEL_PARAMS, FIXED_TRAINING_PARAMS
from OptunaTuning.utils import (
    load_data,
    set_seed,
    create_model,
    create_optimizer,
    create_scheduler,
)
from OptunaTuning.objective import validate


# ──────────────────────────────────────────────────────────────────────
# Best-epoch selection (mirrors TRAIN_EVAL_SUTRAN_DA.py logic)
# ──────────────────────────────────────────────────────────────────────

def _select_best_epoch(epoch_metrics):
    """Select the best epoch using rank(MAE_rrt) + rank(1 - DL_similarity).

    This mirrors the base SuTraN paper's model-selection criterion
    (see TRAIN_EVAL_SUTRAN_DA.py lines 271-279).

    Parameters
    ----------
    epoch_metrics : list of dict
        One dict per epoch, each containing at least "MAE_rrt_stand"
        and "DL_similarity".

    Returns
    -------
    best_epoch : int
        The epoch index (0-based) with the lowest rank-sum.
    """
    n = len(epoch_metrics)
    rrt_values = [m["MAE_rrt_stand"] for m in epoch_metrics]
    dl_values = [m["DL_similarity"] for m in epoch_metrics]

    # Rank MAE_rrt (lower is better → ascending)
    rrt_order = sorted(range(n), key=lambda i: rrt_values[i])
    rrt_ranks = [0] * n
    for rank, idx in enumerate(rrt_order):
        rrt_ranks[idx] = rank + 1

    # Rank DL similarity (higher is better → sort descending, then rank)
    dl_order = sorted(range(n), key=lambda i: dl_values[i], reverse=True)
    dl_ranks = [0] * n
    for rank, idx in enumerate(dl_order):
        dl_ranks[idx] = rank + 1

    # Best epoch = lowest rank-sum
    rank_sums = [rrt_ranks[i] + dl_ranks[i] for i in range(n)]
    best_epoch = min(range(n), key=lambda i: rank_sums[i])
    return best_epoch


# ──────────────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────────────

def train_and_evaluate(params, log_name, n_epochs=100, seed=42,
                       run_name=None, patience=24):
    """Train with fixed HPs, select best epoch, evaluate on test set.

    Parameters
    ----------
    params : dict
        Hyperparameters. Must contain all keys that sample_hyperparams()
        would produce (d_model, d_ff_multiplier, num_heads, etc.).
        A "d_ff" key is derived automatically if missing.
    log_name : str
        Dataset folder name (e.g. "BPIC_19").
    n_epochs : int
        Maximum number of training epochs.
    seed : int
        Random seed for reproducibility.
    run_name : str or None
        Subfolder name for results. Defaults to "seed_{seed}".
    patience : int
        Stop training after this many epochs without improvement in
        any of (MAE_rrt_stand, DL_similarity). Matches the base
        SuTraN paper's early-stopping logic.

    Returns
    -------
    results : dict
        Contains "best_epoch", "val_metrics", "test_metrics",
        "all_epoch_metrics", "params", and "results_dir".
    """
    if run_name is None:
        run_name = f"seed_{seed}"

    # Ensure d_ff is derived
    if "d_ff" not in params:
        params["d_ff"] = params["d_model"] * params["d_ff_multiplier"]

    # ── Output directory ──
    results_dir = os.path.join("OptunaTuning", "eval_results", log_name, run_name)
    os.makedirs(results_dir, exist_ok=True)

    # ── Seed ──
    set_seed(seed)

    # ── Load data (includes test set) ──
    print(f"Loading data for '{log_name}'...")
    data = load_data(log_name)
    print(f"  {data['num_activities']} activities, "
          f"{len(data['train_dataset'])} training instances")

    # Load test set separately (load_data only loads train + val)
    test_dataset = torch.load(os.path.join(log_name, "test_tensordataset.pt"))

    remaining_runtime_head = FIXED_MODEL_PARAMS["remaining_runtime_head"]
    outcome_bool = FIXED_MODEL_PARAMS["outcome_bool"]

    # ── Build model / optimizer / scheduler / loss ──
    model = create_model(params, data)
    optimizer = create_optimizer(model, params)
    scheduler = create_scheduler(optimizer, params, n_epochs)

    num_classes = data["num_activities"]
    loss_weights = (
        1.0,
        params.get("weight_ttne", 1.0),
        params.get("weight_rrt", 1.0),
    )
    label_smoothing = params.get("label_smoothing", 0.0)
    loss_fn = MultiOutputLoss(
        num_classes, remaining_runtime_head, outcome_bool,
        weights=loss_weights, label_smoothing=label_smoothing,
    )

    train_loader = DataLoader(
        data["train_dataset"],
        batch_size=params.get("batch_size", 512),
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=2,
    )

    # ── Training loop with early stopping ──
    all_epoch_metrics = []
    best_mae_rrt = float("inf")
    best_dl_sim = -float("inf")
    epochs_without_improvement = 0
    best_model_state = None
    start_time = time.time()

    for epoch in range(n_epochs):
        torch.manual_seed(epoch)
        epoch_start = time.time()

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
        val_metrics["train_loss"] = _last_running_avgs[0]
        val_metrics["epoch"] = epoch
        val_metrics["wall_clock_seconds"] = time.time() - start_time
        all_epoch_metrics.append(val_metrics)

        # Early stopping check (same logic as base SuTraN: improvement
        # in ANY of the tracked metrics resets the patience counter)
        improved = False
        if val_metrics["MAE_rrt_stand"] < best_mae_rrt:
            best_mae_rrt = val_metrics["MAE_rrt_stand"]
            improved = True
        if val_metrics["DL_similarity"] > best_dl_sim:
            best_dl_sim = val_metrics["DL_similarity"]
            improved = True

        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Step scheduler
        composite = val_metrics["MAE_rrt_stand"] + (1.0 - val_metrics["DL_similarity"])
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(composite)
        else:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        print(
            f"  Epoch {epoch}/{n_epochs - 1} ({epoch_time:.1f}s) | "
            f"MAE_rrt={val_metrics['MAE_rrt_stand']:.4f} | "
            f"DL_sim={val_metrics['DL_similarity']:.4f} | "
            f"patience={epochs_without_improvement}/{patience}"
        )

        # Save checkpoint every epoch (for best-epoch selection later)
        ckpt_path = os.path.join(results_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)

        if epochs_without_improvement >= patience:
            print(f"  Early stopping after {epoch + 1} epochs "
                  f"(no improvement for {patience} epochs).")
            break

    # ── Select best epoch ──
    best_epoch = _select_best_epoch(all_epoch_metrics)
    print(f"\n  Best epoch (rank-sum criterion): {best_epoch}")
    print(f"    MAE_rrt_stand  = {all_epoch_metrics[best_epoch]['MAE_rrt_stand']:.4f}")
    print(f"    DL_similarity  = {all_epoch_metrics[best_epoch]['DL_similarity']:.4f}")

    # ── Reload best model ──
    best_ckpt_path = os.path.join(results_dir, f"model_epoch_{best_epoch}.pt")
    checkpoint = torch.load(best_ckpt_path)
    model_for_test = create_model(params, data)
    model_for_test.load_state_dict(checkpoint["model_state_dict"])
    model_for_test.eval()

    # ── Test-set evaluation ──
    print("\n  Evaluating on test set...")
    test_inf = inference_loop(
        model=model_for_test,
        inference_dataset=test_dataset,
        remaining_runtime_head=remaining_runtime_head,
        outcome_bool=outcome_bool,
        num_categoricals_pref=data["num_categoricals_pref"],
        mean_std_ttne=data["mean_std_ttne"],
        mean_std_tsp=data["mean_std_tsp"],
        mean_std_tss=data["mean_std_tss"],
        mean_std_rrt=data["mean_std_rrt"],
        results_path=results_dir,
        val_batch_size=FIXED_TRAINING_PARAMS["val_batch_size"],
    )

    test_metrics = {
        "MAE_ttne_stand": test_inf[0],
        "MAE_ttne_minutes": test_inf[1],
        "DL_similarity": test_inf[2],
        "perc_too_early": test_inf[3],
        "perc_too_late": test_inf[4],
        "perc_correct": test_inf[5],
        "mean_absolute_length_diff": test_inf[6],
        "mean_too_early": test_inf[7],
        "mean_too_late": test_inf[8],
        "MAE_rrt_stand": test_inf[9],
        "MAE_rrt_minutes": test_inf[10],
    }

    print("\n  Test set results:")
    print(f"    MAE_rrt:       {test_metrics['MAE_rrt_stand']:.4f} "
          f"({test_metrics['MAE_rrt_minutes']:.2f} min)")
    print(f"    DL_similarity: {test_metrics['DL_similarity']:.4f}")
    print(f"    MAE_ttne:      {test_metrics['MAE_ttne_stand']:.4f} "
          f"({test_metrics['MAE_ttne_minutes']:.2f} min)")

    # ── Save results ──
    results = {
        "params": params,
        "log_name": log_name,
        "seed": seed,
        "n_epochs_trained": len(all_epoch_metrics),
        "best_epoch": best_epoch,
        "val_metrics_best_epoch": all_epoch_metrics[best_epoch],
        "test_metrics": test_metrics,
        "all_epoch_metrics": all_epoch_metrics,
        "results_dir": results_dir,
    }

    # Pickle with everything
    with open(os.path.join(results_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # CSV of per-epoch training progress
    fieldnames = list(all_epoch_metrics[0].keys())
    with open(os.path.join(results_dir, "training_progress.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_epoch_metrics)

    # Summary CSV (single row — easy to append across runs)
    summary_path = os.path.join(results_dir, "summary.csv")
    summary_row = {
        "log_name": log_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "n_epochs_trained": len(all_epoch_metrics),
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"val_{k}": v for k, v in all_epoch_metrics[best_epoch].items()
           if k not in ("epoch", "wall_clock_seconds", "train_loss")},
    }
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    # Clean up checkpoints (keep only the best)
    for epoch_idx in range(len(all_epoch_metrics)):
        if epoch_idx != best_epoch:
            ckpt = os.path.join(results_dir, f"model_epoch_{epoch_idx}.pt")
            if os.path.exists(ckpt):
                os.remove(ckpt)

    print(f"\n  Results saved to {results_dir}/")
    return results


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

# Paper defaults for baseline comparison
PAPER_DEFAULTS = {
    "d_model": 32,
    "d_ff_multiplier": 4,
    "num_prefix_encoder_layers": 4,
    "num_decoder_layers": 4,
    "num_heads": 8,
    "learning_rate": 0.0002,
    "weight_decay": 0.0001,
    "dropout": 0.2,
    "activation": "relu",
    "optimizer": "adamw",
    "lr_schedule": "exponential",
    "batch_size": 512,
    "weight_ttne": 1.0,
    "weight_rrt": 1.0,
    "label_smoothing": 0.0,
}


def main():
    """Run evaluation with paper defaults on the configured dataset.

    For custom params, import train_and_evaluate() and call directly.
    """
    from OptunaTuning.config import DATASET_CONFIG

    log_name = DATASET_CONFIG["log_name"]
    print(f"Running paper-default evaluation on {log_name}...")
    results = train_and_evaluate(
        params=PAPER_DEFAULTS.copy(),
        log_name=log_name,
        n_epochs=200,
        seed=42,
        run_name="paper_defaults",
        patience=24,
    )
    return results


if __name__ == "__main__":
    main()
