"""
HPO Pipeline: Random Search → TPE + Hyperband
==============================================

Runs the full two-phase hyperparameter optimization pipeline:

  Phase 1 — Random Search (LHS):
      Explores the search space uniformly using Latin Hypercube Sampling.
      All trials run for the full number of epochs (no pruning), producing
      complete learning curves.  Results are saved to:
          results/random_search_{log_name}/

  Phase 2 — TPE + Hyperband:
      Uses Tree-structured Parzen Estimator sampling with Hyperband
      early stopping.  The composite objective is normalized using the
      means from Phase 1.  Results are saved to:
          results/tpe_{log_name}/

Usage (from the project root):
    python -m OptunaTuning.run_pipeline

The dataset is determined by DATASET_CONFIG["log_name"] in config.py.
Both phases are resumable: if interrupted, re-running this script will
pick up where it left off (SQLite storage).
"""

import time

from OptunaTuning.random_search import main as random_search_main
from OptunaTuning.tpe_search import main as tpe_search_main
from OptunaTuning.config import DATASET_CONFIG


def main():
    log_name = DATASET_CONFIG["log_name"]

    print("=" * 60)
    print(f"HPO PIPELINE — dataset: {log_name}")
    print("=" * 60)

    # ── Phase 1: Random Search ──
    print("\n" + "─" * 60)
    print("PHASE 1: Random Search (LHS)")
    print("─" * 60 + "\n")

    t0 = time.time()
    random_search_main()
    t1 = time.time()
    print(f"\n  Phase 1 finished in {(t1 - t0) / 60:.1f} minutes.")

    # ── Phase 2: TPE + Hyperband ──
    print("\n" + "─" * 60)
    print("PHASE 2: TPE + Hyperband (normalized composite)")
    print("─" * 60 + "\n")

    t2 = time.time()
    tpe_search_main()
    t3 = time.time()
    print(f"\n  Phase 2 finished in {(t3 - t2) / 60:.1f} minutes.")

    # ── Done ──
    total = (t3 - t0) / 60
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE — total time: {total:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
