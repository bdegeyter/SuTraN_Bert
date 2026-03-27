"""
HPO Pipeline: TPE + Hyperband
==============================

Runs single-objective TPE with Hyperband pruning.

  TPE + Hyperband:
      Uses Tree-structured Parzen Estimator sampling with Hyperband
      early stopping.  The composite objective is normalized using
      mu_rrt and mu_dl derived from a prior random search CSV if one
      exists; otherwise falls back to mu=1.0.  Results are saved to:
          results/tpe_{log_name}/

Usage (from the project root):
    python -m OptunaTuning.run_pipeline

The dataset is determined by DATASET_CONFIG["log_name"] in config.py.
Resumable: if interrupted, re-running this script will pick up where
it left off (SQLite storage).
"""

import time

from OptunaTuning.tpe_search import main as tpe_search_main
from OptunaTuning.config import DATASET_CONFIG


def main():
    log_name = DATASET_CONFIG["log_name"]

    print("=" * 60)
    print(f"HPO PIPELINE — dataset: {log_name}")
    print("=" * 60)

    # TPE + Hyperband (normalization computed from random search CSV if
    # available; falls back to mu=1.0 if no Phase 1 results exist).
    print("\n" + "─" * 60)
    print("TPE + Hyperband (normalized composite)")
    print("─" * 60 + "\n")

    t0 = time.time()
    tpe_search_main()
    t1 = time.time()
    print(f"\n  TPE finished in {(t1 - t0) / 60:.1f} minutes.")

    total = (t1 - t0) / 60
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE — total time: {total:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
