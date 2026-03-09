"""Toy version of create_BPIC19_data.py

Creates a small subset (first 150 cases) of the BPIC19 dataset
for quick CPU-based testing, debugging and output inspection.
"""
import pandas as pd
import numpy as np
from Preprocessing.from_log_to_tensors import log_to_tensors
import os
import torch

# Re-use the same preprocessing function from the original script
from create_BPIC19_data import preprocess_bpic19

# ------------------------------------------------------------------ #
#  Number of cases to retain for the toy dataset                      #
# ------------------------------------------------------------------ #
NUM_TOY_CASES = 150


def construct_BPIC19_TOY_datasets():
    df = pd.read_csv(r'BPIC19.csv')
    df = preprocess_bpic19(df)

    # -------------------------------------------------------------- #
    #  Subset to the first NUM_TOY_CASES unique cases                 #
    # -------------------------------------------------------------- #
    unique_cases = df['case:concept:name'].unique()[:NUM_TOY_CASES]
    df = df[df['case:concept:name'].isin(unique_cases)].copy()
    print(f"[TOY] Retained {df['case:concept:name'].nunique()} cases "
          f"({len(df)} events)")

    categorical_casefeatures = ['case:Spend area text', 'case:Company',
                                'case:Document Type',
                                'case:Sub spend area text', 'case:Item',
                                'case:Vendor', 'case:Item Type',
                                'case:Item Category',
                                'case:Spend classification text',
                                'case:GR-Based Inv. Verif.',
                                'case:Goods Receipt']
    numeric_eventfeatures = ['Cumulative net worth (EUR)']
    categorical_eventfeatures = ['org:resource']
    num_casefts = []
    case_id = 'case:concept:name'
    timestamp = 'time:timestamp'
    act_label = 'concept:name'

    start_date = "2018-01"
    end_date = "2019-02"
    max_days = 143.33
    window_size = 17
    log_name = 'BPIC_19_TOY'
    start_before_date = "2018-09"
    test_len_share = 0.25
    val_len_share = 0.2
    mode = 'workaround'
    outcome = None

    result = log_to_tensors(df,
                            log_name=log_name,
                            start_date=start_date,
                            start_before_date=start_before_date,
                            end_date=end_date,
                            max_days=max_days,
                            test_len_share=test_len_share,
                            val_len_share=val_len_share,
                            window_size=window_size,
                            mode=mode,
                            case_id=case_id,
                            act_label=act_label,
                            timestamp=timestamp,
                            cat_casefts=categorical_casefeatures,
                            num_casefts=num_casefts,
                            cat_eventfts=categorical_eventfeatures,
                            num_eventfts=numeric_eventfeatures,
                            outcome=outcome)

    train_data, val_data, test_data = result

    # Create the log_name subfolder
    output_directory = log_name
    os.makedirs(output_directory, exist_ok=True)

    # Save training tuples
    train_tensors_path = os.path.join(output_directory,
                                      'train_tensordataset.pt')
    torch.save(train_data, train_tensors_path)

    # Save validation tuples
    val_tensors_path = os.path.join(output_directory,
                                    'val_tensordataset.pt')
    torch.save(val_data, val_tensors_path)

    # Save test tuples
    test_tensors_path = os.path.join(output_directory,
                                     'test_tensordataset.pt')
    torch.save(test_data, test_tensors_path)

    print(f"[TOY] Saved tensor datasets to '{output_directory}/'")


if __name__ == '__main__':
    construct_BPIC19_TOY_datasets()
