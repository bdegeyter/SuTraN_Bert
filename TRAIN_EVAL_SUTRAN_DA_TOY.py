"""Toy version of TRAIN_EVAL_SUTRAN_DA.py

Trains and evaluates SuTraN on the toy (150-case) subset of BPIC19.
All heavy parameters are scaled down so this can run on CPU in minutes.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle

# Re-use the checkpoint loader from the original module
from TRAIN_EVAL_SUTRAN_DA import load_checkpoint


def train_eval_toy(log_name='BPIC_19_TOY'):
    """Training and evaluating SuTraN on a small toy dataset.

    Uses reduced model dimensions, fewer epochs and small batch sizes
    so the full pipeline completes on CPU in a reasonable time.

    Parameters
    ----------
    log_name : str
        Name of the toy event log folder produced by
        ``create_BPIC19_TOY_data.py``.
    """

    # ---------------------------------------------------------------- #
    #  Helper                                                           #
    # ---------------------------------------------------------------- #
    def load_dict(path_name):
        with open(path_name, 'rb') as file:
            return pickle.load(file)

    # ---------------------------------------------------------------- #
    #  Load pre-computed metadata / dictionaries                        #
    # ---------------------------------------------------------------- #
    temp_string = log_name + '_cardin_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_dict = load_dict(temp_path)
    num_activities = cardinality_dict['concept:name'] + 2
    print("num_activities:", num_activities)

    temp_string = log_name + '_cardin_list_prefix.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_list_prefix = load_dict(temp_path)

    temp_string = log_name + '_cardin_list_suffix.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cardinality_list_suffix = load_dict(temp_path)

    temp_string = log_name + '_num_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    num_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_cat_cols_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    cat_cols_dict = load_dict(temp_path)

    temp_string = log_name + '_train_means_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    train_means_dict = load_dict(temp_path)

    temp_string = log_name + '_train_std_dict.pkl'
    temp_path = os.path.join(log_name, temp_string)
    train_std_dict = load_dict(temp_path)

    mean_std_ttne = [train_means_dict['timeLabel_df'][0],
                     train_std_dict['timeLabel_df'][0]]
    mean_std_tsp = [train_means_dict['suffix_df'][1],
                    train_std_dict['suffix_df'][1]]
    mean_std_tss = [train_means_dict['suffix_df'][0],
                    train_std_dict['suffix_df'][0]]
    mean_std_rrt = [train_means_dict['timeLabel_df'][1],
                    train_std_dict['timeLabel_df'][1]]
    num_numericals_pref = len(num_cols_dict['prefix_df'])
    num_numericals_suf = len(num_cols_dict['suffix_df'])
    num_categoricals_pref = len(cat_cols_dict['prefix_df'])
    num_categoricals_suf = len(cat_cols_dict['suffix_df'])

    # ---------------------------------------------------------------- #
    #  Toy-friendly hyper-parameters                                    #
    #  (smaller model, fewer epochs, small batch)                       #
    # ---------------------------------------------------------------- #
    d_model = 32
    num_prefix_encoder_layers = 2    # reduced from 4
    num_decoder_layers = 2           # reduced from 4
    num_heads = 4                    # reduced from 8
    d_ff = 4 * d_model
    prefix_embedding = False
    layernorm_embeds = True
    outcome_bool = False
    remaining_runtime_head = True

    only_rrt = (not outcome_bool) & remaining_runtime_head
    only_out = outcome_bool & (not remaining_runtime_head)
    both_not = (not outcome_bool) & (not remaining_runtime_head)
    both = outcome_bool & remaining_runtime_head
    log_transformed = False
    num_target_tens = 3

    dropout = 0.2
    batch_size = 32                  # reduced from 128

    # ---------------------------------------------------------------- #
    #  Paths                                                            #
    # ---------------------------------------------------------------- #
    backup_path = os.path.join(log_name, "SUTRAN_DA_results")
    os.makedirs(backup_path, exist_ok=True)

    # ---------------------------------------------------------------- #
    #  Load tensor datasets                                             #
    # ---------------------------------------------------------------- #
    temp_path = os.path.join(log_name, 'train_tensordataset.pt')
    train_dataset = torch.load(temp_path)

    temp_path = os.path.join(log_name, 'val_tensordataset.pt')
    val_dataset = torch.load(temp_path)

    temp_path = os.path.join(log_name, 'test_tensordataset.pt')
    test_dataset = torch.load(temp_path)

    train_dataset = TensorDataset(*train_dataset)

    # ---------------------------------------------------------------- #
    #  Force CPU                                                        #
    # ---------------------------------------------------------------- #
    device = torch.device('cpu')
    print("device:", device)

    # ---------------------------------------------------------------- #
    #  Reproducibility                                                  #
    # ---------------------------------------------------------------- #
    import random
    seed_value = 24
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # ---------------------------------------------------------------- #
    #  Initialise model                                                 #
    # ---------------------------------------------------------------- #
    from SuTraN.SuTraN import SuTraN
    model = SuTraN(num_activities=num_activities,
                   d_model=d_model,
                   cardinality_categoricals_pref=cardinality_list_prefix,
                   num_numericals_pref=num_numericals_pref,
                   num_prefix_encoder_layers=num_prefix_encoder_layers,
                   num_decoder_layers=num_decoder_layers,
                   num_heads=num_heads,
                   d_ff=d_ff,
                   dropout=dropout,
                   remaining_runtime_head=True,
                   layernorm_embeds=layernorm_embeds,
                   outcome_bool=outcome_bool)

    model.to(device)

    # ---------------------------------------------------------------- #
    #  Optimizer & scheduler                                            #
    # ---------------------------------------------------------------- #
    decay_factor = 0.96
    lr = 0.0002
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=decay_factor)

    # ---------------------------------------------------------------- #
    #  Training                                                         #
    # ---------------------------------------------------------------- #
    from SuTraN.train_procedure import train_model
    start_epoch = 0
    num_epochs = 15           # reduced from 200
    num_classes = num_activities
    batch_interval = 100      # reduced from 800

    train_model(model,
                optimizer,
                train_dataset,
                val_dataset,
                start_epoch,
                num_epochs,
                remaining_runtime_head,
                outcome_bool,
                num_classes,
                batch_interval,
                backup_path,
                num_categoricals_pref,
                mean_std_ttne,
                mean_std_tsp,
                mean_std_tss,
                mean_std_rrt,
                batch_size,
                patience=24,
                lr_scheduler_present=True,
                lr_scheduler=lr_scheduler)

    # ---------------------------------------------------------------- #
    #  Load best checkpoint                                             #
    # ---------------------------------------------------------------- #
    model = SuTraN(num_activities=num_activities,
                   d_model=d_model,
                   cardinality_categoricals_pref=cardinality_list_prefix,
                   num_numericals_pref=num_numericals_pref,
                   num_prefix_encoder_layers=num_prefix_encoder_layers,
                   num_decoder_layers=num_decoder_layers,
                   num_heads=num_heads,
                   d_ff=d_ff,
                   dropout=dropout,
                   remaining_runtime_head=True,
                   layernorm_embeds=layernorm_embeds,
                   outcome_bool=outcome_bool)

    model.to(device)

    final_results_path = os.path.join(backup_path, 'backup_results.csv')
    df = pd.read_csv(final_results_path)
    dl_col = 'Activity suffix: 1-DL (validation)'
    rrt_col = 'RRT - mintues MAE validation'
    df['rrt_rank_val'] = df[rrt_col].rank(method='min').astype(int)
    df['dl_rank_val'] = df[dl_col].rank(method='min',
                                        ascending=False).astype(int)
    df['summed_rank_val'] = df['rrt_rank_val'] + df['dl_rank_val']

    row_with_lowest_loss = df.loc[df['summed_rank_val'].idxmin()]
    epoch_value = row_with_lowest_loss['epoch']

    best_epoch_string = 'model_epoch_{}.pt'.format(int(epoch_value))
    best_epoch_path = os.path.join(backup_path, best_epoch_string)

    model, _, _, _ = load_checkpoint(
        model, path_to_checkpoint=best_epoch_path,
        train_or_eval='eval', lr=0.002)
    model.to(device)
    model.eval()

    # ---------------------------------------------------------------- #
    #  Inference on test set                                            #
    # ---------------------------------------------------------------- #
    from SuTraN.inference_procedure import inference_loop

    results_path = os.path.join(backup_path, "TEST_SET_RESULTS")
    os.makedirs(results_path, exist_ok=True)

    inf_results = inference_loop(model,
                                 test_dataset,
                                 remaining_runtime_head,
                                 outcome_bool,
                                 num_categoricals_pref,
                                 mean_std_ttne,
                                 mean_std_tsp,
                                 mean_std_tss,
                                 mean_std_rrt,
                                 results_path=results_path,
                                 val_batch_size=512)

    # ---------------------------------------------------------------- #
    #  Unpack & print metrics                                           #
    # ---------------------------------------------------------------- #
    avg_MAE_ttne_stand, avg_MAE_ttne_minutes = inf_results[:2]
    avg_dam_lev = inf_results[2]
    perc_too_early = inf_results[3]
    perc_too_late = inf_results[4]
    perc_correct = inf_results[5]
    mean_absolute_length_diff = inf_results[6]
    mean_too_early = inf_results[7]
    mean_too_late = inf_results[8]

    if only_rrt:
        avg_MAE_stand_RRT = inf_results[9]
        avg_MAE_minutes_RRT = inf_results[10]

    elif only_out:
        avg_BCE_out = inf_results[9]
        auc_roc = inf_results[10]
        auc_pr = inf_results[11]

    elif both:
        avg_MAE_stand_RRT = inf_results[9]
        avg_MAE_minutes_RRT = inf_results[10]
        avg_BCE_out = inf_results[11]
        auc_roc = inf_results[12]
        auc_pr = inf_results[13]

    print("Avg MAE TTNE: {} (standardized) ; {} (minutes)".format(
        avg_MAE_ttne_stand, avg_MAE_ttne_minutes))
    print("Avg 1-(normalized) DL distance: {}".format(avg_dam_lev))
    print("Suffix END prediction: too early {}, correct {}, too late {}"
          .format(perc_too_early, perc_correct, perc_too_late))
    print("Too early - avg events: {}".format(mean_too_early))
    print("Too late  - avg events: {}".format(mean_too_late))
    print("Avg absolute length diff: {}".format(
        mean_absolute_length_diff))
    if remaining_runtime_head:
        print("Avg MAE RRT: {} (standardized) ; {} (minutes)".format(
            avg_MAE_stand_RRT, avg_MAE_minutes_RRT))
    if outcome_bool:
        print("Avg BCE outcome: {}".format(avg_BCE_out))
        print("AUC-ROC: {}".format(auc_roc))
        print("AUC-PR: {}".format(auc_pr))

    # ---------------------------------------------------------------- #
    #  Persist results                                                  #
    # ---------------------------------------------------------------- #
    avg_results_dict = {"MAE TTNE minutes": avg_MAE_ttne_minutes,
                        "DL sim": avg_dam_lev,
                        "MAE RRT minutes": avg_MAE_minutes_RRT}
    path_name_average_results = os.path.join(results_path,
                                             'averaged_results.pkl')

    results_dict_pref = inf_results[-2]
    results_dict_suf = inf_results[-1]

    path_name_prefix = os.path.join(results_path,
                                    'prefix_length_results_dict.pkl')
    path_name_suffix = os.path.join(results_path,
                                    'suffix_length_results_dict.pkl')
    with open(path_name_prefix, 'wb') as file:
        pickle.dump(results_dict_pref, file)
    with open(path_name_suffix, 'wb') as file:
        pickle.dump(results_dict_suf, file)
    with open(path_name_average_results, 'wb') as file:
        pickle.dump(avg_results_dict, file)

    print("\n[TOY] Done – results saved to", results_path)


if __name__ == '__main__':
    train_eval_toy()
