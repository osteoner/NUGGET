from html import parser
import random
import os
import pickle
import sys
import argparse
import json
import torch
from typing import Any
import numpy as np
from torch_geometric.data import TemporalData
import pandas as pd
import torch


def set_random_seed(random_seed: int):
    r"""
    set random seed for reproducibility
    Args:
        random_seed (int): random seed
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'INFO: fixed random seed: {random_seed}')


def get_args():
    parser = argparse.ArgumentParser('*** NUGGET ***')

    # Learning rate
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)

    # Batch size (VERY IMPORTANT)
    parser.add_argument('--bs', type=int, help='Batch size', default=200)

    # Ranking metric k
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)

    # Number of epochs
    parser.add_argument('--num_epoch', type=int, help='Number of epochs', default=50)

    # Seeds
    parser.add_argument('--seed', type=int, help='Random seed', default=42)

    # TGAT embedding sizes (CRITICAL)
    parser.add_argument('--mem_dim', type=int, help='Memory dimension', default=128)
    parser.add_argument('--time_dim', type=int, help='Time encoding dimension', default=128)
    parser.add_argument('--emb_dim', type=int, help='Node embedding dimension', default=128)
    parser.add_argument('--dropout', type=float, help='Dropout rate', default=0.1)


    # Early stopping
    parser.add_argument('--tolerance', type=float, help='Early stopper tolerance', default=1e-4)
    parser.add_argument('--patience', type=int, help='Early stopper patience', default=5)

    # Number of runs (averaging results)
    parser.add_argument('--num_run', type=int, help='Number of iteration runs', default=1)
    parser.add_argument('--train_ratio', type=float, help='Train set ratio', default=0.60)
    parser.add_argument('--val_ratio', type=float, help='Validation set ratio', default=0.10)

    # Optional loading of saved weights
    parser.add_argument("--model-path", type=str, default=None)


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv




def save_results(new_results: dict, filename: str):
    r"""
    save (new) results into a json file
    :param: new_results (dictionary): a dictionary of new results to be saved
    :filename: the name of the file to save the (new) results
    """
    if os.path.isfile(filename):
        # append to the file
        with open(filename, 'r+') as json_file:
            file_data = json.load(json_file)
            # convert file_data to list if not
            if type(file_data) is dict:
                file_data = [file_data]
            file_data.append(new_results)
            json_file.seek(0)
            json.dump(file_data, json_file, indent=4)
    else:
        # dump the results
        with open(filename, 'w') as json_file:
            json.dump(new_results, json_file, indent=4)


