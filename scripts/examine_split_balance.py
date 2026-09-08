"""Examines class balance in predetermined folds.

Split index files use Python pickle for compatibility with Chemprop v1. Pickle
can execute code while loading, so this script must only be used with files
created locally or obtained from a trusted source.
"""

import os
import pickle
from pprint import pprint
import sys
from typing import List
from typing_extensions import Literal

import numpy as np
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import get_data, MoleculeDataset


BASE = '/data/rsg/chemistry/yangk/lsc_experiments_dump_splits/data'
DATASETS = ['pcba', 'muv', 'hiv', 'bace', 'bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'chembl']


class Args(Tap):
    split_type: Literal['random', 'scaffold']  # Split type
    base_dir: str = BASE  # Root containing dataset CSVs and split directories
    datasets: List[str] = DATASETS  # Dataset directory names to inspect

    def process_args(self) -> None:
        self.datasets = list(self.datasets)


def compute_ratios(data: MoleculeDataset) -> np.ndarray:
    if len(data) == 0:
        raise ValueError('Cannot compute class balance for an empty dataset.')
    targets = np.array(data.targets(), dtype=float)
    observed = targets[np.isfinite(targets)]
    if np.any((observed != 0) & (observed != 1)):
        raise ValueError('Class-balance analysis requires binary 0/1 targets.')
    ratios = np.nanmean(targets, axis=0)
    ratios = np.minimum(ratios, 1 - ratios)

    return ratios


def examine_split_balance(
    split_type: str,
    base_dir: str = BASE,
    datasets: List[str] = None,
):
    results = []
    datasets = DATASETS if datasets is None else datasets

    for dataset in datasets:
        # Load task names for the dataset
        data_path = os.path.join(base_dir, dataset, f'{dataset}.csv')
        data = get_data(data_path)

        # Get class balance ratios for full dataset
        ratios = compute_ratios(data)

        # Initialize array of diffs between ratios
        ratio_diffs = []

        # Loop through folds
        split_dir = os.path.join(base_dir, dataset, split_type)
        for fold in sorted(os.listdir(split_dir)):
            # Open fold indices
            with open(os.path.join(split_dir, fold, '0', 'split_indices.pckl'), 'rb') as f:
                indices = pickle.load(f)
            if not isinstance(indices, (list, tuple)) or len(indices) != 3:
                raise ValueError(f'Invalid train/val/test split schema in fold {fold}.')
            test_indices = list(indices[2])
            if any(
                not isinstance(index, (int, np.integer))
                or isinstance(index, (bool, np.bool_))
                or index < 0
                or index >= len(data)
                for index in test_indices
            ):
                raise ValueError(f'Out-of-range test index in fold {fold}.')

            # Get test data
            test_data = MoleculeDataset([data[index] for index in test_indices])

            # Get test ratios
            test_ratios = compute_ratios(test_data)

            # Compute ratio diff
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_diff = np.maximum(ratios / test_ratios, test_ratios / ratios)
            ratio_diff[np.where(np.isinf(ratio_diff))[0]] = np.nan

            # Add ratio diff
            ratio_diffs.append(ratio_diff)

        # Convert to numpy array
        if not ratio_diffs:
            raise ValueError(f'No fold index files found in {split_dir}.')
        ratio_diffs = np.array(ratio_diffs)  # num_folds x num_tasks

        # Determine number of folds and number of failures
        num_folds = len(ratio_diffs)
        num_failures = np.sum(np.isnan(ratio_diffs))

        # Average across tasks
        ratio_diffs = np.nanmean(ratio_diffs, axis=1)  # num_folds

        # Compute mean and standard deviation across folds
        mean, std = np.nanmean(ratio_diffs), np.nanstd(ratio_diffs)

        # Add results
        results.append({
            'dataset': dataset,
            'mean': mean,
            'std': std,
            'num_folds': num_folds,
            'num_failures': num_failures
        })

    pprint(results)


if __name__ == '__main__':
    args = Args().parse_args()

    examine_split_balance(args.split_type, args.base_dir, args.datasets)
