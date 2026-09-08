"""Averages the target values for duplicate smiles strings. (Only used for regression datasets.)"""

from collections import defaultdict
import csv
import os
import sys
from typing import List

import numpy as np
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import get_data, get_task_names, preprocess_smiles_columns


class Args(Tap):
    data_path: str  # Path to data CSV file
    smiles_columns: List[str] = None  # Name of the columns containing SMILES strings. By default, uses the first column.
    target_columns: List[str] = None  # Name of the columns containing target values. By default, uses all columns except the SMILES column.
    save_path: str  # Path where average data CSV file will be saved


def average_duplicates(args: Args):
    """Averages duplicate data points in a dataset."""
    print('Loading data')
    smiles_columns = preprocess_smiles_columns(
        path=args.data_path,
        smiles_columns=args.smiles_columns,
        number_of_molecules=1,
    )
    task_names = get_task_names(
        path=args.data_path,
        smiles_columns=smiles_columns,
        target_columns=args.target_columns,
    )
    data = get_data(path=args.data_path, smiles_columns=args.smiles_columns, target_columns=args.target_columns)
    print(f'Data size = {len(data):,}')
    if len(data) == 0:
        raise ValueError('Cannot average duplicates in an empty dataset.')

    # Map SMILES string to lists of targets
    smiles_in_order = []
    smiles_to_targets = defaultdict(list)
    for smiles, targets in zip(map(tuple, data.smiles(flatten=False)), data.targets()):
        smiles_to_targets[smiles].append(targets)
        if len(smiles_to_targets[smiles]) == 1:
            smiles_in_order.append(smiles)

    # Find duplicates
    duplicate_count = 0
    stds = []
    new_data = []
    for smiles in smiles_in_order:
        all_targets = smiles_to_targets[smiles]
        duplicate_count += len(all_targets) - 1
        num_tasks = len(all_targets[0])

        targets_by_task = [[] for _ in range(num_tasks)]
        for task in range(num_tasks):
            for targets in all_targets:
                if targets[task] is not None:
                    targets_by_task[task].append(targets[task])

        stds.append([np.std(task_targets) if len(task_targets) > 0 else 0.0 for task_targets in targets_by_task])
        means = [np.mean(task_targets) if len(task_targets) > 0 else None for task_targets in targets_by_task]
        new_data.append((smiles, means))

    print(f'Number of duplicates = {duplicate_count:,}')
    mean_stds = np.mean(stds, axis=0)
    print(
        'Duplicate standard deviation per task = '
        + ', '.join(
            f'{task}: {std:.4e}' for task, std in zip(task_names, mean_stds)
        )
    )
    print(f'New data size = {len(new_data):,}')

    # Save new data
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(smiles_columns + task_names)
        for smiles, avg_targets in new_data:
            writer.writerow(
                list(smiles)
                + [value if value is not None else '' for value in avg_targets]
            )


if __name__ == '__main__':
    average_duplicates(Args().parse_args())
