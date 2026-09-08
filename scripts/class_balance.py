import os
import sys
from typing import List
from typing_extensions import Literal


import numpy as np
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import get_class_sizes, get_data, get_task_names, split_data


class Args(Tap):
    data_path: str  # Path to data CSV file
    smiles_columns: List[str] = None  # Name of the columns containing SMILES strings. By default, uses the first column.
    target_columns: List[str] = None  # Name of the columns containing target values. By default, uses all columns except the SMILES column.
    split_type: Literal['random', 'scaffold'] = 'scaffold'  # Split type, either "random" or "scaffold"
    splits_dir: str = '/data/rsg/chemistry/yangk/lsc_experiments_dump_splits/data'  # Root containing dataset/split_type/fold_* directories
    num_folds: int = 10  # Number of folds to inspect


def class_balance(args: Args):
    requested_split_type = args.split_type
    if args.num_folds <= 0:
        raise ValueError('num_folds must be positive.')
    # Update args
    args.val_fold_index, args.test_fold_index = 1, 2
    args.split_type = 'predetermined'

    # Load data
    data = get_data(path=args.data_path, smiles_columns=args.smiles_columns, target_columns=args.target_columns)
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns, target_columns=args.target_columns)

    # Average class sizes
    all_class_sizes = {
        'train': [],
        'val': [],
        'test': []
    }

    for fold_index in range(args.num_folds):
        print(f'Fold {fold_index}')

        # Update args
        data_name = os.path.splitext(os.path.basename(args.data_path))[0]
        args.folds_file = os.path.join(
            args.splits_dir,
            data_name,
            requested_split_type,
            f'fold_{fold_index}',
            '0',
            'split_indices.pckl',
        )

        if not os.path.exists(args.folds_file):
            print('Fold indices do not exist')
            continue

        # Split data
        train_data, val_data, test_data = split_data(
            data=data,
            split_type=args.split_type,
            args=args
        )

        # Determine class balance
        for data_split, split_name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            class_sizes = get_class_sizes(data_split)
            print(f'Class sizes for {split_name}')

            for task_index, task_class_sizes in enumerate(class_sizes):
                print(f'{args.task_names[task_index]} '
                      f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

            all_class_sizes[split_name].append(class_sizes)

        print()

    # Mean and std across folds
    for split_name in ['train', 'val', 'test']:
        print(f'Average class sizes for {split_name}')

        if not all_class_sizes[split_name]:
            print('No fold data found')
            continue

        mean_class_sizes, std_class_sizes = np.mean(all_class_sizes[split_name], axis=0), np.std(all_class_sizes[split_name], axis=0)

        for task_index, (mean_task_class_sizes, std_task_class_sizes) in enumerate(zip(mean_class_sizes, std_class_sizes)):
            print(f'{args.task_names[task_index]} '
                  f'{", ".join(f"{cls}: {mean_size * 100:.2f}% +/- {std_size * 100:.2f}%" for cls, (mean_size, std_size) in enumerate(zip(mean_task_class_sizes, std_task_class_sizes)))}')


if __name__ == '__main__':
    class_balance(args=Args().parse_args())
