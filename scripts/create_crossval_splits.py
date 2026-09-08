from copy import deepcopy
import os
import pickle
import random
from typing import List
from typing_extensions import Literal


import numpy as np
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

from chemprop.data import MoleculeDataset
from chemprop.data import get_data, scaffold_to_smiles


class Args(Tap):
    data_path: str  # Path to CSV file with dataset of molecules
    save_dir: str  # Path to CSV file where splits will be saved
    split_type: Literal['random', 'scaffold', 'time_window']  # Random or scaffold based split
    num_folds: int = 10  # Number of cross validation folds
    test_folds_to_test: int = 3  # Number of test folds
    val_folds_per_test: int = 3  # Number of val folds
    time_folds_per_train_set: int = 3  # X:1:1 train:val:test for time split sliding window
    smiles_columns: List[str] = None # columns in CSV dataset file containing SMILES
    split_key_molecule: int = 0 # index of the molecule to use for splitting in muli-molecule data
    seed: int = 0  # Random seed used for all fold shuffling


def split_indices(all_indices: List[int],
                  num_folds: int,
                  scaffold: bool = False,
                  split_key_molecule: int = 0,
                  data: MoleculeDataset = None,
                  shuffle: bool = True,
                  rng: random.Random = None) -> List[List[int]]:
    if not isinstance(num_folds, int) or isinstance(num_folds, bool) or num_folds <= 0:
        raise ValueError('num_folds must be a positive integer.')
    if len(all_indices) < num_folds:
        raise ValueError(
            f'Cannot create {num_folds} non-empty folds from {len(all_indices)} rows.'
        )
    rng = random.Random(0) if rng is None else rng
    all_indices = list(all_indices)
    num_data = len(all_indices)
    if scaffold:
        if data is None or len(data) != num_data:
            raise ValueError(
                'Scaffold splitting requires data aligned one-to-one with all_indices.'
            )
        if not isinstance(split_key_molecule, int) or isinstance(split_key_molecule, bool) \
                or split_key_molecule < 0 or split_key_molecule >= data.number_of_molecules:
            raise ValueError('split_key_molecule is out of range for this dataset.')
        key_mols = [m[split_key_molecule] for m in data.mols(flatten=False)]
        scaffold_to_indices = scaffold_to_smiles(key_mols, use_indices=True)
        # scaffold_to_smiles returns indices local to ``data``. Translate them
        # back to the caller's index space (important for sliding time windows).
        index_sets = [
            [all_indices[local_index] for local_index in sorted(index_set)]
            for index_set in scaffold_to_indices.values()
        ]
        index_sets.sort(key=lambda index_set: (-len(index_set), index_set[0]))
        fold_indices = [[] for _ in range(num_folds)]
        for s in index_sets:
            length_array = [len(fi) for fi in fold_indices]
            min_index = length_array.index(min(length_array))
            fold_indices[min_index] += s
        if shuffle:
            rng.shuffle(fold_indices)
    else:  # random
        if shuffle:
            rng.shuffle(all_indices)
        fold_indices = []
        for i in range(num_folds):
            begin, end = int(i * num_data / num_folds), int((i + 1) * num_data / num_folds)
            fold_indices.append(np.array(all_indices[begin:end]))
    if any(len(indices) == 0 for indices in fold_indices):
        raise ValueError(
            f'Unable to create {num_folds} non-empty folds. For scaffold '
            'splitting, reduce num_folds below the number of unique scaffolds.'
        )
    return fold_indices


def create_time_splits(args: Args):
    # ASSUME DATA GIVEN IN CHRONOLOGICAL ORDER.
    # this will dump a very different format of indices, with all in one file; TODO modify as convenient later.
    data = get_data(path=args.data_path, smiles_columns=args.smiles_columns)
    num_data = len(data)
    if args.time_folds_per_train_set <= 0:
        raise ValueError('time_folds_per_train_set must be a positive integer.')
    if args.num_folds <= args.time_folds_per_train_set + 1:
        raise ValueError(
            'num_folds must leave at least one train, validation, and test window.'
        )
    rng = random.Random(args.seed)
    all_indices = list(range(num_data))
    fold_indices = {'random': [], 'scaffold': [], 'time': []}
    for i in range(args.num_folds - args.time_folds_per_train_set - 1):
        begin, end = int(i * num_data / args.num_folds), int(
            (i + args.time_folds_per_train_set + 2) * num_data / args.num_folds)
        subset_indices = all_indices[begin:end]
        subset_data = MoleculeDataset(data[begin:end])
        fold_indices['random'].append(split_indices(
            deepcopy(subset_indices), args.time_folds_per_train_set + 2, rng=rng
        ))
        fold_indices['scaffold'].append(
            split_indices(
                subset_indices,
                args.time_folds_per_train_set + 2,
                scaffold=True,
                split_key_molecule=args.split_key_molecule,
                data=subset_data,
                rng=rng,
            )
        )
        fold_indices['time'].append(split_indices(
            subset_indices, args.time_folds_per_train_set + 2, shuffle=False, rng=rng
        ))
    for split_type in ['random', 'scaffold', 'time']:
        for i in range(len(fold_indices[split_type])):
            fold_dir = os.path.join(args.save_dir, split_type, f'fold_{i}')
            os.makedirs(os.path.join(fold_dir, '0'), exist_ok=True)
            train = np.concatenate([
                fold_indices[split_type][i][j]
                for j in range(args.time_folds_per_train_set)
            ])
            val = fold_indices[split_type][i][-2]
            test = fold_indices[split_type][i][-1]
            split = [train, val, test]
            with open(os.path.join(fold_dir, '0', 'split_indices.pckl'), 'wb') as wf:
                pickle.dump(split, wf)
            # Match the predetermined split schema used elsewhere: the parent
            # file contains a list of available validation splits.
            with open(os.path.join(fold_dir, 'split_indices.pckl'), 'wb') as wf:
                pickle.dump([split], wf)


def create_crossval_splits(args: Args):
    data = get_data(path=args.data_path, smiles_columns=args.smiles_columns)
    num_data = len(data)
    if args.num_folds < 3:
        raise ValueError('num_folds must be at least 3 for train/val/test splitting.')
    if not 1 <= args.test_folds_to_test <= args.num_folds:
        raise ValueError('test_folds_to_test must be between 1 and num_folds.')
    if not 1 <= args.val_folds_per_test <= args.num_folds - 1:
        raise ValueError(
            'val_folds_per_test must be between 1 and num_folds - 1.'
        )
    rng = random.Random(args.seed)
    if args.split_type == 'random':
        all_indices = list(range(num_data))
        fold_indices = split_indices(
            all_indices, args.num_folds, scaffold=False, rng=rng
        )
    elif args.split_type == 'scaffold':
        all_indices = list(range(num_data))
        fold_indices = split_indices(
            all_indices,
            args.num_folds,
            scaffold=True,
            split_key_molecule=args.split_key_molecule,
            data=data,
            rng=rng,
        )
    else:
        raise ValueError(f'Unsupported split_type {args.split_type!r}.')
    rng.shuffle(fold_indices)
    for i in range(args.test_folds_to_test):
        all_splits = []
        for j in range(1, args.val_folds_per_test + 1):
            os.makedirs(os.path.join(args.save_dir, args.split_type, f'fold_{i}', f'{j - 1}'), exist_ok=True)
            with open(os.path.join(args.save_dir, args.split_type, f'fold_{i}', f'{j - 1}', 'split_indices.pckl'),
                      'wb') as wf:
                val_idx = (i + j) % args.num_folds
                val = fold_indices[val_idx]
                test = fold_indices[i]
                train = []
                for k in range(args.num_folds):
                    if k != i and k != val_idx:
                        train.append(fold_indices[k])
                train = np.concatenate(train)
                pickle.dump([train, val, test], wf)
                all_splits.append([train, val, test])
        with open(os.path.join(args.save_dir, args.split_type, f'fold_{i}', 'split_indices.pckl'), 'wb') as wf:
            pickle.dump(all_splits, wf)


if __name__ == '__main__':
    args = Args().parse_args()

    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.data_path)
        if args.split_type == 'time_window':
            args.save_dir = os.path.join(args.save_dir, 'time_window')

    if args.split_type == 'time_window':
        create_time_splits(args)
    else:
        create_crossval_splits(args)
