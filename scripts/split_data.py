"""Splits data into train, validation, and test sets."""

import csv
import os
import sys
from typing import Tuple, List

from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)
from tqdm import tqdm
from typing_extensions import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import (
    is_valid_datapoint,
    MoleculeDatapoint,
    MoleculeDataset,
    preprocess_smiles_columns,
    split_data,
)
from chemprop.utils import makedirs


class Args(Tap):
    data_path: str  # Path to data CSV file
    save_dir: str  # Directory where train, validation, and test sets will be saved
    smiles_columns: List[str] = None  # Name of the column containing SMILES strings. By default, uses the first column.
    split_type: Literal['random', 'scaffold_balanced'] = 'random'  # Split type
    split_sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # Split sizes
    seed: int = 0  # Random seed


def run_split_data(args: Args):
    # Load raw data
    with open(args.data_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError('Input CSV is empty.') from exc
        lines = list(reader)

    smiles_columns = preprocess_smiles_columns(
        path=args.data_path,
        smiles_columns=args.smiles_columns,
        number_of_molecules=1,
    )
    smiles_indices = [header.index(column) for column in smiles_columns]

    # Create data
    data = []
    for row_number, line in enumerate(tqdm(lines), start=2):
        if any(index >= len(line) for index in smiles_indices):
            raise ValueError(
                f'CSV row {row_number} has fewer columns than the header.'
            )
        smile = [line[index] for index in smiles_indices]
        datapoint = MoleculeDatapoint(smiles=smile)
        if not is_valid_datapoint(datapoint):
            raise ValueError(
                f'CSV row {row_number} contains an empty or invalid SMILES.'
            )
        datapoint.line = line
        data.append(datapoint)
    data = MoleculeDataset(data)

    train, val, test = split_data(
        data=data,
        split_type=args.split_type,
        sizes=args.split_sizes,
        seed=args.seed
    )

    makedirs(args.save_dir)

    for name, dataset in [('train', train), ('val', val), ('test', test)]:
        with open(os.path.join(args.save_dir, f'{name}.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for datapoint in dataset:
                writer.writerow(datapoint.line)


if __name__ == '__main__':
    run_split_data(Args().parse_args())
