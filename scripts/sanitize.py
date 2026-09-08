"""Cleans a dataset by removing molecules which cannot be parsed by RDKit."""

import csv
import os
from typing import List

from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

from chemprop.data import is_valid_datapoint, MoleculeDatapoint, preprocess_smiles_columns


class Args(Tap):
    data_path: str  # Data CSV to sanitize
    save_path: str  # Path to CSV where sanitized data will be saved
    smiles_columns: List[str] = None  # SMILES columns; defaults to the first column


def sanitize(data_path: str, save_path: str, smiles_columns: List[str] = None):
    with open(data_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError('Input CSV is empty.') from exc
        smiles_columns = preprocess_smiles_columns(
            path=data_path,
            smiles_columns=smiles_columns,
            number_of_molecules=1,
        )
        smiles_indices = [header.index(column) for column in smiles_columns]
        lines = []
        for line in reader:
            if any(index >= len(line) for index in smiles_indices):
                continue
            datapoint = MoleculeDatapoint(
                smiles=[line[index] for index in smiles_indices]
            )
            if is_valid_datapoint(datapoint):
                lines.append(line)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for line in lines:
            writer.writerow(line)


if __name__ == '__main__':
    args = Args().parse_args()

    sanitize(args.data_path, args.save_path, args.smiles_columns)
