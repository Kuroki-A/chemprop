"""Computes the overlap of molecules between two datasets."""

import csv
import os
from typing import List

from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

class Args(Tap):
    data_path_1: str  # Path to first data CSV file
    data_path_2: str  # Path to second data CSV file
    smiles_columns_1: List[str] = None  # Name of the column containing SMILES strings for the first data. By default, uses the first column.
    smiles_columns_2: List[str] = None  # Name of the column containing SMILES strings for the second data. By default, uses the first column.
    save_intersection_path: str = None  # Path to save intersection at; labeled with data_path 1 header
    save_difference_path: str = None  # Path to save molecules in dataset 1 that are not in dataset 2; labeled with data_path 1 header


def _read_rows_and_smiles(path: str, smiles_columns: List[str]):
    with open(path, newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f'Dataset {path!r} is empty.') from exc
        columns = [header[0]] if smiles_columns is None else list(smiles_columns)
        if not columns or len(columns) != len(set(columns)):
            raise ValueError('SMILES columns must be a non-empty list without duplicates.')
        missing = [column for column in columns if column not in header]
        if missing:
            raise ValueError(
                f'Dataset {path!r} is missing SMILES columns {missing!r}.'
            )
        indices = [header.index(column) for column in columns]
        rows = list(reader)
    smiles = []
    for row_number, row in enumerate(rows, start=2):
        if any(index >= len(row) for index in indices):
            raise ValueError(
                f'Dataset {path!r} row {row_number} has fewer columns than its header.'
            )
        smiles.append(tuple(row[index] for index in indices))
    return header, rows, smiles


def _write_selected_rows(path: str, header, rows, keys, selected_keys) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(
            row for row, key in zip(rows, keys) if key in selected_keys
        )


def overlap(args: Args):
    header_1, rows_1, smiles_1_rows = _read_rows_and_smiles(
        args.data_path_1, args.smiles_columns_1
    )
    _, _, smiles_2_rows = _read_rows_and_smiles(
        args.data_path_2, args.smiles_columns_2
    )
    if smiles_1_rows and smiles_2_rows and \
            len(smiles_1_rows[0]) != len(smiles_2_rows[0]):
        raise ValueError('Both datasets must specify the same number of SMILES columns.')

    smiles_1, smiles_2 = set(smiles_1_rows), set(smiles_2_rows)
    size_1, size_2 = len(smiles_1), len(smiles_2)
    intersection = smiles_1.intersection(smiles_2)
    size_intersect = len(intersection)
    print(f'Size of dataset 1: {size_1}')
    print(f'Size of dataset 2: {size_2}')
    print(f'Size of intersection: {size_intersect}')
    print(f'Size of intersection as frac of dataset 1: {size_intersect / size_1 if size_1 else float("nan")}')
    print(f'Size of intersection as frac of dataset 2: {size_intersect / size_2 if size_2 else float("nan")}')

    if args.save_intersection_path is not None:
        _write_selected_rows(
            args.save_intersection_path,
            header_1,
            rows_1,
            smiles_1_rows,
            intersection,
        )

    if args.save_difference_path is not None:
        _write_selected_rows(
            args.save_difference_path,
            header_1,
            rows_1,
            smiles_1_rows,
            smiles_1 - smiles_2,
        )


if __name__ == '__main__':
    overlap(Args().parse_args())
