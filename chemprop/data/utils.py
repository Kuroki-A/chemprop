from __future__ import annotations

from collections import OrderedDict, defaultdict
import hashlib
import sys
import csv
import ctypes
from logging import Logger
import pickle
from random import Random
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union
import os
import json

from rdkit import Chem
import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import MoleculeDatapoint, MoleculeDataset, generate_features_for_smiles_batch, \
    load_selected_feature_columns, make_mols
from .scaffold import log_scaffold_stats, scaffold_split
from chemprop.features import get_features_generator_schema, is_mol, load_features, \
    load_valid_atom_or_bond_features
from chemprop.rdkit import make_mol

if TYPE_CHECKING:
    from chemprop.args import PredictArgs, TrainArgs

# Increase maximum size of field in the csv processing for the current architecture
csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.
    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


_FEATURE_MANIFEST_SCHEMA_FIELDS = (
    'schema_version',
    'generator',
    'generator_config',
    'versions',
    'feature_names',
    'dimension',
    'dtype',
    'implementation_sha256',
)


def _load_feature_manifest(path: str) -> Optional[dict]:
    """Loads and minimally validates a save_features sidecar manifest."""
    manifest_path = f'{path}.manifest.json'
    if not os.path.isfile(manifest_path):
        return None

    try:
        with open(manifest_path, encoding='utf-8') as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f'Invalid feature manifest {manifest_path}: {error}'
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError(
            f'Invalid feature manifest {manifest_path}: expected a JSON object.'
        )

    return manifest


def _manifest_dimension(path: str, manifest: dict) -> int:
    """Returns a validated feature width from a sidecar manifest."""
    manifest_path = f'{path}.manifest.json'

    recorded_dimension = manifest.get('dimension')
    if (
        isinstance(recorded_dimension, bool)
        or not isinstance(recorded_dimension, int)
        or recorded_dimension < 0
    ):
        raise ValueError(
            f'Invalid feature manifest {manifest_path}: dimension must be a '
            'non-negative integer.'
        )
    return recorded_dimension


def _normalize_feature_matrix(
    path: str,
    values: np.ndarray,
    manifest: Optional[dict],
) -> np.ndarray:
    """Normalizes the special empty save_features archive to a 2-D matrix."""
    array = np.asarray(values)
    if array.ndim == 2:
        return array
    if array.ndim == 1 and array.size == 0:
        manifest_path = f'{path}.manifest.json'
        if manifest is None:
            raise ValueError(
                f'Feature file {path} is an empty 1-D array. A complete '
                'save_features manifest is required to recover its width.'
            )
        if manifest.get('status') != 'complete':
            raise ValueError(
                f'Invalid feature manifest {manifest_path}: status must be '
                "'complete' to load an empty feature array."
            )
        dimension = _manifest_dimension(path, manifest)
        return array.reshape((0, dimension))
    raise ValueError(
        f'Feature file {path} must contain a 2-D matrix, got shape {array.shape}.'
    )


def _ordered_smiles_sha256(smiles: List[str]) -> str:
    """Hashes ordered SMILES using the save_features length-prefixed format."""
    digest = hashlib.sha256(b'chemprop-ordered-smiles-v1\0')
    for value in smiles:
        encoded = str(value).encode('utf-8')
        digest.update(len(encoded).to_bytes(8, byteorder='big', signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _manifest_smiles_column_indices(
    path: str,
    manifest: dict,
    ordered_smiles_columns: List[List[str]],
) -> Optional[List[int]]:
    """Validates manifest row identity and returns matching SMILES columns."""
    manifest_path = f'{path}.manifest.json'
    input_identity = manifest.get('input')
    if input_identity is None:
        return None
    if not isinstance(input_identity, dict):
        raise ValueError(
            f'Invalid feature manifest {manifest_path}: input must be a JSON object.'
        )

    has_hash = 'ordered_smiles_sha256' in input_identity
    has_count = 'num_smiles' in input_identity
    if not has_hash and not has_count:
        return None
    if not has_hash or not has_count:
        raise ValueError(
            f'Invalid feature manifest {manifest_path}: input must contain both '
            'ordered_smiles_sha256 and num_smiles.'
        )

    encoding = input_identity.get(
        'ordered_smiles_encoding', 'chemprop-length-prefixed-utf8-v1',
    )
    if encoding != 'chemprop-length-prefixed-utf8-v1':
        raise ValueError(
            f'Invalid feature manifest {manifest_path}: unsupported '
            f'ordered_smiles_encoding {encoding!r}.'
        )

    recorded_count = input_identity['num_smiles']
    if (
        isinstance(recorded_count, bool)
        or not isinstance(recorded_count, int)
        or recorded_count < 0
    ):
        raise ValueError(
            f'Invalid feature manifest {manifest_path}: input.num_smiles must '
            'be a non-negative integer.'
        )
    expected_count = (
        len(ordered_smiles_columns[0]) if ordered_smiles_columns else 0
    )
    if recorded_count != expected_count:
        raise ValueError(
            f'Feature manifest {manifest_path} records input.num_smiles '
            f'{recorded_count}, but the current data has {expected_count} rows.'
        )

    recorded_hash = input_identity['ordered_smiles_sha256']
    if not isinstance(recorded_hash, str):
        raise ValueError(
            f'Invalid feature manifest {manifest_path}: '
            'input.ordered_smiles_sha256 must be a string.'
        )
    matching_indices = [
        index
        for index, smiles in enumerate(ordered_smiles_columns)
        if _ordered_smiles_sha256(smiles) == recorded_hash
    ]
    if not matching_indices:
        raise ValueError(
            f'Feature manifest {manifest_path} input.ordered_smiles_sha256 '
            'does not match any configured SMILES column in the current data; '
            'the SMILES values or row order changed.'
        )
    return matching_indices


def _feature_manifest_schema(
    path: str,
    array: np.ndarray,
    manifest: Optional[dict] = None,
) -> Optional[dict]:
    """Loads row-independent semantic fields from a save_features manifest."""
    if manifest is None:
        manifest = _load_feature_manifest(path)
    if manifest is None:
        return None
    manifest_path = f'{path}.manifest.json'

    recorded_dimension = _manifest_dimension(path, manifest)
    actual_dimension = int(array.shape[1])
    if recorded_dimension != actual_dimension:
        raise ValueError(
            f'Feature manifest {manifest_path} records dimension '
            f'{recorded_dimension}, but the feature array has width '
            f'{actual_dimension}.'
        )

    recorded_dtype = manifest.get('dtype')
    if recorded_dtype is not None:
        try:
            dtype_matches = np.dtype(recorded_dtype) == array.dtype
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'Invalid feature manifest {manifest_path}: unsupported dtype '
                f'{recorded_dtype!r}.'
            ) from error
        if not dtype_matches:
            raise ValueError(
                f'Feature manifest {manifest_path} records dtype '
                f'{recorded_dtype!r}, but the feature array has dtype '
                f'{array.dtype!s}.'
            )

    feature_names = manifest.get('feature_names')
    if feature_names is not None and (
        not isinstance(feature_names, list)
        or len(feature_names) != actual_dimension
    ):
        raise ValueError(
            f'Invalid feature manifest {manifest_path}: feature_names must '
            f'contain exactly {actual_dimension} entries.'
        )

    return {
        field: manifest[field]
        for field in _FEATURE_MANIFEST_SCHEMA_FIELDS
        if field in manifest
    }


def _feature_source_entry(
    path: str,
    values: np.ndarray,
    manifest: Optional[dict] = None,
    ordered_smiles_columns: Optional[List[List[str]]] = None,
) -> dict:
    """Describes one ordered external feature matrix without recording row data."""
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(
            f'Feature file {path} must contain a 2-D matrix, got shape {array.shape}.'
        )

    extension = os.path.splitext(path)[1].lower()
    # ``load_features`` may be supplied by an embedding application or mocked in
    # tests without a corresponding on-disk CSV.  The matrix shape remains the
    # authoritative schema in that case; record a header only when it can be
    # inspected safely.
    csv_header = (
        get_header(path)
        if extension in {'.csv', '.txt'} and os.path.isfile(path)
        else None
    )
    if csv_header is not None and len(csv_header) != array.shape[1]:
        raise ValueError(
            f'Feature file {path} has {len(csv_header)} header columns but '
            f'{array.shape[1]} feature columns.'
        )

    entry = {
        'dimension': int(array.shape[1]),
        'dtype': str(array.dtype),
        'csv_header': csv_header,
    }
    feature_manifest = _feature_manifest_schema(path, array, manifest=manifest)
    if feature_manifest is not None:
        entry['feature_manifest'] = feature_manifest
        if ordered_smiles_columns is not None:
            expected_rows = (
                len(ordered_smiles_columns[0])
                if ordered_smiles_columns
                else 0
            )
            if array.shape[0] != expected_rows:
                raise ValueError(
                    f'Feature file {path} has {array.shape[0]} rows, but its '
                    f'manifest is being loaded with {expected_rows} data rows.'
                )
            smiles_column_indices = _manifest_smiles_column_indices(
                path, manifest, ordered_smiles_columns,
            )
            if smiles_column_indices is not None:
                entry['smiles_column_indices'] = smiles_column_indices

    return entry


def _features_source_metadata(
    external_features: List[dict],
    phase_features: Optional[dict],
    generated_dimension: Optional[int],
) -> dict:
    """Builds the ordered, row-independent molecular feature source schema."""
    external_dimension = sum(source['dimension'] for source in external_features)
    phase_dimension = 0 if phase_features is None else phase_features['dimension']
    return {
        'schema_version': 1,
        'external_features': external_features,
        'phase_features': phase_features,
        'generated_dimension': (
            None if generated_dimension is None else int(generated_dimension)
        ),
        'total_dimension': (
            None
            if generated_dimension is None
            else int(external_dimension + phase_dimension + generated_dimension)
        ),
    }


def _static_generated_dimension(
    features_generators: List[str],
    selected_feature_columns: dict,
    number_of_molecules: int,
) -> Optional[int]:
    """Returns a generator width without evaluating molecules when it is known."""
    if not features_generators:
        return 0

    dimension = 0
    for generator_name in features_generators:
        try:
            generator_schema = get_features_generator_schema(
                generator_name,
                selected_feature_columns=selected_feature_columns.get(generator_name),
            )
        except ImportError:
            return None
        generator_dimension = generator_schema.get('dimension')
        if generator_dimension is None:
            return None
        dimension += int(generator_dimension)

    return dimension * number_of_molecules


def preprocess_smiles_columns(path: str,
                              smiles_columns: Union[str, List[str]] = None,
                              number_of_molecules: int = 1) -> List[str]:
    """
    Preprocesses the :code:`smiles_columns` variable to ensure that it is a list of column
    headings corresponding to the columns in the data file holding SMILES. Assumes file has a header.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param number_of_molecules: The number of molecules with associated SMILES for each
                           data point.
    :return: The preprocessed version of :code:`smiles_columns` which is guaranteed to be a list.
    """

    if smiles_columns is None:
        if os.path.isfile(path):
            columns = get_header(path)
            smiles_columns = columns[:number_of_molecules]
        else:
            smiles_columns = [None]*number_of_molecules
    else:
        if isinstance(smiles_columns, str):
            smiles_columns = [smiles_columns]
        if os.path.isfile(path):
            columns = get_header(path)
            if len(smiles_columns) != number_of_molecules:
                raise ValueError('Length of smiles_columns must match number_of_molecules.')
            if any([smiles not in columns for smiles in smiles_columns]):
                raise ValueError('Provided smiles_columns do not match the header of data file.')

    return smiles_columns


def _expand_quantile_task_names(
    target_names: List[str], loss_function: str = None,
) -> List[str]:
    """Returns lower/upper output names without duplicating an expanded list."""
    names = list(target_names)
    if loss_function != 'quantile_interval':
        return names
    midpoint = len(names) // 2
    already_expanded = (
        len(names) > 0
        and len(names) % 2 == 0
        and names[:midpoint] == names[midpoint:]
    )
    return names if already_expanded else names * 2


def get_task_names(
    path: str,
    smiles_columns: Union[str, List[str]] = None,
    target_columns: List[str] = None,
    ignore_columns: List[str] = None,
    loss_function: str = None,
) -> List[str]:
    """
    Gets the task names from a data CSV file.
    If :code:`target_columns` is provided, returns `target_columns`.
    Otherwise, returns all columns except the :code:`smiles_columns`
    (or the first column, if the :code:`smiles_columns` is None) and
    the :code:`ignore_columns`.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_columns` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :return: A list of task names.
    """
    if target_columns is not None:
        return _expand_quantile_task_names(target_columns, loss_function)

    columns = get_header(path)

    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    ignore_columns = set(smiles_columns + ([] if ignore_columns is None else ignore_columns))

    target_names = [column for column in columns if column not in ignore_columns]

    return _expand_quantile_task_names(target_names, loss_function)


def get_mixed_task_names(path: str,
                         smiles_columns: Union[str, List[str]] = None,
                         target_columns: List[str] = None,
                         ignore_columns: List[str] = None,
                         keep_h: bool = None,
                         add_h: bool = None,
                         keep_atom_map: bool = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Gets the task names for atomic, bond, and molecule targets separately from a data CSV file.

    If :code:`target_columns` is provided, returned lists based off `target_columns`.
    Otherwise, returned lists based off all columns except the :code:`smiles_columns`
    (or the first column, if the :code:`smiles_columns` is None) and
    the :code:`ignore_columns`.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_columns` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h: Boolean whether to add hydrogens to the input smiles.
    :param keep_atom_map: Boolean whether to keep the original atom mapping.
    :return: A tuple containing the task names of atomic, bond, and molecule properties separately.
    """
    columns = get_header(path)

    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    ignore_columns = set(smiles_columns + ([] if ignore_columns is None else ignore_columns))

    if target_columns is not None:
        target_names =  target_columns
    else:
        target_names = [column for column in columns if column not in ignore_columns]

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            atom_target_names, bond_target_names, molecule_target_names = [], [], []
            smiles = [row[c] for c in smiles_columns]
            for s in smiles:
                if keep_atom_map:
                    # When the original atom mapping is used, the explicit hydrogens specified in the input SMILES should be used
                    # However, the explicit Hs can only be added for reactions with `--explicit_h` flag
                    # To fix this, `keep_h` is set to True when `keep_atom_map` is also True
                    mol = make_mol(s, keep_h=True, add_h=add_h, keep_atom_map=True)
                else:
                    mol = make_mol(s, keep_h=keep_h, add_h=add_h, keep_atom_map=False)
                if len(mol.GetAtoms()) != len(mol.GetBonds()):
                    break

            for column in target_names:
                value = row[column]
                value = value.replace('None', 'null')
                target = np.array(json.loads(value))

                is_atom_target, is_bond_target, is_molecule_target = False, False, False
                if len(target.shape) == 0:
                    is_molecule_target = True
                elif len(target.shape) == 1:
                    if len(target) == len(mol.GetAtoms()):  # Atom targets saved as 1D list
                        is_atom_target = True
                    elif len(target) == len(mol.GetBonds()):  # Bond targets saved as 1D list
                        is_bond_target = True
                    else:
                        raise RuntimeError(f'Unrecognized targets of column {column} in {path}. '
                                           'Expected targets should be either atomic or bond targets. '
                                           'Please ensure the content is correct.')
                elif len(target.shape) == 2:  # Bond targets saved as 2D list
                    is_bond_target = True
                else:
                    raise ValueError(f'Unrecognized targets of column {column} in {path}.')
                
                if is_atom_target:
                    atom_target_names.append(column)
                elif is_bond_target:
                    bond_target_names.append(column)
                elif is_molecule_target:
                    molecule_target_names.append(column)
            if len(atom_target_names) + len(bond_target_names) + len(molecule_target_names) == len(target_names):
                break

    return atom_target_names, bond_target_names, molecule_target_names


def get_data_weights(path: str) -> List[float]:
    """
    Returns the list of data weights for the loss function as stored in a CSV file.

    :param path: Path to a CSV file.
    :return: A list of floats containing the data weights.
    """
    weights = []
    with open(path) as f:
        reader = csv.reader(f)
        try:
            next(reader)  # skip header row
        except StopIteration as error:
            raise ValueError('Data weights file must contain a header row.') from error
        for line in reader:
            if len(line) != 1 or line[0] == '':
                raise ValueError('Each data weights row must contain exactly one value.')
            weights.append(float(line[0]))
    # normalize the data weights
    if not weights:
        raise ValueError('At least one data weight must be provided.')
    weights_array = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weights_array)):
        raise ValueError('Data weights must be finite for each datapoint.')
    if np.any(weights_array < 0):
        raise ValueError('Data weights must be non-negative for each datapoint.')
    if float(weights_array.sum()) <= 0:
        raise ValueError('At least one data weight must be positive.')
    weights_array /= float(weights_array.mean())
    return weights_array.tolist()


def get_constraints(path: str,
                    target_columns: List[str],
                    save_raw_data: bool = False) -> Tuple[List[float], List[float]]:
    """
    Returns lists of data constraints for the atomic/bond targets as stored in a CSV file.

    :param path: Path to a CSV file.
    :param target_columns: Name of the columns containing target values.
    :param save_raw_data: Whether to save all user-provided atom/bond-level constraints in input data,
                          which will be used to construct constraints files for each train/val/test split
                          for prediction convenience later.
    :return: Lists of floats containing the data constraints.
    """
    constraints_data = []
    reader = pd.read_csv(path)
    reader_columns = reader.columns.tolist()
    if len(reader_columns) != len(set(reader_columns)):
        raise ValueError(f'There are duplicates in {path}.')
    for target in target_columns:
        if target in reader_columns:
            constraints_data.append(reader[target].values)
        else:
            constraints_data.append([None] * len(reader))
    constraints_data = np.transpose(constraints_data)  # each is num_data x num_targets

    if save_raw_data:
        raw_constraints_data = []
        for target in reader_columns:
            raw_constraints_data.append(reader[target].values)
        raw_constraints_data = np.transpose(raw_constraints_data)  # each is num_data x num_columns
    else:
        raw_constraints_data = None
    
    return constraints_data, raw_constraints_data


def get_smiles(path: str,
               smiles_columns: Union[str, List[str]] = None,
               number_of_molecules: int = 1,
               header: bool = True,
               flatten: bool = False
               ) -> Union[List[str], List[List[str]]]:
    """
    Returns the SMILES from a data CSV file.

    :param path: Path to a CSV file.
    :param smiles_columns: A list of the names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param number_of_molecules: The number of molecules for each data point. Not necessary if
                                the names of smiles columns are previously processed.
    :param header: Whether the CSV file contains a header.
    :param flatten: Whether to flatten the returned SMILES to a list instead of a list of lists.
    :return: A list of SMILES or a list of lists of SMILES, depending on :code:`flatten`.
    """
    if smiles_columns is not None and not header:
        raise ValueError('If smiles_column is provided, the CSV file must have a header.')

    if (isinstance(smiles_columns, str) or smiles_columns is None) and header:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns, number_of_molecules=number_of_molecules)

    with open(path) as f:
        if header:
            reader = csv.DictReader(f)
        else:
            reader = csv.reader(f)
            smiles_columns = list(range(number_of_molecules))

        smiles = [[row[c] for c in smiles_columns] for row in reader]

    if flatten:
        smiles = [smile for smiles_list in smiles for smile in smiles_list]

    return smiles


def is_valid_molecule(mol) -> bool:
    """Returns whether a parsed molecule or reaction has usable heavy atoms."""
    if isinstance(mol, tuple):
        return (
            len(mol) == 2
            and all(component is not None for component in mol)
            and sum(component.GetNumHeavyAtoms() for component in mol) > 0
        )
    return mol is not None and mol.GetNumHeavyAtoms() > 0


def is_valid_datapoint(datapoint: MoleculeDatapoint) -> bool:
    """Returns whether every SMILES entry in a datapoint parses successfully."""
    if any(smiles == '' for smiles in datapoint.smiles):
        return False
    return all(is_valid_molecule(mol) for mol in datapoint.mol)


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :return: A :class:`~chemprop.data.MoleculeDataset` with only the valid molecules.
    """
    return MoleculeDataset([
        datapoint for datapoint in tqdm(data) if is_valid_datapoint(datapoint)
    ])


def get_invalid_smiles_from_file(path: str = None,
                                 smiles_columns: Union[str, List[str]] = None,
                                 header: bool = True,
                                 reaction: bool = False,
                                 ) -> Union[List[str], List[List[str]]]:
    """
    Returns the invalid SMILES from a data CSV file.

    :param path: Path to a CSV file.
    :param smiles_columns: A list of the names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param header: Whether the CSV file contains a header.
    :param reaction: Boolean whether the SMILES strings are to be treated as a reaction.
    :return: A list of lists of SMILES, for the invalid SMILES in the file.
    """
    smiles = get_smiles(path=path, smiles_columns=smiles_columns, header=header)

    invalid_smiles = get_invalid_smiles_from_list(smiles=smiles, reaction=reaction)

    return invalid_smiles


def get_invalid_smiles_from_list(smiles: List[List[str]], reaction: bool = False) -> List[List[str]]:
    """
    Returns the invalid SMILES from a list of lists of SMILES strings.

    :param smiles: A list of list of SMILES.
    :param reaction: Boolean whether the SMILES strings are to be treated as a reaction.
    :return: A list of lists of SMILES, for the invalid SMILES among the lists provided.
    """
    invalid_smiles = []

    # If the first SMILES in the column is a molecule, the remaining SMILES in the same column should all be a molecule.
    # Similarly, if the first SMILES in the column is a reaction, the remaining SMILES in the same column should all
    # correspond to reaction. Therefore, get `is_mol_list` only using the first element in smiles.
    is_mol_list = [is_mol(s) for s in smiles[0]]
    is_reaction_list = [True if not x and reaction else False for x in is_mol_list]
    is_explicit_h_list = [False for x in is_mol_list]  # set this to False as it is not needed for invalid SMILES check
    is_adding_hs_list = [False for x in is_mol_list]  # set this to False as it is not needed for invalid SMILES check
    keep_atom_map_list = [False for x in is_mol_list]  # set this to False as it is not needed for invalid SMILES check

    for mol_smiles in smiles:
        mols = make_mols(smiles=mol_smiles, reaction_list=is_reaction_list, keep_h_list=is_explicit_h_list,
                         add_h_list=is_adding_hs_list, keep_atom_map_list=keep_atom_map_list)
        invalid_molecule = any(
            (
                any(component is None for component in mol)
                or sum(component.GetNumHeavyAtoms() for component in mol) == 0
            )
            if isinstance(mol, tuple)
            else mol is None or mol.GetNumHeavyAtoms() == 0
            for mol in mols
        )
        if any(s == '' for s in mol_smiles) or invalid_molecule:

            invalid_smiles.append(mol_smiles)

    return invalid_smiles


def get_data(path: str,
             smiles_columns: Union[str, List[str]] = None,
             target_columns: List[str] = None,
             ignore_columns: List[str] = None,
             skip_invalid_smiles: bool = True,
             args: Union[TrainArgs, PredictArgs] = None,
             data_weights_path: str = None,
             features_path: List[str] = None,
             features_generator: List[str] = None,
             phase_features_path: str = None,
             atom_descriptors_path: str = None,
             bond_descriptors_path: str = None,
             constraints_path: str = None,
             max_data_size: int = None,
             store_row: bool = False,
             logger: Logger = None,
             loss_function: str = None,
             skip_none_targets: bool = False,
             selected_features_path: str = None) -> MoleculeDataset:
    """
    Gets SMILES and target values from a CSV file.

    :param path: Path to a CSV file.
    :param smiles_columns: The names of the columns containing SMILES.
                           By default, uses the first :code:`number_of_molecules` columns.
    :param target_columns: Name of the columns containing target values. By default, uses all columns
                           except the :code:`smiles_column` and the :code:`ignore_columns`.
    :param ignore_columns: Name of the columns to ignore when :code:`target_columns` is not provided.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles using :func:`filter_invalid_smiles`.
    :param args: Arguments, either :class:`~chemprop.args.TrainArgs` or :class:`~chemprop.args.PredictArgs`.
    :param data_weights_path: A path to a file containing weights for each molecule in the loss function.
    :param features_path: A list of paths to files containing features. If provided, it is used
                          in place of :code:`args.features_path`.
    :param features_generator: A list of features generators to use. If provided, it is used
                               in place of :code:`args.features_generator`.
    :param selected_features_path: Path to a CSV mapping generators to selected descriptor names.
    :param phase_features_path: A path to a file containing phase features as applicable to spectra.
    :param atom_descriptors_path: The path to the file containing the custom atom descriptors.
    :param bond_descriptors_path: The path to the file containing the custom bond descriptors.
    :param constraints_path: The path to the file containing constraints applied to different atomic/bond properties.
    :param max_data_size: The maximum number of data points to load.
    :param logger: A logger for recording output.
    :param store_row: Whether to store the raw CSV row in each :class:`~chemprop.data.data.MoleculeDatapoint`.
    :param skip_none_targets: Whether to skip targets that are all 'None'. This is mostly relevant when --target_columns
                              are passed in, so only a subset of tasks are examined.
    :param loss_function: The loss function to be used in training.
    :return: A :class:`~chemprop.data.MoleculeDataset` containing SMILES and target values along
             with other info such as additional features when desired.
    """
    debug = logger.debug if logger is not None else print

    if args is not None:
        # Prefer explicit function arguments but default to args if not provided
        smiles_columns = smiles_columns if smiles_columns is not None else args.smiles_columns
        target_columns = target_columns if target_columns is not None else args.target_columns
        ignore_columns = ignore_columns if ignore_columns is not None else args.ignore_columns
        features_path = features_path if features_path is not None else args.features_path
        features_generator = features_generator if features_generator is not None else args.features_generator
        selected_features_path = selected_features_path if selected_features_path is not None \
            else args.selected_features_path
        phase_features_path = phase_features_path if phase_features_path is not None else args.phase_features_path
        atom_descriptors_path = atom_descriptors_path if atom_descriptors_path is not None \
            else args.atom_descriptors_path
        bond_descriptors_path = bond_descriptors_path if bond_descriptors_path is not None \
            else args.bond_descriptors_path
        constraints_path = constraints_path if constraints_path is not None else args.constraints_path
        data_weights_path = data_weights_path if data_weights_path is not None \
            else getattr(args, 'data_weights_path', None)
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size
        loss_function = loss_function if loss_function is not None else args.loss_function

    if target_columns is not None:
        target_columns = _expand_quantile_task_names(target_columns, loss_function)

    if isinstance(smiles_columns, str) or smiles_columns is None:
        smiles_columns = preprocess_smiles_columns(path=path, smiles_columns=smiles_columns)

    max_data_size = max_data_size or float('inf')

    feature_paths = [] if features_path is None else features_path
    feature_manifests = [
        _load_feature_manifest(feature_path) for feature_path in feature_paths
    ]
    phase_features_manifest = (
        _load_feature_manifest(phase_features_path)
        if phase_features_path is not None
        else None
    )

    # A save_features manifest identifies the raw, unfiltered CSV input. Build
    # one ordered sequence per configured molecule column: save_features accepts
    # one SMILES column and its ``flatten=True`` result is exactly one of these
    # sequences, even for a multi-molecule Chemprop dataset.
    ordered_smiles_columns = None
    if any(manifest is not None for manifest in feature_manifests) \
            or phase_features_manifest is not None:
        raw_smiles_rows = get_smiles(
            path=path,
            smiles_columns=smiles_columns,
            flatten=False,
        )
        ordered_smiles_columns = [[] for _ in smiles_columns]
        for smiles_row in raw_smiles_rows:
            for column_index, smiles in enumerate(smiles_row):
                ordered_smiles_columns[column_index].append(smiles)
        raw_data_row_count = len(raw_smiles_rows)
    elif (
        feature_paths
        or phase_features_path is not None
        or data_weights_path is not None
        or constraints_path is not None
        or (
            atom_descriptors_path is not None
            and os.path.splitext(atom_descriptors_path)[1].lower() != '.sdf'
        )
        or (
            bond_descriptors_path is not None
            and os.path.splitext(bond_descriptors_path)[1].lower() != '.sdf'
        )
    ):
        with open(path) as raw_data_file:
            raw_data_row_count = sum(1 for _ in csv.DictReader(raw_data_file))
    else:
        raw_data_row_count = None

    # Load features
    external_features_metadata = []
    if feature_paths:
        features_data = []
        for feat_path, manifest in zip(feature_paths, feature_manifests):
            loaded_features = _normalize_feature_matrix(
                feat_path, load_features(feat_path), manifest,
            )
            if len(loaded_features) != raw_data_row_count:
                raise ValueError(
                    f'Feature file {feat_path} has {len(loaded_features)} rows, '
                    f'but the input CSV has {raw_data_row_count} data rows. '
                    'Features must preserve every input row in order.'
                )
            external_features_metadata.append(
                _feature_source_entry(
                    feat_path,
                    loaded_features,
                    manifest=manifest,
                    ordered_smiles_columns=ordered_smiles_columns,
                )
            )
            features_data.append(loaded_features)  # each is num_data x num_features
        feature_row_counts = {len(values) for values in features_data}
        if len(feature_row_counts) != 1:
            raise ValueError(
                'External feature files have inconsistent row counts: '
                + ', '.join(
                    f'{feature_path}={len(values)}'
                    for feature_path, values in zip(feature_paths, features_data)
                )
                + '.'
            )
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    if phase_features_path is not None:
        phase_features = _normalize_feature_matrix(
            phase_features_path,
            load_features(phase_features_path),
            phase_features_manifest,
        )
        if len(phase_features) != raw_data_row_count:
            raise ValueError(
                f'Phase feature file {phase_features_path} has '
                f'{len(phase_features)} rows, but the input CSV has '
                f'{raw_data_row_count} data rows. Phase features must '
                'preserve every input row in order.'
            )
        phase_features_metadata = _feature_source_entry(
            phase_features_path,
            phase_features,
            manifest=phase_features_manifest,
            ordered_smiles_columns=ordered_smiles_columns,
        )
        for d_phase in phase_features:
            if not (d_phase.sum() == 1 and np.count_nonzero(d_phase) == 1):
                raise ValueError('Phase features must be one-hot encoded.')
        if features_data is not None:
            if len(features_data) != len(phase_features):
                raise ValueError(
                    'External and phase feature files have inconsistent row '
                    f'counts: {len(features_data)} and {len(phase_features)}.'
                )
            features_data = np.concatenate((features_data, phase_features), axis=1)
        else:  # if there are no other molecular features, phase features become the only molecular features
            features_data = np.array(phase_features)
    else:
        phase_features = None
        phase_features_metadata = None

    # Resolve target columns before loading constraints or inequality metadata,
    # both of which require the concrete task order. This also keeps the public
    # ``get_data(path, args=None, constraints_path=...)`` API usable.
    if target_columns is None:
        target_columns = get_task_names(
            path=path,
            smiles_columns=smiles_columns,
            target_columns=target_columns,
            ignore_columns=ignore_columns,
            loss_function=loss_function,
        )

    # Load constraints
    if constraints_path is not None:
        constraints_data, raw_constraints_data = get_constraints(
            path=constraints_path,
            target_columns=target_columns,
            save_raw_data=getattr(args, 'save_smiles_splits', False)
        )
        if len(constraints_data) != raw_data_row_count:
            raise ValueError(
                f'Constraints file {constraints_path} has '
                f'{len(constraints_data)} rows, but the input CSV has '
                f'{raw_data_row_count} data rows.'
            )
    else:
        constraints_data = None
        raw_constraints_data = None

    # Load data weights
    if data_weights_path is not None:
        data_weights = get_data_weights(data_weights_path)
        if len(data_weights) != raw_data_row_count:
            raise ValueError(
                f'Data weights file {data_weights_path} has '
                f'{len(data_weights)} rows, but the input CSV has '
                f'{raw_data_row_count} data rows.'
            )
    else:
        data_weights = None

    # Find targets provided as inequalities
    if loss_function == 'bounded_mse':
        gt_targets, lt_targets = get_inequality_targets(path=path, target_columns=target_columns)
    else:
        gt_targets, lt_targets = None, None

    # Load data
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if any([c not in fieldnames for c in smiles_columns]):
            raise ValueError(f'Data file did not contain all provided smiles columns: {smiles_columns}. Data file field names are: {fieldnames}')
        if any([c not in fieldnames for c in target_columns]):
            raise ValueError(f'Data file did not contain all provided target columns: {target_columns}. Data file field names are: {fieldnames}')

        all_smiles, all_targets, all_atom_targets, all_bond_targets, all_rows, all_features, all_phase_features, all_constraints_data, all_raw_constraints_data, all_weights, all_gt, all_lt, all_row_indices = [], [], [], [], [], [], [], [], [], [], [], [], []
        for i, row in enumerate(tqdm(reader)):
            smiles = [row[c] for c in smiles_columns]

            targets, atom_targets, bond_targets = [], [], []
            for column in target_columns:
                value = row[column]
                if value in ['', 'nan']:
                    targets.append(None)
                elif '>' in value or '<' in value:
                    if loss_function == 'bounded_mse':
                        targets.append(float(value.strip('<>')))
                    else:
                        raise ValueError('Inequality found in target data. To use inequality targets (> or <), the regression loss function bounded_mse must be used.')
                elif '[' in value or ']' in value:
                    value = value.replace('None', 'null')
                    target = np.array(json.loads(value))
                    if len(target.shape) == 1 and column in getattr(args, 'atom_targets', []):  # Atom targets saved as 1D list
                        atom_targets.append(target)
                        targets.append(target)
                    elif len(target.shape) == 1 and column in getattr(args, 'bond_targets', []):  # Bond targets saved as 1D list
                        bond_targets.append(target)
                        targets.append(target)
                    elif len(target.shape) == 2:  # Bond targets saved as 2D list
                        bond_target_arranged = []
                        mol = make_mol(
                            smiles[0],
                            getattr(args, 'explicit_h', False),
                            getattr(args, 'adding_h', False),
                            getattr(args, 'keeping_atom_map', False),
                        )
                        for bond in mol.GetBonds():
                            bond_target_arranged.append(target[bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
                        bond_targets.append(np.array(bond_target_arranged))
                        targets.append(np.array(bond_target_arranged))
                    else:
                        raise ValueError(f'Unrecognized targets of column {column} in {path}.')
                else:
                    targets.append(float(value))

            # Check whether all targets are None and skip if so
            if skip_none_targets and all(x is None for x in targets):
                continue

            all_smiles.append(smiles)
            all_targets.append(targets)
            all_atom_targets.append(atom_targets)
            all_bond_targets.append(bond_targets)
            all_row_indices.append(i)

            if features_data is not None:
                if i >= len(features_data):
                    raise ValueError(
                        'Molecular feature files do not contain the feature row '
                        f'for CSV row {i + 1}. Features must preserve the input '
                        'CSV row order, including rows skipped for missing targets.'
                    )
                all_features.append(features_data[i])

            if phase_features is not None:
                all_phase_features.append(phase_features[i])

            if constraints_data is not None:
                all_constraints_data.append(constraints_data[i])

            if raw_constraints_data is not None:
                all_raw_constraints_data.append(raw_constraints_data[i])

            if data_weights is not None:
                all_weights.append(data_weights[i])

            if gt_targets is not None:
                all_gt.append(gt_targets[i])

            if lt_targets is not None:
                all_lt.append(lt_targets[i])

            if store_row:
                all_rows.append(row)

            if len(all_smiles) >= max_data_size:
                break

        atom_features = None
        atom_descriptors = None
        if args is not None and args.atom_descriptors is not None:
            try:
                descriptors = load_valid_atom_or_bond_features(atom_descriptors_path, [x[0] for x in all_smiles])
            except Exception as e:
                raise ValueError(f'Failed to load or validate custom atomic descriptors or features: {e}')
            if os.path.splitext(atom_descriptors_path)[1].lower() != '.sdf':
                if len(descriptors) != raw_data_row_count:
                    raise ValueError(
                        f'Atom descriptor file {atom_descriptors_path} has '
                        f'{len(descriptors)} rows, but the input CSV has '
                        f'{raw_data_row_count} data rows.'
                    )
                descriptors = [descriptors[index] for index in all_row_indices]
            elif len(descriptors) != len(all_smiles):
                raise ValueError(
                    f'Atom descriptor file {atom_descriptors_path} did not '
                    'resolve exactly one entry per loaded molecule.'
                )

            if args.atom_descriptors == 'feature':
                atom_features = descriptors
            elif args.atom_descriptors == 'descriptor':
                atom_descriptors = descriptors

        bond_features = None
        bond_descriptors = None
        if args is not None and args.bond_descriptors is not None:
            try:
                descriptors = load_valid_atom_or_bond_features(bond_descriptors_path, [x[0] for x in all_smiles])
            except Exception as e:
                raise ValueError(f'Failed to load or validate custom bond descriptors or features: {e}')
            if os.path.splitext(bond_descriptors_path)[1].lower() != '.sdf':
                if len(descriptors) != raw_data_row_count:
                    raise ValueError(
                        f'Bond descriptor file {bond_descriptors_path} has '
                        f'{len(descriptors)} rows, but the input CSV has '
                        f'{raw_data_row_count} data rows.'
                    )
                descriptors = [descriptors[index] for index in all_row_indices]
            elif len(descriptors) != len(all_smiles):
                raise ValueError(
                    f'Bond descriptor file {bond_descriptors_path} did not '
                    'resolve exactly one entry per loaded molecule.'
                )

            if args.bond_descriptors == 'feature':
                bond_features = descriptors
            elif args.bond_descriptors == 'descriptor':
                bond_descriptors = descriptors

        selected_feature_columns = (
            load_selected_feature_columns(selected_features_path)
            if selected_features_path is not None
            else {}
        )

        generated_features = None
        if features_generator:
            debug(
                'Generating molecular features in batches: '
                + ', '.join(features_generator)
            )
            generated_features = generate_features_for_smiles_batch(
                all_smiles,
                features_generator,
                selected_feature_columns,
                use_atom_mapping_for_hydrogens=[
                    bool(atom_row or bond_row)
                    for atom_row, bond_row in zip(all_atom_targets, all_bond_targets)
                ],
            )

        observed_generated_dimension = max(
            (int(np.asarray(row).size) for row in generated_features),
            default=0,
        ) if generated_features is not None else 0
        generated_dimension = observed_generated_dimension
        if observed_generated_dimension == 0:
            generated_dimension = _static_generated_dimension(
                features_generator or [],
                selected_feature_columns,
                number_of_molecules=len(smiles_columns),
            )
        features_source_metadata = _features_source_metadata(
            external_features=external_features_metadata,
            phase_features=phase_features_metadata,
            generated_dimension=generated_dimension,
        )

        generated_features_precomputed = generated_features is not None
        if features_data is None:
            # Avoid copying every generated vector through a one-element
            # concatenate before MoleculeDatapoint construction.
            combined_features = generated_features
        elif generated_features is None:
            combined_features = all_features
        else:
            combined_features = [
                np.concatenate((
                    np.asarray(all_features[index]),
                    np.asarray(generated_features[index]),
                ))
                for index in range(len(all_smiles))
            ]
            # The combined arrays own their data, so release both source
            # collections before constructing all datapoints.
            generated_features = None
            features_data = None
            all_features = []

        data = MoleculeDataset([
            MoleculeDatapoint(
                smiles=smiles,
                targets=targets,
                atom_targets=all_atom_targets[i] if all_atom_targets[i] else None,
                bond_targets=all_bond_targets[i] if all_bond_targets[i] else None,
                row=all_rows[i] if store_row else None,
                data_weight=all_weights[i] if data_weights is not None else None,
                gt_targets=all_gt[i] if gt_targets is not None else None,
                lt_targets=all_lt[i] if lt_targets is not None else None,
                features_generator=features_generator,
                features_generator_precomputed=generated_features_precomputed,
                selected_features_path=selected_features_path,
                selected_feature_columns=selected_feature_columns,
                features=combined_features[i] if combined_features is not None else None,
                phase_features=all_phase_features[i] if phase_features is not None else None,
                atom_features=atom_features[i] if atom_features is not None else None,
                atom_descriptors=atom_descriptors[i] if atom_descriptors is not None else None,
                bond_features=bond_features[i] if bond_features is not None else None,
                bond_descriptors=bond_descriptors[i] if bond_descriptors is not None else None,
                constraints=all_constraints_data[i] if constraints_data is not None else None,
                raw_constraints=all_raw_constraints_data[i] if raw_constraints_data is not None else None,
                overwrite_default_atom_features=args.overwrite_default_atom_features if args is not None else False,
                overwrite_default_bond_features=args.overwrite_default_bond_features if args is not None else False
            ) for i, (smiles, targets) in tqdm(enumerate(zip(all_smiles, all_targets)),
                                            total=len(all_smiles))
        ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    data._features_source_metadata = features_source_metadata

    return data


def get_data_from_smiles(smiles: List[List[str]],
                         skip_invalid_smiles: bool = True,
                         logger: Logger = None,
                         features_generator: List[str] = None,
                         selected_features_path: str = None) -> MoleculeDataset:
    """
    Converts a list of SMILES to a :class:`~chemprop.data.MoleculeDataset`.

    :param smiles: A list of lists of SMILES with length depending on the number of molecules.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles using :func:`filter_invalid_smiles`
    :param logger: A logger for recording output.
    :param features_generator: List of features generators.
    :param selected_features_path: Path to a CSV mapping generators to selected descriptor names.
    :return: A :class:`~chemprop.data.MoleculeDataset` with all of the provided SMILES.
    """
    debug = logger.debug if logger is not None else print

    selected_feature_columns = (
        load_selected_feature_columns(selected_features_path)
        if selected_features_path is not None
        else {}
    )

    generated_features = (
        generate_features_for_smiles_batch(
            smiles,
            features_generator,
            selected_feature_columns,
        )
        if features_generator
        else None
    )
    observed_generated_dimension = max(
        (int(np.asarray(row).size) for row in generated_features),
        default=0,
    ) if generated_features is not None else 0
    if features_generator and not smiles:
        # An empty public SMILES list does not reveal how many molecule columns
        # the checkpoint expects. Prediction validation treats this width as
        # unknown and relies on ordered generator metadata instead.
        generated_dimension = None
    else:
        generated_dimension = observed_generated_dimension
        if observed_generated_dimension == 0:
            generated_dimension = _static_generated_dimension(
                features_generator or [],
                selected_feature_columns,
                number_of_molecules=len(smiles[0]) if smiles else 0,
            )

    data = MoleculeDataset([
        MoleculeDatapoint(
            smiles=smile,
            row=OrderedDict({'smiles': smile}),
            features_generator=features_generator,
            features_generator_precomputed=generated_features is not None,
            selected_features_path=selected_features_path,
            selected_feature_columns=selected_feature_columns,
            features=generated_features[index] if generated_features is not None else None,
        ) for index, smile in enumerate(smiles)
    ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    data._features_source_metadata = _features_source_metadata(
        external_features=[],
        phase_features=None,
        generated_dimension=generated_dimension,
    )

    return data


def get_inequality_targets(path: str, target_columns: List[str] = None) -> List[str]:
    """

    """
    gt_targets = []
    lt_targets = []

    with open(path) as f:
        reader = csv.DictReader(f)
        for line in reader:
            values = [line[col] for col in target_columns]
            gt_targets.append(['>' in val for val in values])
            lt_targets.append(['<' in val for val in values])
            if any(['<' in val and '>' in val for val in values]):
                raise ValueError(f'A target value in csv file {path} contains both ">" and "<" symbols. Inequality targets must be on one edge and not express a range.')

    return gt_targets, lt_targets

def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               key_molecule_index: int = 0,
               seed: int = 0,
               num_folds: int = 1,
               args: TrainArgs = None,
               logger: Logger = None) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:
    r"""
    Splits data into training, validation, and test splits.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :param split_type: Split type.
    :param sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    :param key_molecule_index: For data with multiple molecules, this sets which molecule will be considered during splitting.
    :param seed: The random seed to use before shuffling data.
    :param num_folds: Number of folds to create (only needed for "cv" split type).
    :param args: A :class:`~chemprop.args.TrainArgs` object.
    :param logger: A logger for recording output.
    :return: A tuple of :class:`~chemprop.data.MoleculeDataset`\ s containing the train,
             validation, and test splits of the data.
    """
    if not (len(sizes) == 3 and np.isclose(sum(sizes), 1)):
        raise ValueError(f"Split sizes do not sum to 1. Received train/val/test splits: {sizes}")
    if any([size < 0 for size in sizes]):
        raise ValueError(f"Split sizes must be non-negative. Received train/val/test splits: {sizes}")

    random = Random(seed)

    if args is not None:
        folds_file, val_fold_index, test_fold_index = \
            args.folds_file, args.val_fold_index, args.test_fold_index
    else:
        folds_file = val_fold_index = test_fold_index = None

    if split_type == 'crossval':
        index_set = args.crossval_index_sets[args.seed]
        data_split = []
        for split in range(3):
            split_indices = []
            for index in index_set[split]:
                with open(os.path.join(args.crossval_index_dir, f'{index}.pkl'), 'rb') as rf:
                    split_indices.extend(pickle.load(rf))
            data_split.append([data[i] for i in split_indices])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type in {'cv', 'cv-no-test'}:
        if num_folds <= 1 or num_folds > len(data):
            raise ValueError(f'Number of folds for cross-validation must be between 2 and the number of valid datapoints ({len(data)}), inclusive.')

        random = Random(0)

        indices = np.tile(np.arange(num_folds), 1 + len(data) // num_folds)[:len(data)]
        random.shuffle(indices)
        test_index = seed % num_folds
        val_index = (seed + 1) % num_folds

        train, val, test = [], [], []
        for d, index in zip(data, indices):
            if index == test_index and split_type != 'cv-no-test':
                test.append(d)
            elif index == val_index:
                val.append(d)
            else:
                train.append(d)

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'index_predetermined':
        split_indices = args.crossval_index_sets[args.seed]

        if len(split_indices) != 3:
            raise ValueError('Split indices must have three splits: train, validation, and test')

        data_split = []
        for split in range(3):
            data_split.append([data[i] for i in split_indices[split]])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'predetermined':
        if not val_fold_index and sizes[2] != 0:
            raise ValueError('Test size must be zero since test set is created separately '
                             'and we want to put all other data in train and validation')

        if folds_file is None:
            raise ValueError('arg "folds_file" can not be None!')
        if test_fold_index is None:
            raise ValueError('arg "test_fold_index" can not be None!')

        try:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f)
        except UnicodeDecodeError:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f, encoding='latin1')  # in case we're loading indices from python2

        log_scaffold_stats(data, all_fold_indices, logger=logger)

        folds = [[data[i] for i in fold_indices] for fold_indices in all_fold_indices]

        test = folds[test_fold_index]
        if val_fold_index is not None:
            val = folds[val_fold_index]

        train_val = []
        for i in range(len(folds)):
            if i != test_fold_index and (val_fold_index is None or i != val_fold_index):
                train_val.extend(folds[i])

        if val_fold_index is not None:
            train = train_val
        else:
            random.shuffle(train_val)
            train_size = int(sizes[0] * len(train_val))
            train = train_val[:train_size]
            val = train_val[train_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, key_molecule_index=key_molecule_index, seed=seed, logger=logger)

    elif split_type == 'random_with_repeated_smiles':  # Use to constrain data with the same smiles go in the same split.
        smiles_dict = defaultdict(set)
        for i, smiles in enumerate(data.smiles()):
            smiles_dict[smiles[key_molecule_index]].add(i)
        index_sets = list(smiles_dict.values())
        random.seed(seed)
        random.shuffle(index_sets)
        train, val, test = [], [], []
        train_size = int(sizes[0] * len(data))
        val_size = int(sizes[1] * len(data))
        for index_set in index_sets:
            if len(train)+len(index_set) <= train_size:
                train += index_set
            elif len(val) + len(index_set) <= val_size:
                val += index_set
            else:
                test += index_set
        train = [data[i] for i in train]
        val = [data[i] for i in val]
        test = [data[i] for i in test]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'random':
        indices = list(range(len(data)))
        random.shuffle(indices)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = [data[i] for i in indices[:train_size]]
        val = [data[i] for i in indices[train_size:train_val_size]]
        test = [data[i] for i in indices[train_val_size:]]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)
    elif split_type == 'molecular_weight':
        train_size, val_size, test_size = [int(size * len(data)) for size in sizes]

        sorted_data = sorted(data._data, key=lambda x: x.max_molwt, reverse=False)
        indices = list(range(len(sorted_data)))

        train_end_idx = int(train_size)
        val_end_idx = int(train_size + val_size)
        train_indices = indices[:train_end_idx]
        val_indices = indices[train_end_idx:val_end_idx]
        test_indices = indices[val_end_idx:]

        # Create MoleculeDataset for each split
        train = MoleculeDataset([sorted_data[i] for i in train_indices])
        val = MoleculeDataset([sorted_data[i] for i in val_indices])
        test = MoleculeDataset([sorted_data[i] for i in test_indices])

        return train, val, test
    else:
        raise ValueError(f'split_type "{split_type}" not supported.')


def get_class_sizes(data: MoleculeDataset, proportion: bool = True) -> List[List[float]]:
    """
    Determines the proportions of the different classes in a classification dataset.

    :param data: A classification :class:`~chemprop.data.MoleculeDataset`.
    :param proportion: Choice of whether to return proportions for class size or counts.
    :return: A list of lists of class proportions. Each inner list contains the class proportions for a task.
    """
    targets = data.targets()

    # Filter out Nones
    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if data.is_atom_bond_targets:
                for target in targets[i][task_num]:
                    if targets[i][task_num] is not None:
                        valid_targets[task_num].append(target)
            else:
                if targets[i][task_num] is not None:
                    valid_targets[task_num].append(targets[i][task_num])

    class_sizes = []
    for task_targets in valid_targets:
        if set(np.unique(task_targets)) > {0, 1}:
            raise ValueError('Classification dataset must only contains 0s and 1s.')
        if proportion:
            try:
                ones = np.count_nonzero(task_targets) / len(task_targets)
            except ZeroDivisionError:
                ones = float('nan')
                print('Warning: class has no targets')
            class_sizes.append([1 - ones, ones])
        else:  # counts
            ones = np.count_nonzero(task_targets)
            class_sizes.append([len(task_targets) - ones, ones])

    return class_sizes


#  TODO: Validate multiclass dataset type.
def validate_dataset_type(data: MoleculeDataset, dataset_type: str) -> None:
    """
    Validates the dataset type to ensure the data matches the provided type.

    :param data: A :class:`~chemprop.data.MoleculeDataset`.
    :param dataset_type: The dataset type to check.
    """
    target_list = [target for targets in data.targets() for target in targets]

    if data.is_atom_bond_targets:
        target_set = set(list(np.concatenate(target_list).flat)) - {None}
    else:
        target_set = set(target_list) - {None}
    classification_target_set = {0, 1}

    if dataset_type == 'classification' and not (target_set <= classification_target_set):
        raise ValueError('Classification data targets must only be 0 or 1 (or None). '
                         'Please switch to regression.')
    elif dataset_type == 'regression' and target_set <= classification_target_set:
        raise ValueError('Regression data targets must be more than just 0 or 1 (or None). '
                         'Please switch to classification.')


def validate_data(data_path: str) -> Set[str]:
    """
    Validates a data CSV file, returning a set of errors.

    :param data_path: Path to a data CSV file.
    :return: A set of error messages.
    """
    errors = set()

    header = get_header(data_path)

    with open(data_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        smiles, targets = [], []
        for line in reader:
            smiles.append(line[0])
            targets.append(line[1:])

    # Validate header
    if len(header) == 0:
        errors.add('Empty header')
    elif len(header) < 2:
        errors.add('Header must include task names.')

    mol = Chem.MolFromSmiles(header[0])
    if mol is not None:
        errors.add('First row is a SMILES string instead of a header.')

    # Validate smiles
    for smile in tqdm(smiles, total=len(smiles)):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            errors.add('Data includes an invalid SMILES.')

    # Validate targets
    num_tasks_set = set(len(mol_targets) for mol_targets in targets)
    if len(num_tasks_set) != 1:
        errors.add('Inconsistent number of tasks for each molecule.')

    if len(num_tasks_set) == 1:
        num_tasks = num_tasks_set.pop()
        if num_tasks != len(header) - 1:
            errors.add('Number of tasks for each molecule doesn\'t match number of tasks in header.')

    unique_targets = set(np.unique([target for mol_targets in targets for target in mol_targets]))

    if unique_targets <= {''}:
        errors.add('All targets are missing.')

    for target in unique_targets - {''}:
        try:
            float(target)
        except ValueError:
            errors.add('Found a target which is not a number.')

    return errors
