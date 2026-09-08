import csv
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools


_PICKLE_EXTENSIONS = {'.pkl', '.pckl', '.pickle'}


def _align_pickle_descriptors(
    features_df: pd.DataFrame,
    smiles: List[str],
    path: str,
) -> pd.DataFrame:
    """Aligns a descriptor dataframe to an ordered, complete SMILES input."""
    expected = list(smiles)
    actual = features_df.index.tolist()

    # Duplicate SMILES cannot be safely reordered because their per-row
    # descriptors may differ. They are unambiguous when already in input order.
    if actual == expected:
        return features_df

    if len(actual) != len(expected):
        raise ValueError(
            f'Atom/bond descriptors input {path} has {len(actual)} rows, but '
            f'{len(expected)} SMILES were provided.'
        )

    expected_index = pd.Index(expected)
    if features_df.index.has_duplicates or expected_index.has_duplicates:
        raise ValueError(
            f'Atom/bond descriptors input {path} contains duplicate SMILES '
            'and is not already in exactly the same order as the input data; '
            'the row correspondence is ambiguous.'
        )

    missing = expected_index.difference(features_df.index).tolist()
    extra = features_df.index.difference(expected_index).tolist()
    if missing or extra:
        raise ValueError(
            f'Atom/bond descriptors input {path} does not match the input '
            f'SMILES (missing={missing[:5]!r}, extra={extra[:5]!r}).'
        )

    return features_df.reindex(expected)


def _align_sdf_descriptors(
    features_df: pd.DataFrame,
    smiles: List[str],
    path: str,
) -> pd.DataFrame:
    """Selects requested SMILES from an SDF without choosing duplicates."""
    expected_index = pd.Index(smiles)
    duplicated = features_df.index[
        features_df.index.duplicated(keep=False)
    ].unique()
    ambiguous = expected_index.intersection(duplicated).tolist()
    if ambiguous:
        raise ValueError(
            f'Atom/bond descriptors input {path} contains multiple SDF '
            f'records for requested SMILES {ambiguous[:5]!r}.'
        )

    missing = expected_index.difference(features_df.index).tolist()
    if missing:
        raise ValueError(
            f'Atom/bond descriptors input {path} is missing requested SMILES '
            f'{missing[:5]!r}.'
        )

    return features_df.reindex(smiles)


def _descriptor_frame_to_arrays(
    features_df: pd.DataFrame,
    path: str,
) -> List[np.ndarray]:
    """Combines descriptor columns while validating their dimensionality."""
    if len(features_df) == 0:
        return []
    cell_dimensions = {
        np.asarray(value).ndim
        for value in features_df.to_numpy(dtype=object).flat
    }
    if cell_dimensions not in ({1}, {2}):
        raise ValueError(
            f'Atom/bond descriptors input {path} must contain consistently '
            '1-D arrays or consistently 2-D arrays.'
        )

    dimensions = next(iter(cell_dimensions))
    features = []
    for row_number, row in enumerate(features_df.itertuples(index=False, name=None)):
        arrays = [np.asarray(value) for value in row]
        try:
            combined = (
                np.stack(arrays, axis=1)
                if dimensions == 1
                else np.concatenate(arrays, axis=1)
            )
        except ValueError as error:
            raise ValueError(
                f'Atom/bond descriptors input {path} has inconsistent shapes '
                f'in descriptor row {row_number + 1}.'
            ) from error
        features.append(combined)

    return features


def save_features(path: str, features: List[np.ndarray]) -> None:
    """
    Saves features to a compressed :code:`.npz` file with array name "features".

    :param path: Path to a :code:`.npz` file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:

    * :code:`.npz` compressed (assumes features are saved with name "features")
    * .npy
    * :code:`.csv` / :code:`.txt` (assumes comma-separated features with a header and with one line per molecule)
    * :code:`.pkl` / :code:`.pckl` / :code:`.pickle` containing a sparse numpy array

    .. note::

       All formats assume that the SMILES loaded elsewhere in the code are in the same
       order as the features loaded here.

    .. warning::

       Python pickle formats can execute code while loading. Only load ``.pkl``,
       ``.pckl`` or ``.pickle`` feature files from a trusted source.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size :code:`(num_molecules, features_size)` containing the features.
    """
    extension = os.path.splitext(path)[1].lower()

    if extension == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            if 'features' not in archive.files:
                raise ValueError(
                    f'Compressed feature archive {path} does not contain an array named "features".'
                )
            features = archive['features']
    elif extension == '.npy':
        features = np.load(path, allow_pickle=False)
    elif extension in ['.csv', '.txt']:
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            features = np.array([[float(value) for value in row] for row in reader])
    elif extension in _PICKLE_EXTENSIONS:
        with open(path, 'rb') as f:
            features = np.array([np.squeeze(np.array(feat.todense())) for feat in pickle.load(f)])
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


def load_valid_atom_or_bond_features(path: str, smiles: List[str]) -> List[np.ndarray]:
    """
    Loads features saved in a variety of formats.

    Supported formats:

    * :code:`.npz` descriptors are saved as 2D array for each molecule in the order of that in the data.csv
    * :code:`.pkl` / :code:`.pckl` / :code:`.pickle` containing a pandas dataframe with smiles as index and numpy array of descriptors as columns
    * :code:'.sdf' containing all mol blocks with descriptors as entries

    :param path: Path to file containing atomwise features.
    :return: A list of 2D array.

    .. warning::

       Python pickle formats can execute code while loading. Only load ``.pkl``,
       ``.pckl`` or ``.pickle`` descriptor files from a trusted source.
    """

    extension = os.path.splitext(path)[1].lower()

    if extension == '.npz':
        with np.load(path, allow_pickle=False) as container:
            features = [container[key] for key in container.files]

    elif extension in _PICKLE_EXTENSIONS:
        features_df = pd.read_pickle(path)
        if features_df.empty or features_df.shape[1] == 0:
            raise ValueError(f'Atom/bond descriptors input {path} is empty')
        features_df = _align_pickle_descriptors(features_df, smiles, path)
        features = _descriptor_frame_to_arrays(features_df, path)

    elif extension == '.sdf':
        features_df = PandasTools.LoadSDF(path)
        if features_df is None or features_df.empty:
            raise ValueError(f'Atom/bond descriptors input {path} is empty')
        if 'SMILES' not in features_df.columns:
            raise ValueError(
                f'Atom/bond descriptors input {path} has no SMILES property.'
            )

        features_df = features_df.drop(
            columns=['ID', 'ROMol'], errors='ignore'
        ).set_index('SMILES')

        # A property can be scalar for a one-atom molecule. Inspect every row,
        # rather than only the first, when locating descriptor columns.
        descriptor_columns = [
            column
            for column in features_df.columns
            if any(
                isinstance(value, str) and ',' in value
                for value in features_df[column]
            )
        ]
        if not descriptor_columns:
            raise ValueError(
                f'Atom/bond descriptors input {path} contains no '
                'comma-separated descriptor properties.'
            )
        features_df = _align_sdf_descriptors(
            features_df[descriptor_columns], smiles, path,
        )

        def parse_sdf_descriptor(value: object) -> np.ndarray:
            if not isinstance(value, str):
                raise ValueError('descriptor value is not a string')
            cleaned = value.replace('\r', '').replace('\n', '')
            return np.asarray(cleaned.split(','), dtype=float)

        try:
            features_df = features_df.apply(
                lambda column: column.map(parse_sdf_descriptor)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'Atom/bond descriptors input {path} contains missing or '
                'non-numeric descriptor values.'
            ) from error

        features = _descriptor_frame_to_arrays(features_df, path)

    else:
        raise ValueError(f'Extension "{extension}" is not supported.')

    return features
