import csv
import os
import threading
from collections import OrderedDict
from random import Random
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Union, Tuple

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset, Sampler
from rdkit import Chem

from .scaler import StandardScaler, AtomBondScaler
from chemprop.features import generate_features_batch, get_features_generator
from chemprop.features import BatchMolGraph, MolGraph
from chemprop.features import is_explicit_h, is_reaction, is_adding_hs, is_mol, is_keeping_atom_map
from chemprop.features.featurization import reaction_mode
from chemprop.rdkit import make_mol

# Cache of graph featurizations
CACHE_GRAPH = True
SMILES_TO_GRAPH: Dict[Tuple[object, ...], MolGraph] = {}


# Cache of RDKit molecules
CACHE_MOL = True
SMILES_TO_MOL: Dict[Tuple[object, ...], Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]] = {}


# Small, bounded cache for duplicate-SMILES molecular descriptors. Keeping this
# separate from the graph cache avoids retaining an unbounded number of large
# dense feature vectors during long-running processes.
MAX_FEATURE_CACHE_SIZE = 1024
FEATURES_CACHE: "OrderedDict[Tuple[object, ...], np.ndarray]" = OrderedDict()
FEATURES_CACHE_LOCK = threading.Lock()


# Parsed selected-feature files are shared by all datapoints. The file metadata
# is part of the key so editing a CSV invalidates the cached mapping.
SELECTED_FEATURES_CACHE: Dict[Tuple[object, ...], Dict[str, Tuple[str, ...]]] = {}
SELECTED_FEATURES_CACHE_LOCK = threading.Lock()


class _CallableIdentity:
    """Hashable identity wrapper for arbitrary (even unhashable) callables."""

    __slots__ = ('value',)

    def __init__(self, value) -> None:
        self.value = value

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other) -> bool:
        return isinstance(other, _CallableIdentity) and self.value is other.value


def load_selected_feature_columns(path: str) -> Dict[str, Tuple[str, ...]]:
    """Loads a selected-feature CSV once and returns generator-to-column mappings."""
    resolved_path = os.path.abspath(os.path.expanduser(path))
    stat = os.stat(resolved_path)
    cache_key = (
        resolved_path,
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )

    with SELECTED_FEATURES_CACHE_LOCK:
        cached = SELECTED_FEATURES_CACHE.get(cache_key)
        if cached is not None:
            # Never expose the cache's mutable dictionary to callers. Values
            # are tuples, so a shallow copy is sufficient.
            return dict(cached)

    with open(
        resolved_path, newline='', encoding='utf-8-sig',
    ) as selected_features_file:
        try:
            raw_header = next(csv.reader(selected_features_file))
        except StopIteration as error:
            raise ValueError(
                'Selected-feature CSV must contain a header row.'
            ) from error
    if not raw_header or any(not column.strip() for column in raw_header):
        raise ValueError(
            'Selected-feature CSV must contain non-blank generator names.'
        )
    if len(raw_header) != len(set(raw_header)):
        raise ValueError(
            'Selected-feature CSV contains duplicate generator columns.'
        )

    selected_features_df = pd.read_csv(resolved_path, encoding='utf-8-sig')
    mapping = {
        column: tuple(str(value) for value in selected_features_df[column].dropna().values)
        for column in selected_features_df.columns
    }

    with SELECTED_FEATURES_CACHE_LOCK:
        # Discard stale entries for the same file before storing the new version.
        stale_keys = [key for key in SELECTED_FEATURES_CACHE if key[0] == resolved_path]
        for key in stale_keys:
            SELECTED_FEATURES_CACHE.pop(key, None)
        SELECTED_FEATURES_CACHE[cache_key] = mapping

    return dict(mapping)


def _feature_cache_key(features_generator: str,
                       generator,
                       mol: Chem.Mol,
                       selected_feature_columns: Optional[Sequence[str]]) -> Tuple[object, ...]:
    # Custom registry functions may depend on RDKit atom order. Canonicalizing
    # the key would incorrectly merge inputs such as ``CO`` and ``OC`` even
    # though their atom-0 features differ.
    smiles = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
    selected = None if selected_feature_columns is None else tuple(selected_feature_columns)
    # Include the callable because register_features_generator intentionally
    # permits replacing a name during extension/plugin development.
    return features_generator, _CallableIdentity(generator), selected, smiles


def _generate_features(features_generator_name: str,
                       mol: Chem.Mol,
                       selected_feature_columns: Optional[Sequence[str]]) -> np.ndarray:
    """Generates molecular features with a small duplicate-SMILES LRU cache."""
    generator = get_features_generator(features_generator_name)
    key = _feature_cache_key(
        features_generator_name, generator, mol, selected_feature_columns
    )
    with FEATURES_CACHE_LOCK:
        cached = FEATURES_CACHE.get(key)
        if cached is not None:
            FEATURES_CACHE.move_to_end(key)
            return cached.copy()

    generated = np.asarray(
        generator(mol, selected_feature_columns=selected_feature_columns)
    )

    with FEATURES_CACHE_LOCK:
        FEATURES_CACHE[key] = generated.copy()
        FEATURES_CACHE.move_to_end(key)
        while len(FEATURES_CACHE) > MAX_FEATURE_CACHE_SIZE:
            FEATURES_CACHE.popitem(last=False)

    return generated


def generate_features_for_smiles_batch(
    smiles_rows: Sequence[Sequence[str]],
    features_generators: Sequence[str],
    selected_feature_columns: Mapping[str, Sequence[str]] = None,
    use_atom_mapping_for_hydrogens: Union[bool, Sequence[bool]] = False,
) -> List[np.ndarray]:
    """Generates dataset features in chunks while preserving v1 concatenation order.

    The historical :class:`MoleculeDatapoint` path loops over generators, rows,
    and molecule columns one molecule at a time. This bulk path applies the
    same molecule/reaction/H2 rules, deduplicates atom-order-preserving SMILES
    within the dataset, invokes native batch transforms where available, and
    scatters the results back into ``generator -> molecule column`` order for
    each row.
    """
    if not features_generators:
        return [np.empty(0, dtype=float) for _ in smiles_rows]

    selected_feature_columns = selected_feature_columns or {}
    if isinstance(use_atom_mapping_for_hydrogens, bool):
        atom_mapping_rows = [use_atom_mapping_for_hydrogens] * len(smiles_rows)
    else:
        atom_mapping_rows = list(use_atom_mapping_for_hydrogens)
        if len(atom_mapping_rows) != len(smiles_rows):
            raise ValueError(
                'use_atom_mapping_for_hydrogens must contain one flag per SMILES row.'
            )

    parsed_rows = []
    for smiles, use_atom_mapping in zip(smiles_rows, atom_mapping_rows):
        is_mol_list = [is_mol(value) for value in smiles]
        reaction_list = [is_reaction(value) for value in is_mol_list]
        mols = make_mols(
            smiles=list(smiles),
            reaction_list=reaction_list,
            keep_h_list=[
                is_keeping_atom_map(value)
                if use_atom_mapping
                else is_explicit_h(value)
                for value in is_mol_list
            ],
            add_h_list=[is_adding_hs(value) for value in is_mol_list],
            keep_atom_map_list=[is_keeping_atom_map(value) for value in is_mol_list],
        )
        parsed_rows.append((mols, reaction_list))

    # Molecule positions and duplicate keys do not depend on the generator.
    # Computing them once avoids a full dataset traversal and SMILES
    # serialization for every requested generator.
    position_keys: List[List[Optional[str]]] = []
    unique_molecules: "OrderedDict[str, Chem.Mol]" = OrderedDict()
    has_hydrogen_only = False
    for mols, reaction_list in parsed_rows:
        row_keys: List[Optional[str]] = []
        for mol, reaction in zip(mols, reaction_list):
            candidate = None
            if reaction:
                if mol[0] is not None and mol[1] is not None:
                    candidate = mol[0]
            elif mol is not None:
                candidate = mol

            if candidate is None:
                row_keys.append(None)
            elif candidate.GetNumHeavyAtoms() == 0:
                row_keys.append('__CHEMPROP_HYDROGEN_ONLY__')
                has_hydrogen_only = True
            else:
                key = Chem.MolToSmiles(
                    candidate, canonical=False, isomericSmiles=True
                )
                row_keys.append(key)
                unique_molecules.setdefault(key, candidate)
        position_keys.append(row_keys)

    # Keep whole NumPy vectors rather than expanding every scalar into a
    # Python list entry. Dense fingerprints otherwise create hundreds of
    # millions of temporary Python objects on large datasets.
    row_feature_parts: List[List[np.ndarray]] = [[] for _ in smiles_rows]
    methane = Chem.MolFromSmiles('C')
    keys = list(unique_molecules)
    unique_molecule_values = list(unique_molecules.values())

    for generator_name in features_generators:
        selected = selected_feature_columns.get(generator_name)

        generated_rows = generate_features_batch(
            generator_name,
            unique_molecule_values,
            selected_feature_columns=selected,
        )
        generated_by_key = {
            key: np.asarray(values) for key, values in zip(keys, generated_rows)
        }

        if has_hydrogen_only:
            zero_template = np.asarray(generate_features_batch(
                generator_name,
                [methane],
                selected_feature_columns=selected,
            )[0])
        else:
            zero_template = None

        expected_size = None
        for values in list(generated_by_key.values()) + (
            [zero_template] if zero_template is not None else []
        ):
            if values.ndim != 1:
                raise ValueError(
                    f'Features generator "{generator_name}" returned a '
                    f'{values.ndim}-dimensional value; expected a 1-D vector.'
                )
            if expected_size is None:
                expected_size = len(values)
            elif len(values) != expected_size:
                raise ValueError(
                    f'Features generator "{generator_name}" returned inconsistent '
                    f'lengths ({expected_size} and {len(values)}).'
                )

        zero_values = (
            np.zeros(len(zero_template), dtype=float)
            if zero_template is not None
            else None
        )
        for row_index, row_keys in enumerate(position_keys):
            for key in row_keys:
                # Invalid inputs historically append no values and are filtered
                # by get_data afterwards when skip_invalid_smiles is enabled.
                if key is None:
                    continue
                values = (
                    zero_values
                    if key == '__CHEMPROP_HYDROGEN_ONLY__'
                    else generated_by_key[key]
                )
                # Extending an empty vector historically had no influence on
                # the final dtype, so omit it from NumPy concatenation too.
                if values.size:
                    row_feature_parts[row_index].append(values)

    return [
        np.concatenate(parts) if parts else np.empty(0, dtype=float)
        for parts in row_feature_parts
    ]


def cache_graph() -> bool:
    r"""Returns whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    return CACHE_GRAPH


def set_cache_graph(cache_graph: bool) -> None:
    r"""Sets whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph


def empty_cache():
    r"""Empties the cache of :class:`~chemprop.features.MolGraph` and RDKit molecules."""
    SMILES_TO_GRAPH.clear()
    SMILES_TO_MOL.clear()
    with FEATURES_CACHE_LOCK:
        FEATURES_CACHE.clear()
    with SELECTED_FEATURES_CACHE_LOCK:
        SELECTED_FEATURES_CACHE.clear()


def cache_mol() -> bool:
    r"""Returns whether RDKit molecules will be cached."""
    return CACHE_MOL


def set_cache_mol(cache_mol: bool) -> None:
    r"""Sets whether RDKit molecules will be cached."""
    global CACHE_MOL
    CACHE_MOL = cache_mol


def _sanitize_feature_array(
    values: np.ndarray,
    name: str,
    expected_ndim: int,
) -> np.ndarray:
    """Returns a numeric feature array with NaNs replaced and no infinities."""
    array = np.asarray(values)
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f'{name} must be a real-valued numeric array.')
    if array.ndim != expected_ndim:
        raise ValueError(
            f'{name} must be a {expected_ndim}-D numeric array, got shape '
            f'{array.shape}.'
        )
    if np.any(np.isinf(array)):
        raise ValueError(f'{name} contains an infinite value.')

    # Preserve the historical behavior of treating missing descriptors as 0.
    return np.where(np.isnan(array), 0, array)


class MoleculeDatapoint:
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    def __init__(self,
                 smiles: List[str],
                 targets: List[Optional[float]] = None,
                 atom_targets: List[Optional[float]] = None,
                 bond_targets: List[Optional[float]] = None,
                 row: OrderedDict = None,
                 data_weight: float = None,
                 gt_targets: List[List[bool]] = None,
                 lt_targets: List[List[bool]] = None,
                 features: np.ndarray = None,
                 features_generator: List[str] = None,
                 selected_features_path: str = None,
                 phase_features: List[float] = None,
                 atom_features: np.ndarray = None,
                 atom_descriptors: np.ndarray = None,
                 bond_features: np.ndarray = None,
                 bond_descriptors: np.ndarray = None,
                 raw_constraints: np.ndarray = None,
                 constraints: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False,
                 selected_feature_columns: Mapping[str, Sequence[str]] = None,
                 features_generator_precomputed: bool = False):
        """
        :param smiles: A list of the SMILES strings for the molecules.
        :param targets: A list of targets for the molecule (contains None for unknown target values).
        :param atom_targets: A list of targets for the atomic properties.
        :param bond_targets: A list of targets for the bond properties.
        :param row: The raw CSV row containing the information for this molecule.
        :param data_weight: Weighting of the datapoint for the loss function.
        :param gt_targets: Indicates whether the targets are an inequality regression target of the form ">x".
        :param lt_targets: Indicates whether the targets are an inequality regression target of the form "<x".
        :param features: A numpy array containing additional features (e.g., Morgan fingerprint).
        :param features_generator: A list of features generators to use.
        :param features_generator_precomputed: Whether ``features`` already includes all requested generated features.
        :param selected_features_path: Path to a CSV containing selected descriptor names.
        :param selected_feature_columns: A preloaded mapping from generator names to selected descriptor names.
        :param phase_features: A one-hot vector indicating the phase of the data, as used in spectra data.
        :param atom_descriptors: A numpy array containing additional atom descriptors to featurize the molecule.
        :param bond_descriptors: A numpy array containing additional bond descriptors to featurize the molecule.
        :param raw_constraints: A numpy array containing all user-provided atom/bond-level constraints in input data.
        :param constraints: A numpy array containing atom/bond-level constraints that are used in training. Param constraints is a subset of param raw_constraints.
        :param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features.
        :param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features.

        """
        self.smiles = smiles
        self.targets = targets
        self.atom_targets = atom_targets
        self.bond_targets = bond_targets
        self.row = row
        self.features = features
        self.features_generator = features_generator
        self.selected_features_path = selected_features_path
        self.selected_feature_columns = (
            dict(selected_feature_columns)
            if selected_feature_columns is not None
            else load_selected_feature_columns(selected_features_path)
            if selected_features_path is not None
            else {}
        )
        self.phase_features = phase_features
        self.atom_descriptors = atom_descriptors
        self.bond_descriptors = bond_descriptors
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.constraints = constraints
        self.raw_constraints = raw_constraints
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features
        self.is_mol_list = [is_mol(s) for s in smiles]
        self.is_reaction_list = [is_reaction(x) for x in self.is_mol_list]
        self.is_explicit_h_list = [is_explicit_h(x) for x in self.is_mol_list]
        self.is_adding_hs_list = [is_adding_hs(x) for x in self.is_mol_list]
        self.is_keeping_atom_map_list = [is_keeping_atom_map(x) for x in self.is_mol_list]

        if data_weight is not None:
            self.data_weight = data_weight
        if gt_targets is not None:
            self.gt_targets = gt_targets
        if lt_targets is not None:
            self.lt_targets = lt_targets

        # Generate additional features if given a generator
        if self.features_generator is not None and not features_generator_precomputed:
            if self.features is None:
                self.features = []
            else:
                self.features = list(self.features)

            for fg in self.features_generator:
                selected_feature_columns = self.selected_feature_columns.get(fg)
                for m, reaction in zip(self.mol, self.is_reaction_list):
                    if not reaction:
                        if m is not None and m.GetNumHeavyAtoms() > 0:
                            self.features.extend(_generate_features(fg, m, selected_feature_columns))
                        # for H2
                        elif m is not None and m.GetNumHeavyAtoms() == 0:
                            # not all features are equally long, so use methane as dummy molecule to determine length
                            self.features.extend(np.zeros(len(_generate_features(
                                fg, Chem.MolFromSmiles('C'), selected_feature_columns
                            ))))
                    else:
                        if m[0] is not None and m[1] is not None and m[0].GetNumHeavyAtoms() > 0:
                            self.features.extend(_generate_features(fg, m[0], selected_feature_columns))
                        elif m[0] is not None and m[1] is not None and m[0].GetNumHeavyAtoms() == 0:
                            self.features.extend(np.zeros(len(_generate_features(
                                fg, Chem.MolFromSmiles('C'), selected_feature_columns
                            ))))
                    

            self.features = np.array(self.features)

        # Validate user/generated feature arrays before they reach scaling or
        # a neural-network layer. NaNs retain the historical zero replacement,
        # while infinities are rejected because they irreversibly poison model
        # activations and fitted scalers.
        if self.features is not None:
            self.features = _sanitize_feature_array(
                self.features, 'Molecular features', expected_ndim=1,
            )

        if self.atom_descriptors is not None:
            self.atom_descriptors = _sanitize_feature_array(
                self.atom_descriptors, 'Atom descriptors', expected_ndim=2,
            )

        if self.atom_features is not None:
            self.atom_features = _sanitize_feature_array(
                self.atom_features, 'Atom features', expected_ndim=2,
            )

        if self.bond_descriptors is not None:
            self.bond_descriptors = _sanitize_feature_array(
                self.bond_descriptors, 'Bond descriptors', expected_ndim=2,
            )

        if self.bond_features is not None:
            self.bond_features = _sanitize_feature_array(
                self.bond_features, 'Bond features', expected_ndim=2,
            )

        # Save a copy of the raw features and targets to enable different scaling later on
        self.raw_features, self.raw_targets, self.raw_atom_targets, self.raw_bond_targets = \
            self.features, self.targets, self.atom_targets, self.bond_targets
        self.raw_atom_descriptors, self.raw_atom_features, self.raw_bond_descriptors, self.raw_bond_features = \
            self.atom_descriptors, self.atom_features, self.bond_descriptors, self.bond_features

    @property
    def mol(self) -> List[Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]]:
        """Gets the corresponding list of RDKit molecules for the corresponding SMILES list."""
        if self.atom_targets is not None or self.bond_targets is not None:
            # When the original atom mapping is used, the explicit hydrogens specified in the input SMILES should be used
            # However, the explicit Hs can only be added for reactions with `--explicit_h` flag
            # To fix this, the attribute of `keep_h_list` in make_mols() is set to match the `keep_atom_map_list`
            mol = make_mols(smiles=self.smiles,
                            reaction_list=self.is_reaction_list,
                            keep_h_list=self.is_keeping_atom_map_list,
                            add_h_list=self.is_adding_hs_list,
                            keep_atom_map_list=self.is_keeping_atom_map_list)
        else:
            mol = make_mols(smiles=self.smiles,
                            reaction_list=self.is_reaction_list,
                            keep_h_list=self.is_explicit_h_list,
                            add_h_list=self.is_adding_hs_list,
                            keep_atom_map_list=self.is_keeping_atom_map_list)
        return mol

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in the :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return len(self.smiles)

    @property
    def number_of_atoms(self) -> int:
        """
        Gets the number of atoms in the :class:`MoleculeDatapoint`.

        :return: A list of number of atoms for each molecule.
        """
        return [len(self.mol[i].GetAtoms()) for i in range(self.number_of_molecules)]

    @property
    def number_of_bonds(self) -> List[int]:
        """
        Gets the number of bonds in the :class:`MoleculeDatapoint`.

        :return: A list of number of bonds for each molecule.
        """
        return [len(self.mol[i].GetBonds()) for i in range(self.number_of_molecules)]

    @property
    def bond_types(self) -> List[List[float]]:
        """
        Gets the bond types in the :class:`MoleculeDatapoint`.

        :return: A list of bond types for each molecule.
        """
        return [[b.GetBondTypeAsDouble() for b in self.mol[i].GetBonds()] for i in range(self.number_of_molecules)]
    @property
    def max_molwt(self) -> float:
        """
        Gets the maximum molecular weight among all the molecules in the :class:`MoleculeDatapoint`.

        :return: The maximum molecular weight.
        """
        return max(Chem.rdMolDescriptors.CalcExactMolWt(mol) for mol in self.mol)

    def set_features(self, features: np.ndarray) -> None:
        """
        Sets the features of the molecule.

        :param features: A 1D numpy array of features for the molecule.
        """
        self.features = _sanitize_feature_array(
            features, 'Molecular features', expected_ndim=1,
        )

    def set_atom_descriptors(self, atom_descriptors: np.ndarray) -> None:
        """
        Sets the atom descriptors of the molecule.

        :param atom_descriptors: A 1D numpy array of atom descriptors for the molecule.
        """
        self.atom_descriptors = _sanitize_feature_array(
            atom_descriptors, 'Atom descriptors', expected_ndim=2,
        )

    def set_atom_features(self, atom_features: np.ndarray) -> None:
        """
        Sets the atom features of the molecule.

        :param atom_features: A 1D numpy array of atom features for the molecule.
        """
        self.atom_features = _sanitize_feature_array(
            atom_features, 'Atom features', expected_ndim=2,
        )

    def set_bond_descriptors(self, bond_descriptors: np.ndarray) -> None:
        """
        Sets the atom descriptors of the molecule.

        :param bond_descriptors: A 1D numpy array of bond descriptors for the molecule.
        """
        self.bond_descriptors = _sanitize_feature_array(
            bond_descriptors, 'Bond descriptors', expected_ndim=2,
        )

    def set_bond_features(self, bond_features: np.ndarray) -> None:
        """
        Sets the bond features of the molecule.

        :param bond_features: A 1D numpy array of bond features for the molecule.
        """
        self.bond_features = _sanitize_feature_array(
            bond_features, 'Bond features', expected_ndim=2,
        )

    def extend_features(self, features: np.ndarray) -> None:
        """
        Extends the features of the molecule.

        :param features: A 1D numpy array of extra features for the molecule.
        """
        combined = (
            np.append(self.features, features)
            if self.features is not None
            else features
        )
        self.features = _sanitize_feature_array(
            combined, 'Molecular features', expected_ndim=1,
        )

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        self.features, self.targets, self.atom_targets, self.bond_targets = \
            self.raw_features, self.raw_targets, self.raw_atom_targets, self.raw_bond_targets
        self.atom_descriptors, self.atom_features, self.bond_descriptors, self.bond_features = \
            self.raw_atom_descriptors, self.raw_atom_features, self.raw_bond_descriptors, self.raw_bond_features


class MoleculeDataset(Dataset):
    r"""A :class:`MoleculeDataset` contains a list of :class:`MoleculeDatapoint`\ s with access to their attributes."""

    def __init__(self, data: List[MoleculeDatapoint]):
        r"""
        :param data: A list of :class:`MoleculeDatapoint`\ s.
        """
        self._data = data
        self._batch_graph = None
        self._random = Random()

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Returns a list containing the SMILES list associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned SMILES to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of SMILES, depending on :code:`flatten`.
        """
        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]

        return [d.smiles for d in self._data]

    def mols(self, flatten: bool = False) -> Union[List[Chem.Mol], List[List[Chem.Mol]], List[Tuple[Chem.Mol, Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]]]:
        """
        Returns a list of the RDKit molecules associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned RDKit molecules to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of RDKit molecules, depending on :code:`flatten`.
        """
        if flatten:
            return [mol for d in self._data for mol in d.mol]

        return [d.mol for d in self._data]

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in each :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    @property
    def number_of_atoms(self) -> List[List[int]]:
        """
        Gets the number of atoms in each :class:`MoleculeDatapoint`.

        :return: A list of number of atoms for each molecule.
        """
        return [d.number_of_atoms for d in self._data]

    @property
    def number_of_bonds(self) -> List[List[int]]:
        """
        Gets the number of bonds in each :class:`MoleculeDatapoint`.

        :return: A list of number of bonds for each molecule.
        """
        return [d.number_of_bonds for d in self._data]

    @property
    def bond_types(self) -> List[List[float]]:
        """
        Gets the bond types in each :class:`MoleculeDatapoint`.

        :return: A list of bond types for each molecule.
        """
        return [d.bond_types for d in self._data]

    @property
    def is_atom_bond_targets(self) -> bool:
        """
        Gets the Boolean whether this is atomic/bond properties prediction.

        :return: A Boolean value.
        """
        if not self._data:
            return False
        if self._data[0].atom_targets is None and self._data[0].bond_targets is None:
            return False
        else:
            return True

    def batch_graph(self) -> List[BatchMolGraph]:
        r"""
        Constructs a :class:`~chemprop.features.BatchMolGraph` with the graph featurization of all the molecules.

        .. note::
           The :class:`~chemprop.features.BatchMolGraph` is cached in after the first time it is computed
           and is simply accessed upon subsequent calls to :meth:`batch_graph`. This means that if the underlying
           set of :class:`MoleculeDatapoint`\ s changes, then the returned :class:`~chemprop.features.BatchMolGraph`
           will be incorrect for the underlying data.

        :return: A list of :class:`~chemprop.features.BatchMolGraph` containing the graph featurization of all the
                 molecules in each :class:`MoleculeDatapoint`.
        """
        if self._batch_graph is None:
            self._batch_graph = []
            if not self._data:
                return self._batch_graph

            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                for mol_index, (s, m) in enumerate(zip(d.smiles, d.mol)):
                    has_row_features = d.atom_features is not None or d.bond_features is not None
                    keep_h = (
                        d.is_keeping_atom_map_list[mol_index]
                        if d.atom_targets is not None or d.bond_targets is not None
                        else d.is_explicit_h_list[mol_index]
                    )
                    graph_key = (
                        s,
                        d.is_reaction_list[mol_index],
                        keep_h,
                        d.is_adding_hs_list[mol_index],
                        d.is_keeping_atom_map_list[mol_index],
                        reaction_mode(),
                    )
                    if cache_graph() and not has_row_features and graph_key in SMILES_TO_GRAPH:
                        mol_graph = SMILES_TO_GRAPH[graph_key]
                    else:
                        if len(d.smiles) > 1 and (d.atom_features is not None or d.bond_features is not None):
                            raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                                      'per input (i.e., number_of_molecules = 1).')

                        mol_graph = MolGraph(m, d.atom_features, d.bond_features,
                                             overwrite_default_atom_features=d.overwrite_default_atom_features,
                                             overwrite_default_bond_features=d.overwrite_default_bond_features)
                        if cache_graph() and not has_row_features:
                            SMILES_TO_GRAPH[graph_key] = mol_graph
                    mol_graphs_list.append(mol_graph)
                mol_graphs.append(mol_graphs_list)

            self._batch_graph = [BatchMolGraph([g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]

        return self._batch_graph

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        return [d.features for d in self._data]

    def phase_features(self) -> List[np.ndarray]:
        """
        Returns the phase features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the phase features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].phase_features is None:
            return None

        return [d.phase_features for d in self._data]

    def atom_features(self) -> List[np.ndarray]:
        """
        Returns the atom descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].atom_features is None:
            return None

        return [d.atom_features for d in self._data]

    def atom_descriptors(self) -> List[np.ndarray]:
        """
        Returns the atom descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].atom_descriptors is None:
            return None

        return [d.atom_descriptors for d in self._data]

    def bond_features(self) -> List[np.ndarray]:
        """
        Returns the bond features associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the bond features
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].bond_features is None:
            return None

        return [d.bond_features for d in self._data]

    def bond_descriptors(self) -> List[np.ndarray]:
        """
        Returns the bond descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the bond descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].bond_descriptors is None:
            return None

        return [d.bond_descriptors for d in self._data]

    def constraints(self) -> List[np.ndarray]:
        """
        Return the constraints applied in atomic/bond properties prediction.
        """
        constraints = []
        for d in self._data:
            if d.constraints is None :
                natom_targets = len(d.atom_targets) if d.atom_targets is not None else 0
                nbond_targets = len(d.bond_targets) if d.bond_targets is not None else 0
                ntargets = natom_targets + nbond_targets
                constraints.append([None] * ntargets)
            else:
                constraints.append(d.constraints)
        return constraints

    def data_weights(self) -> List[float]:
        """
        Returns the loss weighting associated with each datapoint.
        """
        if not self._data:
            return []
        if not hasattr(self._data[0], 'data_weight'):
            return [1. for d in self._data]

        return [d.data_weight for d in self._data]

    def atom_bond_data_weights(self) -> List[List[float]]:
        """
        Returns the loss weighting associated with each datapoint for atomic/bond properties prediction.
        """
        targets = self.targets()
        if not targets:
            return []
        data_weights = self.data_weights()
        atom_bond_data_weights = [[] for _ in targets[0]]
        for i, tb in enumerate(targets):
            weight = data_weights[i]
            for j, x in enumerate(tb): 
                atom_bond_data_weights[j] += [1. * weight] * len(x)

        return atom_bond_data_weights

    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.targets for d in self._data]
    
    def mask(self) -> List[List[bool]]:
        """
        Returns whether the targets associated with each molecule and task are present.

        :return: A list of list of booleans associated with targets.
        """
        targets = self.targets()
        if self.is_atom_bond_targets:
            mask = []
            for dt in zip(*targets):
                dt = np.concatenate(dt)
                mask.append([x is not None for x in dt])
        else:
            mask = [[t is not None for t in dt] for dt in targets]
            mask = list(zip(*mask))
        return mask

    def gt_targets(self) -> List[np.ndarray]:
        """
        Returns indications of whether the targets associated with each molecule are greater-than inequalities.
        
        :return: A list of lists of booleans indicating whether the targets in those positions are greater-than inequality targets.
        """
        if not self._data or not hasattr(self._data[0], 'gt_targets'):
            return None

        return [d.gt_targets for d in self._data]

    def lt_targets(self) -> List[np.ndarray]:
        """
        Returns indications of whether the targets associated with each molecule are less-than inequalities.
        
        :return: A list of lists of booleans indicating whether the targets in those positions are less-than inequality targets.
        """
        if not self._data or not hasattr(self._data[0], 'lt_targets'):
            return None

        return [d.lt_targets for d in self._data]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the additional features vector associated with the molecules.

        :return: The size of the additional features vector.
        """
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def atom_descriptors_size(self) -> int:
        """
        Returns the size of custom additional atom descriptors vector associated with the molecules.

        :return: The size of the additional atom descriptor vector.
        """
        return int(self._data[0].atom_descriptors.shape[1]) \
            if len(self._data) > 0 and self._data[0].atom_descriptors is not None else None

    def atom_features_size(self) -> int:
        """
        Returns the size of custom additional atom features vector associated with the molecules.

        :return: The size of the additional atom feature vector.
        """
        return int(self._data[0].atom_features.shape[1]) \
            if len(self._data) > 0 and self._data[0].atom_features is not None else None

    def bond_descriptors_size(self) -> int:
        """
        Returns the size of custom additional bond descriptors vector associated with the molecules.

        :return: The size of the additional bond descriptor vector.
        """
        return int(self._data[0].bond_descriptors.shape[1]) \
            if len(self._data) > 0 and self._data[0].bond_descriptors is not None else None

    def bond_features_size(self) -> int:
        """
        Returns the size of custom additional bond features vector associated with the molecules.

        :return: The size of the additional bond feature vector.
        """
        return int(self._data[0].bond_features.shape[1]) \
            if len(self._data) > 0 and self._data[0].bond_features is not None else None

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0,
                           scale_atom_descriptors: bool = False, scale_bond_descriptors: bool = False) -> StandardScaler:
        """
        Normalizes the features of the dataset using a :class:`~chemprop.data.StandardScaler`.

        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each feature independently.

        If a :class:`~chemprop.data.StandardScaler` is provided, it is used to perform the normalization.
        Otherwise, a :class:`~chemprop.data.StandardScaler` is first fit to the features in this dataset
        and is then used to perform the normalization.

        :param scaler: A fitted :class:`~chemprop.data.StandardScaler`. If it is provided it is used,
                       otherwise a new :class:`~chemprop.data.StandardScaler` is first fitted to this
                       data and is then used.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        :param scale_atom_descriptors: If the features that need to be scaled are atom features rather than molecule.
        :param scale_bond_descriptors: If the features that need to be scaled are bond features rather than molecule.
        :return: A fitted :class:`~chemprop.data.StandardScaler`. If a :class:`~chemprop.data.StandardScaler`
                 is provided as a parameter, this is the same :class:`~chemprop.data.StandardScaler`. Otherwise,
                 this is a new :class:`~chemprop.data.StandardScaler` that has been fit on this dataset.
        """
        if len(self._data) == 0 or \
                (self._data[0].features is None and not scale_bond_descriptors and not scale_atom_descriptors):
            return None

        if scaler is None:
            if scale_atom_descriptors and not self._data[0].atom_descriptors is None:
                features = np.vstack([d.raw_atom_descriptors for d in self._data])
            elif scale_atom_descriptors and not self._data[0].atom_features is None:
                features = np.vstack([d.raw_atom_features for d in self._data])
            elif scale_bond_descriptors and not self._data[0].bond_descriptors is None:
                features = np.vstack([d.raw_bond_descriptors for d in self._data])
            elif scale_bond_descriptors and not self._data[0].bond_features is None:
                features = np.vstack([d.raw_bond_features for d in self._data])
            else:
                features = np.vstack([d.raw_features for d in self._data])
            scaler = StandardScaler(replace_nan_token=replace_nan_token)
            scaler.fit(features)

        if scale_atom_descriptors and not self._data[0].atom_descriptors is None:
            for d in self._data:
                d.set_atom_descriptors(scaler.transform(d.raw_atom_descriptors))
        elif scale_atom_descriptors and not self._data[0].atom_features is None:
            for d in self._data:
                d.set_atom_features(scaler.transform(d.raw_atom_features))
        elif scale_bond_descriptors and not self._data[0].bond_descriptors is None:
            for d in self._data:
                d.set_bond_descriptors(scaler.transform(d.raw_bond_descriptors))
        elif scale_bond_descriptors and not self._data[0].bond_features is None:
            for d in self._data:
                d.set_bond_features(scaler.transform(d.raw_bond_features))
        else:
            for d in self._data:
                d.set_features(scaler.transform(d.raw_features.reshape(1, -1))[0])

        return scaler

    def normalize_targets(self) -> StandardScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.StandardScaler`.
        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each task independently.
        This should only be used for regression datasets.
        :return: A :class:`~chemprop.data.StandardScaler` fitted to the targets.
        """
        targets = [d.raw_targets for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)

        return scaler

    def normalize_atom_bond_targets(self) -> AtomBondScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.AtomBondScaler`.

        The :class:`~chemprop.data.AtomBondScaler` subtracts the mean and divides by the standard deviation
        for each task independently.

        This should only be used for regression datasets.

        :return: A :class:`~chemprop.data.AtomBondScaler` fitted to the targets.
        """
        atom_targets = self._data[0].atom_targets
        bond_targets = self._data[0].bond_targets
        n_atom_targets = len(atom_targets) if atom_targets is not None else 0
        n_bond_targets = len(bond_targets) if bond_targets is not None else 0
        n_atoms, n_bonds = self.number_of_atoms, self.number_of_bonds

        targets = [d.raw_targets for d in self._data]
        targets = [np.concatenate(x).reshape([-1, 1]) for x in zip(*targets)]
        scaler = AtomBondScaler(
            n_atom_targets=n_atom_targets,
            n_bond_targets=n_bond_targets,
        ).fit(targets)
        scaled_targets = scaler.transform(targets)
        for i in range(n_atom_targets):
            scaled_targets[i] = np.split(np.array(scaled_targets[i]).flatten(), np.cumsum(np.array(n_atoms)))[:-1]
        for i in range(n_bond_targets):
            scaled_targets[i+n_atom_targets] = np.split(np.array(scaled_targets[i+n_atom_targets]).flatten(), np.cumsum(np.array(n_bonds)))[:-1]
        scaled_targets = np.array(scaled_targets, dtype=object).T
        self.set_targets(scaled_targets)

        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        """
        if not len(self._data) == len(targets):
            raise ValueError(
                "number of molecules and targets must be of same length! "
                f"num molecules: {len(self._data)}, num targets: {len(targets)}"
            )
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).

        :return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[item]


class MoleculeSampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = Random(seed)

        if self.class_balance:
            self.positive_indices = []
            self.negative_indices = []
            for index, datapoint in enumerate(dataset):
                if datapoint.targets is None or len(datapoint.targets) != 1:
                    raise ValueError(
                        'Class-balanced sampling supports only single-task '
                        'binary targets.'
                    )
                target = datapoint.targets[0]
                if target is None:
                    # Missing labels are neither negative nor positive and
                    # must not influence the balancing ratio.
                    continue
                target_array = np.asarray(target)
                if target_array.ndim != 0 or target_array.item() not in (0, 1):
                    raise ValueError(
                        'Class-balanced sampling requires observed targets to '
                        'be binary values 0 or 1.'
                    )
                if target_array.item() == 1:
                    self.positive_indices.append(index)
                else:
                    self.negative_indices.append(index)

            if not self.positive_indices or not self.negative_indices:
                raise ValueError(
                    'Class-balanced sampling requires at least one observed '
                    'target from each binary class.'
                )

            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length


def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:
    r"""
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\ s.

    Additionally, precomputes the :class:`~chemprop.features.BatchMolGraph` for the constructed
    :class:`MoleculeDataset`.

    :param data: A list of :class:`MoleculeDatapoint`\ s.
    :return: A :class:`MoleculeDataset` containing all the :class:`MoleculeDatapoint`\ s.
    """
    data = MoleculeDataset(data)
    data.batch_graph()  # Forces computation and caching of the BatchMolGraph for the molecules

    return data


class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Class balance is only available for single task
                              classification datasets. Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].targets for index in self._sampler]

    @property
    def gt_targets(self) -> List[List[Optional[bool]]]:
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each molecule.

        :return: A list of lists of booleans (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')
        
        if not self._dataset or not hasattr(self._dataset[0], 'gt_targets'):
            return None

        return [self._dataset[index].gt_targets for index in self._sampler]

    @property
    def lt_targets(self) -> List[List[Optional[bool]]]:
        """
        Returns booleans for whether each target is an inequality rather than a value target, associated with each molecule.

        :return: A list of lists of booleans (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        if not self._dataset or not hasattr(self._dataset[0], 'lt_targets'):
            return None

        return [self._dataset[index].lt_targets for index in self._sampler]


    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()

    
def make_mols(smiles: List[str], reaction_list: List[bool], keep_h_list: List[bool], add_h_list: List[bool], keep_atom_map_list: List[bool]):
    """
    Builds a list of RDKit molecules (or a list of tuples of molecules if reaction is True) for a list of smiles.

    :param smiles: List of SMILES strings.
    :param reaction_list: List of booleans whether the SMILES strings are to be treated as a reaction.
    :param keep_h_list: List of booleans whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h_list: List of booleasn whether to add hydrogens to the input smiles.
    :param keep_atom_map_list: List of booleasn whether to keep the original atom mapping.
    :return: List of RDKit molecules or list of tuple of molecules.
    """
    mol = []
    for s, reaction, keep_h, add_h, keep_atom_map in zip(smiles, reaction_list, keep_h_list, add_h_list, keep_atom_map_list):
        cache_key = (s, reaction, keep_h, add_h, keep_atom_map)
        if cache_mol() and cache_key in SMILES_TO_MOL:
            parsed_mol = SMILES_TO_MOL[cache_key]
        elif reaction:
            parsed_mol = (
                make_mol(s.split(">")[0], keep_h, add_h, keep_atom_map),
                make_mol(s.split(">")[-1], keep_h, add_h, keep_atom_map),
            )
        else:
            parsed_mol = make_mol(s, keep_h, add_h, keep_atom_map)

        if cache_mol():
            SMILES_TO_MOL[cache_key] = parsed_mol
        mol.append(parsed_mol)
    return mol
