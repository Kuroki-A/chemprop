"""Molecule-level feature generators.

The registry is imported by every Chemprop command to populate the
``--features_generator`` choices. Keep imports lightweight: large optional
packages are imported only when their generator is used.
"""

import functools
import hashlib
import inspect
import itertools
import os
import warnings
from collections import defaultdict
from threading import Lock
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from rdkit import Chem, DataStructs, rdBase
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors


Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]
BatchFeaturesGenerator = Callable[[Sequence[Molecule]], Sequence[np.ndarray]]

FEATURES_GENERATOR_REGISTRY: Dict[str, FeaturesGenerator] = {}


def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    """Registers a feature generator under its command-line name."""
    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator
    return decorator


def get_features_generator(features_generator_name: str) -> FeaturesGenerator:
    """Gets a registered feature generator by name."""
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(
            f'Features generator "{features_generator_name}" could not be found. '
            "If it uses an optional dependency, install the corresponding extra."
        )
    return FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
    """Returns registered feature-generator names in registration order."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())


def _as_mol(mol: Molecule) -> Chem.Mol:
    if isinstance(mol, str):
        parsed = Chem.MolFromSmiles(mol)
        if parsed is None:
            raise ValueError(f"Could not parse SMILES: {mol!r}")
        return parsed
    if mol is None:
        raise ValueError("Cannot generate features for a null molecule.")
    return mol


def _as_smiles(mol: Molecule) -> str:
    return mol if isinstance(mol, str) else Chem.MolToSmiles(_as_mol(mol), isomericSmiles=True)


def _selection_tuple(selected_feature_columns: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    if selected_feature_columns is None:
        return None
    return tuple(str(column) for column in selected_feature_columns)


def _validate_columns(selected: Optional[Tuple[str, ...]], available: Tuple[str, ...]) -> Tuple[str, ...]:
    if selected is None:
        return available
    missing = [column for column in selected if column not in available]
    if missing:
        raise KeyError(f"Feature columns not found: {missing}")
    return selected


# RDKit fingerprints ---------------------------------------------------------

MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048

_RDKIT_FP_GENERATOR_CACHE: Dict[Tuple[object, ...], object] = {}
_RDKIT_FP_GENERATOR_LOCK = Lock()


def _get_rdkit_fp_generator(kind: str, **kwargs):
    key = (kind, tuple(sorted(kwargs.items())))
    generator = _RDKIT_FP_GENERATOR_CACHE.get(key)
    if generator is not None:
        return generator
    factories = {
        "morgan": rdFingerprintGenerator.GetMorganGenerator,
        "rdkit": rdFingerprintGenerator.GetRDKitFPGenerator,
        "atompair": rdFingerprintGenerator.GetAtomPairGenerator,
    }
    with _RDKIT_FP_GENERATOR_LOCK:
        generator = _RDKIT_FP_GENERATOR_CACHE.get(key)
        if generator is None:
            generator = factories[kind](**kwargs)
            _RDKIT_FP_GENERATOR_CACHE[key] = generator
    return generator


def _bit_vector_to_numpy(bit_vector, dtype: np.dtype) -> np.ndarray:
    array = np.empty(bit_vector.GetNumBits(), dtype=dtype)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array


@register_features_generator("morgan")
def morgan_binary_features_generator(
    mol: Molecule, radius: int = MORGAN_RADIUS, num_bits: int = MORGAN_NUM_BITS,
    selected_feature_columns: list = None,
) -> np.ndarray:
    """Generates a binary Morgan fingerprint (legacy float64 output)."""
    generator = _get_rdkit_fp_generator("morgan", radius=radius, fpSize=num_bits)
    return generator.GetFingerprintAsNumPy(_as_mol(mol)).astype(float, copy=False)


@register_features_generator("morgan_count")
def morgan_counts_features_generator(
    mol: Molecule, radius: int = MORGAN_RADIUS, num_bits: int = MORGAN_NUM_BITS,
    selected_feature_columns: list = None,
) -> np.ndarray:
    """Generates a count Morgan fingerprint (legacy float64 output)."""
    generator = _get_rdkit_fp_generator("morgan", radius=radius, fpSize=num_bits)
    return generator.GetCountFingerprintAsNumPy(_as_mol(mol)).astype(float, copy=False)


@register_features_generator("maccs")
def maccs_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _bit_vector_to_numpy(AllChem.GetMACCSKeysFingerprint(_as_mol(mol)), np.int64)


@register_features_generator("rdkit")
def rdkit_fingerprint_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    generator = _get_rdkit_fp_generator("rdkit", fpSize=2048)
    return generator.GetFingerprintAsNumPy(_as_mol(mol)).astype(int, copy=False)


@register_features_generator("avalon")
def avalon_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _bit_vector_to_numpy(pyAvalonTools.GetAvalonFP(_as_mol(mol)), np.int64)


@register_features_generator("atompair")
def atom_pair_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    generator = _get_rdkit_fp_generator("atompair", fpSize=2048)
    return generator.GetFingerprintAsNumPy(_as_mol(mol)).astype(int, copy=False)


@register_features_generator("erg")
def erg_legacy_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    """Historical integer-cast ErG vector; retained for checkpoint compatibility."""
    return np.asarray(AllChem.GetErGFingerprint(_as_mol(mol)), dtype=int)


@register_features_generator("erg_float")
def erg_float_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    """Full-precision 315-dimensional ErG fingerprint for new models."""
    return np.asarray(AllChem.GetErGFingerprint(_as_mol(mol)), dtype=float)


# Descriptastorus RDKit descriptors -----------------------------------------

_DESCRIPTASTORUS_GENERATOR_CACHE: Dict[Tuple[bool, Tuple[str, ...]], object] = {}
_DESCRIPTASTORUS_GENERATOR_LOCK = Lock()
_DESCRIPTASTORUS_COLUMNS: Optional[Tuple[str, ...]] = None


def _load_descriptastorus():
    try:
        from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
    except ImportError as exc:
        raise ImportError(
            "This generator requires descriptastorus. Install it with `pip install descriptastorus`."
        ) from exc
    return rdDescriptors, rdNormalizedDescriptors


def _get_descriptastorus_columns() -> Tuple[str, ...]:
    global _DESCRIPTASTORUS_COLUMNS
    if _DESCRIPTASTORUS_COLUMNS is None:
        rd_descriptors, _ = _load_descriptastorus()
        columns = tuple(column[0] for column in rd_descriptors.RDKit2D().columns)
        with _DESCRIPTASTORUS_GENERATOR_LOCK:
            if _DESCRIPTASTORUS_COLUMNS is None:
                _DESCRIPTASTORUS_COLUMNS = columns
    return _DESCRIPTASTORUS_COLUMNS


def _get_descriptastorus_generator(normalized: bool, columns: Tuple[str, ...]):
    key = (normalized, columns)
    generator = _DESCRIPTASTORUS_GENERATOR_CACHE.get(key)
    if generator is not None:
        return generator
    rd_descriptors, rd_normalized_descriptors = _load_descriptastorus()
    generator_cls = rd_normalized_descriptors.RDKit2DNormalized if normalized else rd_descriptors.RDKit2D
    with _DESCRIPTASTORUS_GENERATOR_LOCK:
        generator = _DESCRIPTASTORUS_GENERATOR_CACHE.get(key)
        if generator is None:
            generator = generator_cls(properties=list(columns))
            _DESCRIPTASTORUS_GENERATOR_CACHE[key] = generator
    return generator


def _descriptastorus_features(
    mol: Molecule, normalized: bool, without_fragments: bool,
    selected_feature_columns: Optional[Sequence[str]],
) -> np.ndarray:
    all_columns = _get_descriptastorus_columns()
    available = tuple(c for c in all_columns if "fr_" not in c) if without_fragments else all_columns
    columns = _validate_columns(_selection_tuple(selected_feature_columns), available)
    if not columns:
        return np.empty(0, dtype=float)
    generator = _get_descriptastorus_generator(normalized, columns)
    smiles = _as_smiles(mol)
    processed = generator.process(smiles)
    if processed is None:
        raise ValueError(f"Descriptastorus could not process molecule {smiles!r}.")
    return np.asarray(processed[1:], dtype=float)


def _descriptastorus_batch_features(
    mols: Sequence[Molecule],
    normalized: bool,
    without_fragments: bool,
    selected_feature_columns: Optional[Sequence[str]] = None,
) -> Sequence[np.ndarray]:
    if not mols:
        return []
    all_columns = _get_descriptastorus_columns()
    available = tuple(c for c in all_columns if "fr_" not in c) if without_fragments else all_columns
    columns = _validate_columns(_selection_tuple(selected_feature_columns), available)
    if not columns:
        return [np.empty(0, dtype=float) for _ in mols]
    generator = _get_descriptastorus_generator(normalized, columns)
    smiles = [_as_smiles(mol) for mol in mols]
    if normalized:
        # Descriptastorus normally invokes SciPy's scalar CDF once per
        # molecule and descriptor.  Descriptor evaluation is still molecule
        # based, but applying each CDF to a complete column removes thousands
        # of Python→SciPy calls for a dataset-sized batch.
        rd_descriptors, rd_normalized_descriptors = _load_descriptastorus()
        rd_mols = [_as_mol(mol) for mol in mols]
        normalized_matrix = np.zeros((len(mols), len(columns)), dtype=float)
        for column_index, column in enumerate(columns):
            raw_values = [
                rd_descriptors.applyFunc(generator.funcs, column, rd_mol)
                for rd_mol in rd_mols
            ]
            valid_indices = [
                index for index, value in enumerate(raw_values) if value is not None
            ]
            cdf = rd_normalized_descriptors.cdfs.get(column)
            if cdf is not None and valid_indices:
                values = np.asarray(
                    [raw_values[index] for index in valid_indices], dtype=float,
                )
                try:
                    normalized_values = cdf(values)
                except Exception:
                    # Match descriptastorus' scalar failure semantics: one bad
                    # descriptor becomes zero without discarding the batch.
                    normalized_values = np.zeros(len(values), dtype=float)
                    for value_index, value in enumerate(values):
                        try:
                            normalized_values[value_index] = cdf(value)
                        except Exception:
                            pass
                normalized_matrix[valid_indices, column_index] = normalized_values
        return normalized_matrix

    processed_rows = generator.processSmiles(smiles, keep_mols=False)
    output = []
    for index, processed in enumerate(processed_rows):
        if processed is None:
            raise ValueError(f"Descriptastorus could not process batch row {index}: {smiles[index]!r}.")
        output.append(np.asarray(processed[1:], dtype=float))
    return output


@register_features_generator("rdkit_2d")
def rdkit_2d_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _descriptastorus_features(mol, False, False, selected_feature_columns)


@register_features_generator("rdkit_2d_normalized")
def rdkit_2d_normalized_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _descriptastorus_features(mol, True, False, selected_feature_columns)


@register_features_generator("rdkit_2d_wo_fr")
def rdkit_2d_without_fragments_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _descriptastorus_features(mol, False, True, selected_feature_columns)


@register_features_generator("rdkit_2d_normalized_wo_fr")
def rdkit_2d_normalized_without_fragments_features_generator(
    mol: Molecule, selected_feature_columns: list = None,
) -> np.ndarray:
    return _descriptastorus_features(mol, True, True, selected_feature_columns)


for _generator, _normalized, _without_fragments in (
    (rdkit_2d_features_generator, False, False),
    (rdkit_2d_normalized_features_generator, True, False),
    (rdkit_2d_without_fragments_features_generator, False, True),
    (rdkit_2d_normalized_without_fragments_features_generator, True, True),
):
    _generator.batch_transform = functools.partial(
        _descriptastorus_batch_features,
        normalized=_normalized,
        without_fragments=_without_fragments,
    )
    _generator.batch_supports_selected_columns = True
    _generator.preferred_batch_size = 512
    _generator.parallel_safe = False


# Native RDKit descriptor collections --------------------------------------

_MOLECULAR_DESCRIPTOR_CALCULATOR_CACHE: Dict[Tuple[str, ...], object] = {}
_MOLECULAR_DESCRIPTOR_CALCULATOR_LOCK = Lock()
_RDKIT_PROPS: Optional[Tuple[str, ...]] = None
_ALL_DESCRIPTOR_FUNCTIONS: Tuple[Tuple[str, Callable], ...] = tuple(
    (name, function)
    for name, function in inspect.getmembers(Descriptors, inspect.isfunction)
    if not name.startswith("_") and name not in {"setupAUTOCorrDescriptors", "CalcMolDescriptors"}
)


def _get_rdkit_props() -> Tuple[str, ...]:
    global _RDKIT_PROPS
    if _RDKIT_PROPS is None:
        try:
            from descriptastorus.descriptors.rdDescriptors import RDKIT_PROPS
        except ImportError as exc:
            raise ImportError(
                "The rdkit_2d_208/400 generators require descriptastorus for their stable schema."
            ) from exc
        props = tuple(RDKIT_PROPS["1.0.0"])
        with _MOLECULAR_DESCRIPTOR_CALCULATOR_LOCK:
            if _RDKIT_PROPS is None:
                _RDKIT_PROPS = props
    return _RDKIT_PROPS


def _get_molecular_descriptor_calculator(columns: Tuple[str, ...]):
    calculator = _MOLECULAR_DESCRIPTOR_CALCULATOR_CACHE.get(columns)
    if calculator is not None:
        return calculator
    with _MOLECULAR_DESCRIPTOR_CALCULATOR_LOCK:
        calculator = _MOLECULAR_DESCRIPTOR_CALCULATOR_CACHE.get(columns)
        if calculator is None:
            calculator = MoleculeDescriptors.MolecularDescriptorCalculator(list(columns))
            _MOLECULAR_DESCRIPTOR_CALCULATOR_CACHE[columns] = calculator
    return calculator


def _rdkit_208_columns() -> Tuple[str, ...]:
    props = set(_get_rdkit_props())
    return tuple(
        name for name, _ in Descriptors._descList
        if name and (name.startswith("BCUT2D_") or name in props)
    )


@register_features_generator("rdkit_2d_208")
def rdkit_2d_208_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    available = _rdkit_208_columns()
    columns = _validate_columns(_selection_tuple(selected_feature_columns), available)
    if not columns:
        return np.empty(0, dtype=float)
    return np.asarray(_get_molecular_descriptor_calculator(columns).CalcDescriptors(_as_mol(mol)), dtype=float)


def _native_descriptor_columns(kind: str) -> Tuple[Tuple[str, Callable], ...]:
    if kind == "all":
        return _ALL_DESCRIPTOR_FUNCTIONS
    if kind == "autocorr":
        return tuple(pair for pair in _ALL_DESCRIPTOR_FUNCTIONS if pair[0].startswith("AUTOCORR2D_"))
    if kind == "bcut":
        return tuple(pair for pair in _ALL_DESCRIPTOR_FUNCTIONS if pair[0].startswith("BCUT2D_"))
    if kind == "400":
        props = set(_get_rdkit_props())
        return tuple(
            pair for pair in _ALL_DESCRIPTOR_FUNCTIONS
            if pair[0].startswith("AUTOCORR2D_") or pair[0].startswith("BCUT2D_") or pair[0] in props
        )
    raise ValueError(f"Unknown native descriptor collection: {kind}")


def _native_descriptor_features(
    mol: Molecule, kind: str, selected_feature_columns: Optional[Sequence[str]],
) -> np.ndarray:
    pairs = _native_descriptor_columns(kind)
    available = tuple(name for name, _ in pairs)
    columns = _validate_columns(_selection_tuple(selected_feature_columns), available)
    if not columns:
        return np.empty(0, dtype=float)
    functions = dict(pairs)
    rd_mol = _as_mol(mol)
    return np.asarray([functions[column](rd_mol) for column in columns], dtype=float)


@register_features_generator("rdkit_2d_400")
def rdkit_2d_400_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _native_descriptor_features(mol, "400", selected_feature_columns)


@register_features_generator("rdkit_2d_autocorr")
def rdkit_2d_autocorr_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _native_descriptor_features(mol, "autocorr", selected_feature_columns)


@register_features_generator("rdkit_2d_bcut")
def rdkit_2d_bcut_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _native_descriptor_features(mol, "bcut", selected_feature_columns)


@register_features_generator("rdkit_2d_all")
def rdkit_2d_all_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    """Generates every public descriptor; its schema can change with RDKit."""
    return _native_descriptor_features(mol, "all", selected_feature_columns)


# Mordred and PaDEL (optional, lazy) ----------------------------------------

@functools.lru_cache(maxsize=1)
def _get_mordred_base():
    try:
        from mordred import (
            Calculator, AcidBase, AdjacencyMatrix, Aromatic, AtomCount, BalabanJ,
            BaryszMatrix, BertzCT, BondCount, CarbonTypes, Constitutional,
            DistanceMatrix, EccentricConnectivityIndex, FragmentComplexity,
            Framework, HydrogenBond, InformationContent, KappaShapeIndex, LogS,
            McGowanVolume, MoeType, MolecularId, PathCount, Polarizability,
            RingCount, RotatableBond, SLogP, TopoPSA, TopologicalCharge,
            TopologicalIndex, VertexAdjacencyInformation, WalkCount, Weight,
            WienerIndex, ZagrebIndex,
        )
    except ImportError as exc:
        raise ImportError(
            "The mordred generator requires `pip install mordredcommunity` "
            "(or legacy `mordred`)."
        ) from exc
    modules = (
        AcidBase, AdjacencyMatrix, Aromatic, AtomCount, BalabanJ, BaryszMatrix,
        BertzCT, BondCount, CarbonTypes, Constitutional, DistanceMatrix,
        EccentricConnectivityIndex, FragmentComplexity, Framework, HydrogenBond,
        InformationContent, KappaShapeIndex, LogS, McGowanVolume, MoeType,
        MolecularId, PathCount, Polarizability, RingCount, RotatableBond, SLogP,
        TopoPSA, TopologicalCharge, TopologicalIndex,
        VertexAdjacencyInformation, WalkCount, Weight, WienerIndex, ZagrebIndex,
    )
    calculator = Calculator()
    for descriptor_module in modules:
        calculator.register(descriptor_module)
    names = tuple(str(descriptor) for descriptor in calculator.descriptors)
    return Calculator, calculator, names


@functools.lru_cache(maxsize=64)
def _get_mordred_calculator(selected: Optional[Tuple[str, ...]]):
    Calculator, base_calculator, base_names = _get_mordred_base()
    columns = _validate_columns(selected, base_names)
    if selected is None:
        return base_calculator, tuple(range(len(base_names)))
    if not columns:
        return None, ()
    unique_columns = tuple(dict.fromkeys(columns))
    descriptor_by_name = {str(d): d for d in base_calculator.descriptors}
    calculator = Calculator()
    for column in unique_columns:
        calculator.register(descriptor_by_name[column])
    computed_index = {str(d): i for i, d in enumerate(calculator.descriptors)}
    return calculator, tuple(computed_index[column] for column in columns)


@register_features_generator("mordred")
def mordred_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    calculator, output_indices = _get_mordred_calculator(_selection_tuple(selected_feature_columns))
    if calculator is None:
        return np.empty(0, dtype=float)
    values = list(calculator(_as_mol(mol)))
    return np.asarray([values[index] for index in output_indices], dtype=float)


_PADEL_COLUMNS: Optional[Tuple[str, ...]] = None
_PADEL_COLUMNS_LOCK = Lock()


def _load_padel_from_smiles():
    try:
        from padelpy import from_smiles
    except ImportError as exc:
        raise ImportError("The padelpy generator requires `pip install padelpy` and Java.") from exc
    return from_smiles


def _padel_row_to_array(row: Dict[str, str], selected_feature_columns: list = None) -> np.ndarray:
    global _PADEL_COLUMNS
    row_columns = tuple(row.keys())[:1444]
    with _PADEL_COLUMNS_LOCK:
        if _PADEL_COLUMNS is None:
            _PADEL_COLUMNS = row_columns
    columns = _validate_columns(_selection_tuple(selected_feature_columns), _PADEL_COLUMNS)
    return np.asarray([row[column] for column in columns], dtype=float)


def _padel_failure_zeros(selected_feature_columns: list, smiles: str, exc: Exception) -> np.ndarray:
    if _PADEL_COLUMNS is None:
        raise RuntimeError(f"PaDEL failed for {smiles!r} before its output schema was known: {exc}") from exc
    columns = _validate_columns(_selection_tuple(selected_feature_columns), _PADEL_COLUMNS)
    warnings.warn(
        f"PaDEL failed for {smiles!r}; returning {len(columns)} zeros. Cause: {exc}", RuntimeWarning,
    )
    return np.zeros(len(columns), dtype=float)


@register_features_generator("padelpy")
def padelpy_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    smiles = _as_smiles(mol)
    try:
        return _padel_row_to_array(_load_padel_from_smiles()(smiles), selected_feature_columns)
    except ImportError:
        raise
    except Exception as exc:
        return _padel_failure_zeros(selected_feature_columns, smiles, exc)


def padelpy_batch_features_generator(
    mols: Sequence[Molecule], selected_feature_columns: list = None,
) -> Sequence[np.ndarray]:
    smiles = [_as_smiles(mol) for mol in mols]
    if not smiles:
        return []
    try:
        rows = _load_padel_from_smiles()(smiles)
        if len(rows) != len(smiles):
            raise RuntimeError(f"PaDEL returned {len(rows)} rows for {len(smiles)} molecules")
        return [_padel_row_to_array(row, selected_feature_columns) for row in rows]
    except ImportError:
        raise
    except Exception as exc:
        warnings.warn(
            f"PaDEL batch failed; retrying {len(smiles)} molecules individually. Cause: {exc}", RuntimeWarning,
        )
        return [
            padelpy_features_generator(value, selected_feature_columns)
            for value in smiles
        ]


padelpy_features_generator.batch_transform = padelpy_batch_features_generator
padelpy_features_generator.batch_supports_selected_columns = True
padelpy_features_generator.preferred_batch_size = 64
padelpy_features_generator.parallel_safe = False


# Lazy molfeat MoleculeTransformer generators -------------------------------

_MOLFEAT_TRANSFORMER_CACHE: Dict[Tuple[object, ...], object] = {}
_MOLFEAT_TRANSFORMER_LOCK = Lock()
_MOLFEAT_IMPORT_LOCK = Lock()
_MOLFEAT_OPTIONAL_COMPAT_READY = False

_MAP4_CALCULATOR_CACHE: Dict[Tuple[str, int, int, bool], object] = {}
_MAP4_CALCULATOR_LOCK = Lock()


def _canonical_map4_mol(mol: Molecule) -> Chem.Mol:
    """Returns MAP4's canonical, non-isomeric RDKit representation.

    Both MAP4 implementations encode rooted SMILES.  Reconstructing the
    molecule from a canonical SMILES first makes the resulting fingerprint
    independent of input SMILES traversal and RDKit atom numbering.  All
    fragments are retained so Chemprop's molecule semantics do not silently
    change for salts and other disconnected inputs.
    """
    canonical_smiles = Chem.MolToSmiles(
        _as_mol(mol), canonical=True, isomericSmiles=False,
    )
    canonical_mol = Chem.MolFromSmiles(canonical_smiles)
    if canonical_mol is None:  # Defensive: the source molecule already parsed.
        raise ValueError(
            f"Could not reconstruct canonical MAP4 molecule from {canonical_smiles!r}."
        )
    return canonical_mol


class _LegacyMap4CalculatorAdapter:
    """Implements folded MAP4 v1.0 and Molfeat 0.11's expected API.

    Molfeat 0.11 imports ``MAP4Calculator`` from map4, while map4 1.1
    exports only ``MAP4``.  The two implementations also order equal-radius
    atom environments differently, so merely aliasing the new class changes
    existing feature vectors.  This small, tmap-free implementation preserves
    the folded v1.0 algorithm used by Molfeat while allowing map4 1.1 to remain
    installed for the explicitly named ``map4_v1_1`` generator.
    """

    def __init__(
        self,
        dimensions: int = 2048,
        radius: int = 2,
        is_counted: bool = False,
        is_folded: bool = True,
        return_strings: bool = False,
        **kwargs,
    ):
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected MAP4 v1.0 option(s): {unexpected}")
        if not is_folded or return_strings:
            raise ValueError(
                "Chemprop supports only folded numeric fingerprints through "
                "the Molfeat compatibility adapter."
            )
        try:
            from mhfp.encoder import MHFPEncoder
        except ImportError as exc:
            raise ImportError(
                "The legacy-compatible map4 generator requires the optional "
                "`mhfp` package."
            ) from exc

        self.dimensions = dimensions
        self.radius = radius
        self.is_counted = is_counted
        # MHFPEncoder's default seed (42) is part of the v1.0 definition.  It
        # does not affect folded output, but retaining it avoids a subtle API
        # difference if MHFP changes implementation details in the future.
        self.encoder = MHFPEncoder(dimensions)

    def calculate(self, mol: Chem.Mol) -> np.ndarray:
        canonical_mol = _canonical_map4_mol(mol)
        pairs = self._all_pairs(canonical_mol, self._get_atom_envs(canonical_mol))
        return self.encoder.fold(
            self.encoder.hash(set(pairs)), self.dimensions,
        )

    def calculate_many(
        self,
        mols: Sequence[Chem.Mol],
        number_of_threads: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        # Deliberately stay in-process.  map4 1.1 creates a new process pool
        # for every batch, which is costly and can oversubscribe Chemprop's
        # own feature workers.  Parallelism remains available at the caller.
        del number_of_threads, verbose
        fingerprints = np.empty((len(mols), self.dimensions), dtype=np.uint8)
        for index, mol in enumerate(mols):
            fingerprints[index] = self.calculate(mol)
        return fingerprints

    def _get_atom_envs(self, mol: Chem.Mol) -> Dict[int, List[str]]:
        atom_envs: Dict[int, List[str]] = {}
        for atom in mol.GetAtoms():
            atom_index = atom.GetIdx()
            atom_envs[atom_index] = [
                self._find_env(mol, atom_index, radius)
                for radius in range(1, self.radius + 1)
            ]
        return atom_envs

    @staticmethod
    def _find_env(mol: Chem.Mol, atom_index: int, radius: int) -> str:
        environment = Chem.FindAtomEnvironmentOfRadiusN(
            mol, radius, rootedAtAtom=atom_index,
        )
        atom_map: Dict[int, int] = {}
        submol = Chem.PathToSubmol(mol, environment, atomMap=atom_map)
        if atom_index not in atom_map:
            return ""
        return Chem.MolToSmiles(
            submol,
            rootedAtAtom=atom_map[atom_index],
            canonical=True,
            isomericSmiles=False,
        )

    def _all_pairs(
        self, mol: Chem.Mol, atom_envs: Dict[int, List[str]],
    ) -> List[bytes]:
        atom_pairs: List[bytes] = []
        shingle_counts: Dict[str, int] = defaultdict(int)
        distance_matrix = Chem.GetDistanceMatrix(mol)
        for first, second in itertools.combinations(range(mol.GetNumAtoms()), 2):
            distance = str(int(distance_matrix[first][second]))
            for radius_index in range(self.radius):
                # Lexicographic ordering is the behavior that distinguishes
                # v1.0 from the length-based ordering introduced in v1.1.
                smaller, larger = sorted(
                    [atom_envs[first][radius_index], atom_envs[second][radius_index]],
                )
                shingle = f"{smaller}|{distance}|{larger}"
                if self.is_counted:
                    shingle_counts[shingle] += 1
                    shingle += f"|{shingle_counts[shingle]}"
                atom_pairs.append(shingle.encode("utf-8"))
        return list(set(atom_pairs))


def _get_map4_calculator(
    version: str, dimensions: int = 2048, radius: int = 2,
    include_duplicated_shingles: bool = False,
):
    if version not in {"v1.0", "v1.1"}:
        raise ValueError(f"Unknown MAP4 implementation: {version}")
    key = (version, dimensions, radius, include_duplicated_shingles)
    calculator = _MAP4_CALCULATOR_CACHE.get(key)
    if calculator is not None:
        return calculator

    with _MAP4_CALCULATOR_LOCK:
        calculator = _MAP4_CALCULATOR_CACHE.get(key)
        if calculator is None:
            if version == "v1.0":
                calculator = _LegacyMap4CalculatorAdapter(
                    dimensions=dimensions,
                    radius=radius,
                    is_counted=include_duplicated_shingles,
                    is_folded=True,
                )
            else:
                try:
                    from map4 import MAP4
                except (AttributeError, ImportError) as exc:
                    raise ImportError(
                        "The map4_v1_1 generator requires map4>=1.1."
                    ) from exc
                calculator = MAP4(
                    dimensions=dimensions,
                    radius=radius,
                    include_duplicated_shingles=include_duplicated_shingles,
                )
            _MAP4_CALCULATOR_CACHE[key] = calculator
    return calculator


def _select_map4_features(
    features: np.ndarray,
    selected_feature_columns: Optional[Sequence[str]] = None,
) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    selected = _selection_tuple(selected_feature_columns)
    if selected is None:
        return features
    available = tuple(f"fp_{index}" for index in range(features.shape[-1]))
    columns = _validate_columns(selected, available)
    indices = {column: index for index, column in enumerate(available)}
    return features[..., [indices[column] for column in columns]]


def _prepare_molfeat_optional_dependency_compatibility() -> None:
    """Makes supported modern optional backends importable by Molfeat 0.11."""
    global _MOLFEAT_OPTIONAL_COMPAT_READY

    if _MOLFEAT_OPTIONAL_COMPAT_READY:
        return
    with _MOLFEAT_IMPORT_LOCK:
        if _MOLFEAT_OPTIONAL_COMPAT_READY:
            return
        try:
            import map4 as map4_module
        except ImportError:
            pass
        else:
            if not hasattr(map4_module, "MAP4Calculator") and hasattr(map4_module, "MAP4"):
                map4_module.MAP4Calculator = _LegacyMap4CalculatorAdapter
        _MOLFEAT_OPTIONAL_COMPAT_READY = True


def _get_molfeat_transformer(kind: str, length: int, **params):
    key = (kind, length, tuple(sorted(params.items())))
    transformer = _MOLFEAT_TRANSFORMER_CACHE.get(key)
    if transformer is not None:
        return transformer
    _prepare_molfeat_optional_dependency_compatibility()
    if kind.lower() == "map4":
        try:
            import map4  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The map4 generator requires the optional `map4` package. "
                "Install it together with molfeat before using this generator."
            ) from exc
    try:
        from molfeat.trans.fp import FPVecTransformer
    except ImportError as exc:
        detail = str(exc) or type(exc).__name__
        raise ImportError(
            f"The {kind} generator could not import Molfeat or one of its optional "
            f"dependencies: {detail}. Install compatible versions with "
            "`pip install 'molfeat[all]'`."
        ) from exc
    with _MOLFEAT_TRANSFORMER_LOCK:
        transformer = _MOLFEAT_TRANSFORMER_CACHE.get(key)
        if transformer is None:
            transformer = FPVecTransformer(kind=kind, length=length, n_jobs=1, dtype=float, **params)
            _MOLFEAT_TRANSFORMER_CACHE[key] = transformer
    return transformer


def _molfeat_output_columns(transformer, width: int) -> Tuple[str, ...]:
    columns = getattr(getattr(transformer, "featurizer", None), "columns", None)
    return tuple(str(c) for c in columns) if columns is not None else tuple(f"fp_{i}" for i in range(width))


def _molfeat_features(
    mol: Molecule, kind: str, length: int, selected_feature_columns: list = None, **params,
) -> np.ndarray:
    transformer = _get_molfeat_transformer(kind, length, **params)
    try:
        features = np.asarray(transformer([_as_smiles(mol)]), dtype=float)[0]
    except ImportError as exc:
        raise ImportError(
            f"The {kind} generator is unavailable: {exc}. Install `molfeat[all]` and its optional dependency."
        ) from exc
    selected = _selection_tuple(selected_feature_columns)
    if selected is None:
        return features
    available = _molfeat_output_columns(transformer, len(features))
    columns = _validate_columns(selected, available)
    indices = {column: index for index, column in enumerate(available)}
    return features[[indices[column] for column in columns]]


def _molfeat_batch_features(
    mols: Sequence[Molecule], kind: str, length: int,
    selected_feature_columns: Optional[Sequence[str]] = None, **params,
) -> Sequence[np.ndarray]:
    if not mols:
        return []
    transformer = _get_molfeat_transformer(kind, length, **params)
    try:
        features = np.asarray(transformer([_as_smiles(mol) for mol in mols]), dtype=float)
    except ImportError as exc:
        raise ImportError(
            f"The {kind} generator is unavailable: {exc}. Install `molfeat[all]` and its optional dependency."
        ) from exc
    selected = _selection_tuple(selected_feature_columns)
    if selected is None:
        return features
    available = _molfeat_output_columns(transformer, features.shape[1])
    columns = _validate_columns(selected, available)
    indices = {column: index for index, column in enumerate(available)}
    return features[:, [indices[column] for column in columns]]


def _configure_molfeat_batch(generator: FeaturesGenerator, kind: str, length: int, batch_size: int = 256):
    generator.batch_transform = functools.partial(_molfeat_batch_features, kind=kind, length=length)
    generator.preferred_batch_size = batch_size
    generator.batch_supports_selected_columns = True
    generator.parallel_safe = False


@register_features_generator("fcfp")
def fcfp_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "fcfp:4", 2048, selected_feature_columns)


@register_features_generator("fcfp_count")
def fcfp_count_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "fcfp-count:4", 2048, selected_feature_columns)


@register_features_generator("topological")
def topological_torsion_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "topological", 2048, selected_feature_columns)


@register_features_generator("topological_count")
def topological_torsion_count_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "topological-count", 2048, selected_feature_columns)


@register_features_generator("layered")
def layered_fingerprint_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "layered", 2048, selected_feature_columns)


@register_features_generator("avalon_count")
def avalon_count_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "avalon-count", 512, selected_feature_columns)


@register_features_generator("rdkit_count")
def rdkit_count_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "rdkit-count", 2048, selected_feature_columns)


@register_features_generator("atompair_count")
def atompair_count_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "atompair-count", 2048, selected_feature_columns)


@register_features_generator("pattern")
def pattern_fingerprint_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "pattern", 2048, selected_feature_columns)


@register_features_generator("estate")
def estate_fingerprint_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "estate", 79, selected_feature_columns)


@register_features_generator("secfp")
def secfp_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "secfp", 2048, selected_feature_columns)


@register_features_generator("map4")
def map4_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    """Generates canonicalized, folded MAP4 v1.0/Molfeat-compatible bits."""
    calculator = _get_map4_calculator("v1.0")
    return _select_map4_features(calculator.calculate(mol), selected_feature_columns)


@register_features_generator("map4_v1_1")
def map4_v1_1_features_generator(
    mol: Molecule, selected_feature_columns: list = None,
) -> np.ndarray:
    """Generates canonicalized folded fingerprints with map4's v1.1 API."""
    calculator = _get_map4_calculator("v1.1")
    features = calculator.calculate(_canonical_map4_mol(mol))
    return _select_map4_features(features, selected_feature_columns)


def _map4_batch_features(
    mols: Sequence[Molecule], version: str,
    selected_feature_columns: Optional[Sequence[str]] = None,
) -> Sequence[np.ndarray]:
    if not mols:
        return []
    calculator = _get_map4_calculator(version)
    if version == "v1.0":
        features = calculator.calculate_many(mols, number_of_threads=1)
    else:
        # Avoid map4 1.1's per-call multiprocessing pool: Chemprop can already
        # parallelize batches, while a serial native loop is faster for common
        # feature-cache batch sizes and behaves consistently on job schedulers.
        features = np.stack([
            calculator.calculate(_canonical_map4_mol(mol)) for mol in mols
        ])
    selected = _selection_tuple(selected_feature_columns)
    return _select_map4_features(features, selected)


map4_features_generator.batch_transform = functools.partial(
    _map4_batch_features, version="v1.0",
)
map4_features_generator.preferred_batch_size = 256
map4_features_generator.batch_supports_selected_columns = True
map4_features_generator.parallel_safe = True
map4_features_generator.prefer_process_pool_when_requested = True

map4_v1_1_features_generator.batch_transform = functools.partial(
    _map4_batch_features, version="v1.1",
)
map4_v1_1_features_generator.preferred_batch_size = 256
map4_v1_1_features_generator.batch_supports_selected_columns = True
map4_v1_1_features_generator.parallel_safe = True
map4_v1_1_features_generator.prefer_process_pool_when_requested = True


@register_features_generator("cats2d")
def cats2d_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "cats2D", 189, selected_feature_columns)


@register_features_generator("scaffoldkeys")
def scaffold_keys_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "scaffoldkeys", 42, selected_feature_columns)


@register_features_generator("pharm2d")
def pharmacophore_2d_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _molfeat_features(mol, "pharm2D", 2048, selected_feature_columns)


for _generator, _kind, _length, _batch_size in (
    (fcfp_features_generator, "fcfp:4", 2048, 512),
    (fcfp_count_features_generator, "fcfp-count:4", 2048, 512),
    (topological_torsion_features_generator, "topological", 2048, 512),
    (topological_torsion_count_features_generator, "topological-count", 2048, 512),
    (layered_fingerprint_features_generator, "layered", 2048, 512),
    (avalon_count_features_generator, "avalon-count", 512, 512),
    (rdkit_count_features_generator, "rdkit-count", 2048, 512),
    (atompair_count_features_generator, "atompair-count", 2048, 512),
    (pattern_fingerprint_features_generator, "pattern", 2048, 512),
    (estate_fingerprint_features_generator, "estate", 79, 512),
    (secfp_features_generator, "secfp", 2048, 256),
    (cats2d_features_generator, "cats2D", 189, 256),
    (scaffold_keys_features_generator, "scaffoldkeys", 42, 256),
    (pharmacophore_2d_features_generator, "pharm2D", 2048, 256),
):
    _configure_molfeat_batch(_generator, _kind, _length, _batch_size)


# Lazy pretrained molfeat transformers -------------------------------------

_PRETRAINED_TRANSFORMER_CACHE: Dict[Tuple[object, Tuple[Tuple[str, object], ...]], object] = {}
_PRETRAINED_TRANSFORMER_CACHE_LOCK = Lock()


def _load_pretrained_transformer_class(transformer_type: str):
    # Importing any ``molfeat.trans`` submodule eagerly imports its fingerprint
    # calculators.  Patch map4's removed v1.0 class name before pretrained
    # backends take that import path as well as before FPVecTransformer does.
    _prepare_molfeat_optional_dependency_compatibility()
    try:
        if transformer_type == "hf":
            from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
            return PretrainedHFTransformer
        if transformer_type == "dgl":
            from molfeat.trans.pretrained import PretrainedDGLTransformer
            return PretrainedDGLTransformer
        if transformer_type == "graphormer":
            from molfeat.trans.pretrained import GraphormerTransformer
            return GraphormerTransformer
    except ImportError as exc:
        raise ImportError(
            f"The {transformer_type} pretrained generator requires `pip install 'molfeat[all]'` "
            "and its model backend."
        ) from exc
    raise ValueError(f"Unknown pretrained transformer type: {transformer_type}")


def _get_cached_pretrained_transformer(transformer_type: str, **init_kwargs):
    transformer_cls = _load_pretrained_transformer_class(transformer_type)
    cache_key = (transformer_cls, tuple(sorted(init_kwargs.items())))
    transformer = _PRETRAINED_TRANSFORMER_CACHE.get(cache_key)
    if transformer is not None:
        return transformer
    with _PRETRAINED_TRANSFORMER_CACHE_LOCK:
        transformer = _PRETRAINED_TRANSFORMER_CACHE.get(cache_key)
        if transformer is None:
            transformer = transformer_cls(**init_kwargs)
            _PRETRAINED_TRANSFORMER_CACHE[cache_key] = transformer
    return transformer


def _pretrained_transformer_features(
    mol: Molecule, transformer_type: str, kind: str, **init_kwargs,
) -> np.ndarray:
    transformer = _get_cached_pretrained_transformer(
        transformer_type, kind=kind, dtype=float, **init_kwargs
    )
    return np.asarray(transformer(_as_smiles(mol)))[0]


def _pretrained_transformer_batch_features(
    mols: Sequence[Molecule], transformer_type: str, kind: str, **init_kwargs,
) -> Sequence[np.ndarray]:
    if not mols:
        return []
    transformer = _get_cached_pretrained_transformer(
        transformer_type, kind=kind, dtype=float, **init_kwargs
    )
    return np.asarray(transformer([_as_smiles(mol) for mol in mols]))


def _configure_pretrained_batch(
    generator: FeaturesGenerator, transformer_type: str, kind: str,
    batch_size: int, **init_kwargs,
):
    generator.batch_transform = functools.partial(
        _pretrained_transformer_batch_features,
        transformer_type=transformer_type, kind=kind, **init_kwargs,
    )
    generator.preferred_batch_size = batch_size
    generator.parallel_safe = False


@register_features_generator("Roberta-Zinc480M-102M")
def roberta_zinc_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _pretrained_transformer_features(mol, "hf", "Roberta-Zinc480M-102M", notation="smiles")


@register_features_generator("GPT2-Zinc480M-87M")
def gpt2_zinc_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _pretrained_transformer_features(mol, "hf", "GPT2-Zinc480M-87M", notation="smiles")


@register_features_generator("MolT5")
def molt5_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _pretrained_transformer_features(mol, "hf", "MolT5", notation="smiles")


@register_features_generator("ChemBERTa-77M-MTR")
def chemberta_mtr_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _pretrained_transformer_features(mol, "hf", "ChemBERTa-77M-MTR", notation="smiles")


@register_features_generator("ChemBERTa-77M-MLM")
def chemberta_mlm_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _pretrained_transformer_features(mol, "hf", "ChemBERTa-77M-MLM", notation="smiles")


@register_features_generator("ChemGPT-19M")
def chemgpt_19m_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _pretrained_transformer_features(mol, "hf", "ChemGPT-19M", notation="selfies")


@register_features_generator("ChemGPT-4.7M")
def chemgpt_47m_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _pretrained_transformer_features(mol, "hf", "ChemGPT-4.7M", notation="selfies")


@register_features_generator("gin_supervised_masking")
def gin_masking_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _pretrained_transformer_features(mol, "dgl", "gin_supervised_masking")


@register_features_generator("gin_supervised_infomax")
def gin_infomax_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _pretrained_transformer_features(mol, "dgl", "gin_supervised_infomax")


@register_features_generator("gin_supervised_edgepred")
def gin_edgepred_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _pretrained_transformer_features(mol, "dgl", "gin_supervised_edgepred")


@register_features_generator("jtvae_zinc_no_kl")
def jtvae_zinc_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _pretrained_transformer_features(mol, "dgl", "jtvae_zinc_no_kl")


@register_features_generator("gin_supervised_contextpred")
def gin_contextpred_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _pretrained_transformer_features(mol, "dgl", "gin_supervised_contextpred")


@register_features_generator("pcqm4mv2_graphormer_base")
def graphormer_pcqm4mv2_features_generator(mol: Molecule, selected_feature_columns: list = None) -> np.ndarray:
    return _pretrained_transformer_features(mol, "graphormer", "pcqm4mv2_graphormer_base")


for _generator, _transformer_type, _kind, _batch_size, _kwargs in (
    (roberta_zinc_features_generator, "hf", "Roberta-Zinc480M-102M", 64, {"notation": "smiles"}),
    (gpt2_zinc_features_generator, "hf", "GPT2-Zinc480M-87M", 64, {"notation": "smiles"}),
    (molt5_features_generator, "hf", "MolT5", 32, {"notation": "smiles"}),
    (chemberta_mtr_features_generator, "hf", "ChemBERTa-77M-MTR", 64, {"notation": "smiles"}),
    (chemberta_mlm_features_generator, "hf", "ChemBERTa-77M-MLM", 64, {"notation": "smiles"}),
    (chemgpt_19m_features_generator, "hf", "ChemGPT-19M", 64, {"notation": "selfies"}),
    (chemgpt_47m_features_generator, "hf", "ChemGPT-4.7M", 64, {"notation": "selfies"}),
    (gin_masking_features_generator, "dgl", "gin_supervised_masking", 128, {}),
    (gin_infomax_features_generator, "dgl", "gin_supervised_infomax", 128, {}),
    (gin_edgepred_features_generator, "dgl", "gin_supervised_edgepred", 128, {}),
    (jtvae_zinc_features_generator, "dgl", "jtvae_zinc_no_kl", 64, {}),
    (gin_contextpred_features_generator, "dgl", "gin_supervised_contextpred", 128, {}),
    (graphormer_pcqm4mv2_features_generator, "graphormer", "pcqm4mv2_graphormer_base", 16, {}),
):
    _configure_pretrained_batch(_generator, _transformer_type, _kind, _batch_size, **_kwargs)


def clear_pretrained_transformer_cache() -> None:
    """Releases references to process-local pretrained transformers."""
    with _PRETRAINED_TRANSFORMER_CACHE_LOCK:
        _PRETRAINED_TRANSFORMER_CACHE.clear()


def clear_features_generator_caches() -> None:
    """Clears all process-local feature generator/calculator caches."""
    global _DESCRIPTASTORUS_COLUMNS, _RDKIT_PROPS, _PADEL_COLUMNS
    with _RDKIT_FP_GENERATOR_LOCK:
        _RDKIT_FP_GENERATOR_CACHE.clear()
    with _DESCRIPTASTORUS_GENERATOR_LOCK:
        _DESCRIPTASTORUS_GENERATOR_CACHE.clear()
        _DESCRIPTASTORUS_COLUMNS = None
    with _MOLECULAR_DESCRIPTOR_CALCULATOR_LOCK:
        _MOLECULAR_DESCRIPTOR_CALCULATOR_CACHE.clear()
        _RDKIT_PROPS = None
    with _MOLFEAT_TRANSFORMER_LOCK:
        _MOLFEAT_TRANSFORMER_CACHE.clear()
    with _MAP4_CALCULATOR_LOCK:
        _MAP4_CALCULATOR_CACHE.clear()
    with _PADEL_COLUMNS_LOCK:
        _PADEL_COLUMNS = None
    _get_mordred_calculator.cache_clear()
    _get_mordred_base.cache_clear()
    clear_pretrained_transformer_cache()


def generate_features_batch(
    features_generator_name: str,
    mols: Sequence[Molecule],
    selected_feature_columns: Optional[Sequence[str]] = None,
    batch_size: Optional[int] = None,
) -> Sequence[np.ndarray]:
    """Generates an ordered molecule batch using a native batch API when safe.

    Callers remain responsible for applying Chemprop's invalid-molecule,
    hydrogen-only, reaction/reactant and multi-SMILES-column policies before
    passing molecules here.  Supplying selected columns deliberately falls
    back to scalar calls unless the generator can preserve that exact schema.
    """
    generator = get_features_generator(features_generator_name)
    batch_transform = getattr(generator, "batch_transform", None)
    effective_batch_size = (
        getattr(generator, "preferred_batch_size", 256)
        if batch_size is None
        else batch_size
    )
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    output: List[np.ndarray] = []
    use_batch = batch_transform is not None and (
        selected_feature_columns is None
        or getattr(generator, "batch_supports_selected_columns", False)
    )
    if use_batch:
        for start in range(0, len(mols), effective_batch_size):
            batch = mols[start:start + effective_batch_size]
            try:
                if selected_feature_columns is None:
                    rows = batch_transform(batch)
                else:
                    rows = batch_transform(
                        batch, selected_feature_columns=selected_feature_columns,
                    )
            except ImportError as exc:
                raise ImportError(
                    f"Generator {features_generator_name!r} failed for batch rows "
                    f"{start}:{start + len(batch)}: {exc}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Generator {features_generator_name!r} failed for batch rows "
                    f"{start}:{start + len(batch)}: {exc}"
                ) from exc
            if len(rows) != len(batch):
                raise RuntimeError(
                    f"Generator {features_generator_name!r} returned {len(rows)} rows "
                    f"for batch rows {start}:{start + len(batch)}."
                )
            output.extend(np.asarray(row) for row in rows)
    else:
        for index, mol in enumerate(mols):
            try:
                row = generator(mol, selected_feature_columns=selected_feature_columns)
            except ImportError as exc:
                raise ImportError(
                    f"Generator {features_generator_name!r} failed at row {index}: {exc}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Generator {features_generator_name!r} failed at row {index}: {exc}"
                ) from exc
            output.append(np.asarray(row))
    if len(output) != len(mols):
        raise RuntimeError(
            f"Generator {features_generator_name!r} returned {len(output)} rows "
            f"for {len(mols)} molecules."
        )
    return output


def _package_version(distribution: str) -> Optional[str]:
    """Returns an installed distribution version without importing the package."""
    try:
        try:
            from importlib.metadata import version
        except ImportError:  # Python 3.7
            from importlib_metadata import version
        return version(distribution)
    except Exception:
        return None


def get_features_generator_config(
    features_generator_name: str,
    selected_feature_columns: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Returns the value-affecting default configuration for a generator."""
    get_features_generator(features_generator_name)
    configurations = {
        "morgan": {"algorithm": "Morgan", "radius": 2, "num_bits": 2048, "counts": False},
        "morgan_count": {"algorithm": "Morgan", "radius": 2, "num_bits": 2048, "counts": True},
        "maccs": {"algorithm": "MACCSKeys", "num_bits": 167},
        "rdkit": {"algorithm": "RDKit", "num_bits": 2048},
        "avalon": {"algorithm": "Avalon", "num_bits": 512},
        "atompair": {"algorithm": "AtomPair", "num_bits": 2048},
        "erg": {"algorithm": "ErG", "cast": "int64"},
        "erg_float": {"algorithm": "ErG", "cast": "float64"},
        "rdkit_2d": {"collection": "descriptastorus.RDKit2D", "normalized": False},
        "rdkit_2d_normalized": {
            "collection": "descriptastorus.RDKit2DNormalized", "normalized": True,
        },
        "rdkit_2d_wo_fr": {
            "collection": "descriptastorus.RDKit2D", "normalized": False,
            "exclude_fragment_descriptors": True,
        },
        "rdkit_2d_normalized_wo_fr": {
            "collection": "descriptastorus.RDKit2DNormalized", "normalized": True,
            "exclude_fragment_descriptors": True,
        },
        "rdkit_2d_208": {"collection": "RDKit2D-208"},
        "rdkit_2d_400": {"collection": "RDKit2D-400"},
        "rdkit_2d_autocorr": {"collection": "AUTOCORR2D"},
        "rdkit_2d_bcut": {"collection": "BCUT2D"},
        "rdkit_2d_all": {"collection": "RDKit-public-functions"},
        "mordred": {"collection": "Chemprop-2D-526"},
        "padelpy": {"collection": "PaDEL-first-1444"},
        "fcfp": {"molfeat_kind": "fcfp:4", "length": 2048},
        "fcfp_count": {"molfeat_kind": "fcfp-count:4", "length": 2048},
        "topological": {"molfeat_kind": "topological", "length": 2048},
        "topological_count": {"molfeat_kind": "topological-count", "length": 2048},
        "layered": {"molfeat_kind": "layered", "length": 2048},
        "avalon_count": {"molfeat_kind": "avalon-count", "length": 512},
        "rdkit_count": {"molfeat_kind": "rdkit-count", "length": 2048},
        "atompair_count": {"molfeat_kind": "atompair-count", "length": 2048},
        "pattern": {"molfeat_kind": "pattern", "length": 2048},
        "estate": {"molfeat_kind": "estate", "length": 79},
        "secfp": {"molfeat_kind": "secfp", "length": 2048},
        "map4": {
            "algorithm": "MAP4-v1.0-folded",
            "dimensions": 2048,
            "radius": 2,
            "counted": False,
            "atom_environment_order": "lexicographic",
            "canonicalization": "canonical-nonisomeric-smiles",
            "fragment_policy": "retain-all",
        },
        "map4_v1_1": {
            "algorithm": "MAP4-v1.1-folded",
            "dimensions": 2048,
            "radius": 2,
            "counted": False,
            "atom_environment_order": "length",
            "canonicalization": "canonical-nonisomeric-smiles",
            "fragment_policy": "retain-all",
        },
        "cats2d": {"molfeat_kind": "cats2D", "length": 189},
        "scaffoldkeys": {"molfeat_kind": "scaffoldkeys", "length": 42},
        "pharm2d": {"molfeat_kind": "pharm2D", "length": 2048},
    }
    pretrained = {
        "Roberta-Zinc480M-102M": ("hf", "smiles"),
        "GPT2-Zinc480M-87M": ("hf", "smiles"),
        "MolT5": ("hf", "smiles"),
        "ChemBERTa-77M-MTR": ("hf", "smiles"),
        "ChemBERTa-77M-MLM": ("hf", "smiles"),
        "ChemGPT-19M": ("hf", "selfies"),
        "ChemGPT-4.7M": ("hf", "selfies"),
        "gin_supervised_masking": ("dgl", None),
        "gin_supervised_infomax": ("dgl", None),
        "gin_supervised_edgepred": ("dgl", None),
        "jtvae_zinc_no_kl": ("dgl", None),
        "gin_supervised_contextpred": ("dgl", None),
        "pcqm4mv2_graphormer_base": ("graphormer", None),
    }
    config = dict(configurations.get(features_generator_name, {}))
    if features_generator_name in pretrained:
        backend, notation = pretrained[features_generator_name]
        config.update({"backend": backend, "model": features_generator_name})
        if notation is not None:
            config["notation"] = notation
    config["selected_feature_columns"] = (
        None if selected_feature_columns is None
        else [str(column) for column in selected_feature_columns]
    )
    return config


def _features_generator_implementation_sha256(generator: FeaturesGenerator) -> str:
    """Hashes the module implementing a generator, with a dynamic fallback."""
    candidates = [generator]
    if isinstance(generator, functools.partial):
        candidates.append(generator.func)
    if not inspect.isroutine(generator) and hasattr(generator, "__call__"):
        candidates.extend([generator.__call__, type(generator)])

    source_path = None
    source_target = generator
    for candidate in candidates:
        try:
            candidate_path = inspect.getsourcefile(candidate)
        except (OSError, TypeError):
            candidate_path = None
        if candidate_path is not None and os.path.isfile(candidate_path):
            source_path = candidate_path
            source_target = candidate
            break

    if source_path is not None:
        with open(source_path, "rb") as source_file:
            implementation = source_file.read()
    else:
        try:
            implementation = inspect.getsource(source_target).encode("utf-8")
        except (OSError, TypeError):
            callable_type = type(generator)
            implementation = (
                f"{callable_type.__module__}:{callable_type.__qualname__}"
            ).encode("utf-8")
    callable_type = type(generator)
    identity = (
        f"{getattr(generator, '__module__', callable_type.__module__)}:"
        f"{getattr(generator, '__qualname__', getattr(generator, '__name__', callable_type.__qualname__))}"
    ).encode("utf-8")
    return hashlib.sha256(identity + b"\0" + implementation).hexdigest()


def _features_dependency_versions(
    features_generator_names: Sequence[str],
) -> Dict[str, Optional[str]]:
    """Returns versions of only the dependencies that affect the generators."""
    names = set(features_generator_names)
    dependency_names = {"rdkit"}

    descriptastorus_names = {
        "rdkit_2d", "rdkit_2d_normalized", "rdkit_2d_wo_fr",
        "rdkit_2d_normalized_wo_fr", "rdkit_2d_208", "rdkit_2d_400",
    }
    normalized_names = {"rdkit_2d_normalized", "rdkit_2d_normalized_wo_fr"}
    molfeat_names = {
        "fcfp", "fcfp_count", "topological", "topological_count", "layered",
        "avalon_count", "rdkit_count", "atompair_count", "pattern", "estate",
        "secfp", "cats2d", "scaffoldkeys", "pharm2d",
        "Roberta-Zinc480M-102M", "GPT2-Zinc480M-87M", "MolT5",
        "ChemBERTa-77M-MTR", "ChemBERTa-77M-MLM", "ChemGPT-19M",
        "ChemGPT-4.7M", "gin_supervised_masking", "gin_supervised_infomax",
        "gin_supervised_edgepred", "jtvae_zinc_no_kl",
        "gin_supervised_contextpred", "pcqm4mv2_graphormer_base",
    }
    hf_pretrained_names = {
        "Roberta-Zinc480M-102M", "GPT2-Zinc480M-87M", "MolT5",
        "ChemBERTa-77M-MTR", "ChemBERTa-77M-MLM", "ChemGPT-19M",
        "ChemGPT-4.7M",
    }
    dgl_pretrained_names = {
        "gin_supervised_masking", "gin_supervised_infomax",
        "gin_supervised_edgepred", "jtvae_zinc_no_kl",
        "gin_supervised_contextpred",
    }
    if names & descriptastorus_names:
        dependency_names.add("descriptastorus")
    if names & normalized_names:
        dependency_names.add("scipy")
    if "mordred" in names:
        dependency_names.update({"mordred", "mordredcommunity"})
    if "padelpy" in names:
        dependency_names.add("padelpy")
    if names & molfeat_names:
        dependency_names.update({"molfeat", "datamol"})
    if "map4" in names:
        dependency_names.add("mhfp")
    if "map4_v1_1" in names:
        dependency_names.update({"map4", "mhfp"})
    if "secfp" in names:
        dependency_names.add("mhfp")
    if names & hf_pretrained_names:
        dependency_names.update({"torch", "transformers"})
    if names & dgl_pretrained_names:
        dependency_names.update({"torch", "dgllife"})
    if "pcqm4mv2_graphormer_base" in names:
        dependency_names.add("torch")

    return {
        dependency: (
            getattr(rdBase, "rdkitVersion", None)
            if dependency == "rdkit"
            else _package_version(dependency)
        )
        for dependency in sorted(dependency_names)
    }


def get_features_generators_metadata(
    features_generator_names: Sequence[str],
    selected_feature_columns: Optional[Dict[str, Sequence[str]]] = None,
    total_dimension: Optional[int] = None,
) -> Dict[str, object]:
    """Builds checkpoint metadata for an ordered generator pipeline.

    The metadata deliberately records only dependencies which can affect the
    selected generators. This prevents silent same-width schema changes while
    avoiding false incompatibilities from unrelated optional packages.
    """
    selected_feature_columns = selected_feature_columns or {}
    names = list(features_generator_names)
    versions = _features_dependency_versions(names)
    generator_entries = []
    for name in names:
        generator = get_features_generator(name)
        generator_entries.append({
            "name": name,
            "config": get_features_generator_config(
                name, selected_feature_columns.get(name),
            ),
            "implementation_sha256": _features_generator_implementation_sha256(generator),
        })

    return {
        "schema_version": 1,
        "generators": generator_entries,
        "versions": versions,
        "total_dimension": total_dimension,
        "reaction_selected_features": "reactant-selected-v1",
    }


def get_features_generator_schema(
    features_generator_name: str,
    feature_vector: Optional[Sequence[float]] = None,
    selected_feature_columns: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Returns the ordered schema and dependency versions for a generator.

    ``feature_vector`` should be supplied by bulk-generation callers.  It lets
    dynamically-sized pretrained models be described without a second model
    invocation and records the exact dtype that was written to disk.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        get_features_generator(features_generator_name)  # raises the standard error

    selected = _selection_tuple(selected_feature_columns)
    names: Tuple[str, ...] = ()
    dtype = "float64"

    fixed_fingerprints = {
        "morgan": (2048, "bit", "float64"),
        "morgan_count": (2048, "count", "float64"),
        "maccs": (167, "bit", "int64"),
        "rdkit": (2048, "bit", "int64"),
        "avalon": (512, "bit", "int64"),
        "atompair": (2048, "bit", "int64"),
        "erg": (315, "erg", "int64"),
        "erg_float": (315, "erg", "float64"),
        "fcfp": (2048, "fp", "float64"),
        "fcfp_count": (2048, "fp", "float64"),
        "topological": (2048, "fp", "float64"),
        "topological_count": (2048, "fp", "float64"),
        "layered": (2048, "fp", "float64"),
        "avalon_count": (512, "fp", "float64"),
        "rdkit_count": (2048, "fp", "float64"),
        "atompair_count": (2048, "fp", "float64"),
        "pattern": (2048, "fp", "float64"),
        "estate": (79, "fp", "float64"),
        "secfp": (2048, "fp", "float64"),
        "map4": (2048, "fp", "float64"),
        "map4_v1_1": (2048, "fp", "float64"),
        "pharm2d": (2048, "pharm2d", "float64"),
    }
    if features_generator_name in fixed_fingerprints:
        width, prefix, dtype = fixed_fingerprints[features_generator_name]
        # Molfeat's Pharm2D calculator exposes stable ``Desc:<index>`` column
        # labels, whereas the bit-vector calculators use ``fp_<index>``.
        available = (
            tuple(f"Desc:{index}" for index in range(width))
            if features_generator_name == "pharm2d"
            else tuple(f"{prefix}_{index}" for index in range(width))
        )
        if features_generator_name in {
            "morgan", "morgan_count", "maccs", "rdkit", "avalon",
            "atompair", "erg", "erg_float",
        }:
            # These legacy generators accept the argument for API
            # compatibility but historically do not select fingerprint bits.
            names = available
        else:
            names = _validate_columns(selected, available)
    elif features_generator_name in {
        "rdkit_2d", "rdkit_2d_normalized", "rdkit_2d_wo_fr",
        "rdkit_2d_normalized_wo_fr",
    }:
        available = _get_descriptastorus_columns()
        if features_generator_name.endswith("_wo_fr"):
            available = tuple(column for column in available if "fr_" not in column)
        names = _validate_columns(selected, available)
    elif features_generator_name == "rdkit_2d_208":
        names = _validate_columns(selected, _rdkit_208_columns())
    elif features_generator_name.startswith("rdkit_2d_"):
        kind = {
            "rdkit_2d_400": "400",
            "rdkit_2d_autocorr": "autocorr",
            "rdkit_2d_bcut": "bcut",
            "rdkit_2d_all": "all",
        }.get(features_generator_name)
        if kind is not None:
            available = tuple(name for name, _ in _native_descriptor_columns(kind))
            names = _validate_columns(selected, available)
    elif features_generator_name == "mordred":
        _, _, available = _get_mordred_base()
        names = _validate_columns(selected, available)
    elif features_generator_name == "padelpy":
        if _PADEL_COLUMNS is not None:
            names = _validate_columns(selected, _PADEL_COLUMNS)
    elif features_generator_name in {"cats2d", "scaffoldkeys"}:
        kind, length = (
            ("cats2D", 189) if features_generator_name == "cats2d" else ("scaffoldkeys", 42)
        )
        try:
            transformer = _get_molfeat_transformer(kind, length)
            available = _molfeat_output_columns(transformer, length)
            names = _validate_columns(selected, available)
        except ImportError:
            # The static dimension remains useful in an environment that is
            # only inspecting a manifest produced elsewhere.
            names = tuple(f"feature_{index}" for index in range(length))

    vector_array = None if feature_vector is None else np.asarray(feature_vector)
    if vector_array is not None:
        dimension = int(vector_array.size)
        dtype = str(vector_array.dtype)
    else:
        dimension = len(names) if names else None

    if not names and dimension is not None:
        prefix = "embedding" if features_generator_name in {
            "Roberta-Zinc480M-102M", "GPT2-Zinc480M-87M", "MolT5",
            "ChemBERTa-77M-MTR", "ChemBERTa-77M-MLM", "ChemGPT-19M",
            "ChemGPT-4.7M", "gin_supervised_masking", "gin_supervised_infomax",
            "gin_supervised_edgepred", "jtvae_zinc_no_kl",
            "gin_supervised_contextpred", "pcqm4mv2_graphormer_base",
        } else "feature"
        names = tuple(f"{prefix}_{index}" for index in range(dimension))
    if dimension is not None and len(names) != dimension:
        # This catches dependency-version schema drift before a misleading
        # manifest is persisted.
        raise ValueError(
            f"Schema for {features_generator_name!r} has {len(names)} names but "
            f"the generated vector has {dimension} values."
        )

    return {
        "schema_version": 1,
        "generator": features_generator_name,
        "feature_names": list(names),
        "dimension": dimension,
        "dtype": dtype if dimension is not None else None,
        "generator_config": get_features_generator_config(
            features_generator_name, selected_feature_columns,
        ),
        "implementation_sha256": _features_generator_implementation_sha256(
            get_features_generator(features_generator_name)
        ),
        "versions": _features_dependency_versions([features_generator_name]),
    }


del _generator, _kind, _length, _batch_size, _normalized, _without_fragments
del _transformer_type, _kwargs
