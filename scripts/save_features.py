"""Computes and saves molecular features for a dataset."""

import hashlib
import json
from multiprocessing import get_context
import os
import shutil
import sys
import tempfile
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple
import warnings
import zipfile

import numpy as np
from tqdm import tqdm
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

# When invoked as ``python scripts/save_features.py``, ``sys.path[0]`` is the
# scripts directory.  Put this checkout first so another editable Chemprop
# installation (for example the production ``~/chemprop`` tree) cannot be
# imported accidentally.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import get_smiles
from chemprop.features import get_available_features_generators, get_features_generator, load_features, save_features
from chemprop.features.features_generators import get_features_generator_schema
from chemprop.utils import makedirs


_FEATURES_WORKER = None


def _initialize_features_worker(features_generator_name: str) -> None:
    """Initializes one generator per worker process."""
    global _FEATURES_WORKER
    _FEATURES_WORKER = get_features_generator(features_generator_name)


def _generate_features_worker(smiles: str):
    """Generates one row through the process-local registry function."""
    if _FEATURES_WORKER is None:
        raise RuntimeError("Feature worker was not initialized.")
    return _FEATURES_WORKER(smiles)


class Args(Tap):
    data_path: str  # Path to data CSV
    smiles_column: str = None  # Name of the column containing SMILES strings. By default, uses the first column.
    features_generator: str = 'rdkit_2d_normalized'  # Type of features to generate
    save_path: str  # Path to .npz file where features will be saved as a compressed numpy archive
    save_frequency: int = 10000  # Frequency with which to save the features
    restart: bool = False  # Whether to not load partially complete featurization and instead start from scratch
    sequential: bool = False  # Whether to run sequentially rather than in parallel
    num_workers: int = None  # Worker processes. Explicit values >1 enable MAP4 multiprocessing.
    chunksize: int = 100  # Number of molecules dispatched to a CPU worker at once
    batch_size: int = None  # Molecules per call for generators exposing a native batch API

    def configure(self) -> None:
        self.add_argument('--features_generator', choices=get_available_features_generators())


def load_temp(temp_dir: str) -> Tuple[List[List[float]], int]:
    """
    Loads all features saved as .npz files in load_dir.

    Assumes temporary files are named in order 0.npz, 1.npz, ...

    :param temp_dir: Directory in which temporary .npz files containing features are stored.
    :return: A tuple with a list of molecule features, where each molecule's features is a list of floats,
    and the number of temporary files.
    """
    features = []
    temp_num = 0
    temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    while os.path.exists(temp_path):
        features.extend(load_features(temp_path))
        temp_num += 1
        temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    return features, temp_num


def _scan_temp_chunks(temp_dir: str) -> Tuple[int, int, np.ndarray, int, str]:
    """Validates persisted chunks without retaining their feature matrices."""
    temp_num = 0
    completed = 0
    feature_vector = None
    dimension = None
    dtype = None
    while True:
        temp_path = os.path.join(temp_dir, f'{temp_num}.npz')
        if not os.path.exists(temp_path):
            break
        chunk = np.asarray(load_features(temp_path))
        if chunk.ndim != 2 or chunk.shape[0] == 0:
            raise ValueError(
                f'Temporary feature archive {temp_num}.npz is not a non-empty 2D array. '
                'Use --restart.'
            )
        chunk_dimension = int(chunk.shape[1])
        chunk_dtype = str(chunk.dtype)
        if dimension is None:
            dimension = chunk_dimension
            dtype = chunk_dtype
            feature_vector = np.array(chunk[0], copy=True)
        elif chunk_dimension != dimension or chunk_dtype != dtype:
            raise ValueError(
                'Temporary feature archives contain inconsistent dimensions or dtypes. '
                'Use --restart.'
            )
        completed += len(chunk)
        temp_num += 1

    return temp_num, completed, feature_vector, dimension, dtype


def _validate_feature_chunk(
    rows: Sequence, dimension: int = None, dtype: str = None,
) -> Tuple[np.ndarray, int, str]:
    """Checks one bounded chunk and returns its first row and realized schema."""
    if not rows:
        raise ValueError('Cannot persist an empty feature chunk.')
    first_row = np.asarray(rows[0])
    if first_row.ndim != 1:
        raise ValueError('Feature generators must return one-dimensional arrays.')
    realized_dimension = int(first_row.size)
    realized_dtype = str(first_row.dtype)
    if dimension is not None and realized_dimension != dimension:
        raise ValueError(
            f'Feature dimension changed from {dimension} to {realized_dimension}.'
        )
    if dtype is not None and realized_dtype != dtype:
        raise ValueError(f'Feature dtype changed from {dtype} to {realized_dtype}.')
    for row in rows[1:]:
        row_array = np.asarray(row)
        if row_array.ndim != 1 or row_array.size != realized_dimension:
            raise ValueError('Feature generator returned inconsistent row dimensions.')
        if str(row_array.dtype) != realized_dtype:
            raise ValueError('Feature generator returned inconsistent row dtypes.')
    return np.array(first_row, copy=True), realized_dimension, realized_dtype


def _atomic_save_feature_chunk(path: str, rows: Sequence) -> None:
    """Persists one resume chunk before publishing its filename."""
    directory = os.path.dirname(os.path.abspath(path))
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(path)}.', suffix='.npz', dir=directory,
    )
    os.close(file_descriptor)
    try:
        save_features(temporary_path, rows)
        with open(temporary_path, 'rb') as file:
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def _atomic_consolidate_chunks(
    save_path: str,
    temp_dir: str,
    temporary_file_count: int,
    total_rows: int,
    dimension: int,
    dtype: str,
) -> None:
    """Streams chunk archives through a disk-backed .npy into an atomic .npz."""
    directory = os.path.dirname(os.path.abspath(save_path))
    npy_descriptor, npy_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(save_path)}.', suffix='.npy', dir=directory,
    )
    os.close(npy_descriptor)
    archive_descriptor, archive_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(save_path)}.', suffix='.npz', dir=directory,
    )
    os.close(archive_descriptor)
    try:
        if total_rows == 0:
            # Preserve the historical ``save_features(path, [])`` archive
            # shape while the manifest continues to describe static width.
            np.save(npy_path, np.asarray([], dtype=np.dtype(dtype or 'float64')))
        elif dimension is None:
            if total_rows != 0:
                raise ValueError('Cannot consolidate non-empty features with unknown width.')
        else:
            feature_memmap = np.lib.format.open_memmap(
                npy_path,
                mode='w+',
                dtype=np.dtype(dtype),
                shape=(total_rows, dimension),
            )
            offset = 0
            for temp_num in range(temporary_file_count):
                chunk = np.asarray(
                    load_features(os.path.join(temp_dir, f'{temp_num}.npz'))
                )
                next_offset = offset + len(chunk)
                if (
                    chunk.ndim != 2
                    or chunk.shape[1] != dimension
                    or str(chunk.dtype) != dtype
                    or next_offset > total_rows
                ):
                    raise ValueError(
                        'Temporary feature archives changed during consolidation.'
                    )
                feature_memmap[offset:next_offset] = chunk
                offset = next_offset
            if offset != total_rows:
                raise ValueError(
                    f'Temporary features contain {offset} rows; expected {total_rows}.'
                )
            feature_memmap.flush()
            del feature_memmap

        with zipfile.ZipFile(
            archive_path,
            mode='w',
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            archive.write(npy_path, arcname='features.npy')
        with open(archive_path, 'rb') as file:
            os.fsync(file.fileno())
        os.replace(archive_path, save_path)
    except Exception:
        if os.path.exists(archive_path):
            os.remove(archive_path)
        raise
    finally:
        if os.path.exists(npy_path):
            os.remove(npy_path)


def _iter_batches(values: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _iter_batched_features(
    smiles: Sequence[str], batch_transform: Callable[[Sequence[str]], Sequence], batch_size: int,
) -> Iterator:
    """Yields native-batch results while bounding temporary memory usage."""
    for batch in _iter_batches(smiles, batch_size):
        batch_features = batch_transform(batch)
        if len(batch_features) != len(batch):
            raise RuntimeError(
                f"Feature batch returned {len(batch_features)} rows for {len(batch)} molecules."
            )
        yield from batch_features


def _default_num_workers() -> int:
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count() or 1
    # Descriptor calculators and model backends can be memory intensive.
    return max(1, min(cpu_count, 8))


def _prefer_requested_process_pool(args: Args, features_generator: Callable) -> bool:
    """Lets selected CPU-heavy pseudo-batch generators honor --num_workers."""
    return (
        not args.sequential
        and args.num_workers is not None
        and args.num_workers > 1
        and getattr(
            features_generator, 'prefer_process_pool_when_requested', False,
        )
    )


def _atomic_write_json(path: str, payload: dict) -> None:
    """Writes JSON without exposing a partially-written manifest."""
    directory = os.path.dirname(os.path.abspath(path))
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(path)}.', suffix='.tmp', dir=directory,
    )
    try:
        with os.fdopen(file_descriptor, 'w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _ordered_smiles_sha256(smiles: Sequence[str]) -> str:
    """Hashes ordered, length-prefixed UTF-8 strings in bounded memory."""
    digest = hashlib.sha256(b'chemprop-ordered-smiles-v1\0')
    for value in smiles:
        encoded = str(value).encode('utf-8')
        digest.update(len(encoded).to_bytes(8, byteorder='big', signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _input_identity(data_path: str, smiles: Sequence[str]) -> dict:
    return {
        'data_path': os.path.abspath(data_path),
        'data_sha256': _file_sha256(data_path),
        'ordered_smiles_sha256': _ordered_smiles_sha256(smiles),
        'ordered_smiles_encoding': 'chemprop-length-prefixed-utf8-v1',
        'num_smiles': len(smiles),
    }


def _validate_resume_manifest(path: str, expected_schema: dict, expected_input: dict) -> dict:
    with open(path, encoding='utf-8') as file:
        manifest = json.load(file)
    comparisons = {
        'schema_version': (
            manifest.get('schema_version'), expected_schema.get('schema_version'),
        ),
        'generator': (manifest.get('generator'), expected_schema.get('generator')),
        'implementation_sha256': (
            manifest.get('implementation_sha256'),
            expected_schema.get('implementation_sha256'),
        ),
        'generator_config': (
            manifest.get('generator_config'), expected_schema.get('generator_config'),
        ),
        'versions': (manifest.get('versions'), expected_schema.get('versions')),
        'data_sha256': (
            manifest.get('input', {}).get('data_sha256'), expected_input['data_sha256'],
        ),
        'ordered_smiles_sha256': (
            manifest.get('input', {}).get('ordered_smiles_sha256'),
            expected_input['ordered_smiles_sha256'],
        ),
        'num_smiles': (
            manifest.get('input', {}).get('num_smiles'), expected_input['num_smiles'],
        ),
    }
    for field in ('dimension', 'dtype'):
        if expected_schema.get(field) is not None:
            comparisons[field] = (manifest.get(field), expected_schema.get(field))
    mismatches = [name for name, (actual, expected) in comparisons.items() if actual != expected]
    if mismatches:
        raise ValueError(
            f"Cannot resume feature generation because {', '.join(mismatches)} changed. "
            "Use --restart to discard the existing temporary features."
        )
    return manifest


def _save_manifest(
    args: Args,
    features: Sequence,
    storage: str,
    temporary_file_count: int,
    input_identity: dict,
    total_molecules: int,
    status: str,
    prior_schema: dict = None,
    feature_vector=None,
    num_molecules_completed: int = None,
) -> dict:
    if feature_vector is None and len(features):
        feature_vector = features[0]
    if num_molecules_completed is None:
        num_molecules_completed = len(features)
    manifest = get_features_generator_schema(
        args.features_generator, feature_vector=feature_vector,
    )
    if prior_schema is not None and manifest['dimension'] is not None:
        generic_names = [f'feature_{index}' for index in range(manifest['dimension'])]
        prior_names = prior_schema.get('feature_names', [])
        if (
            manifest['feature_names'] == generic_names
            and len(prior_names) == manifest['dimension']
        ):
            manifest['feature_names'] = prior_names
    manifest.update({
        'feature_file': os.path.basename(args.save_path),
        'manifest_file': os.path.basename(args.save_path) + '.manifest.json',
        'input': input_identity,
        'num_molecules': total_molecules,
        'num_molecules_completed': num_molecules_completed,
        'status': status,
        'storage': storage,
        'temporary_file_count': temporary_file_count if storage == 'chunk_directory' else 0,
    })
    _atomic_write_json(args.save_path + '.manifest.json', manifest)
    return manifest


def generate_and_save_features(args: Args):
    """
    Computes and saves features for a dataset of molecules as a 2D array in a .npz file.

    :param args: Arguments.
    """
    if args.save_frequency <= 0:
        raise ValueError('save_frequency must be positive.')
    if args.chunksize <= 0:
        raise ValueError('chunksize must be positive.')
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError('batch_size must be positive when supplied.')
    if args.num_workers is not None and args.num_workers <= 0:
        raise ValueError('num_workers must be positive when supplied.')

    # Create directory for save_path
    makedirs(args.save_path, isfile=True)

    # Get data and features function
    all_smiles = get_smiles(path=args.data_path, smiles_columns=args.smiles_column, flatten=True)
    features_generator = get_features_generator(args.features_generator)
    temp_save_dir = args.save_path + '_temp'
    manifest_path = args.save_path + '.manifest.json'
    input_identity = _input_identity(args.data_path, all_smiles)
    expected_schema = get_features_generator_schema(args.features_generator)
    resume_manifest = None
    feature_vector = None
    dimension = expected_schema.get('dimension')
    dtype = expected_schema.get('dtype')

    # Load partially complete data
    if args.restart:
        if os.path.exists(args.save_path):
            os.remove(args.save_path)
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        if os.path.exists(temp_save_dir):
            shutil.rmtree(temp_save_dir)
    else:
        if os.path.exists(args.save_path):
            raise ValueError(f'"{args.save_path}" already exists and args.restart is False.')

        if os.path.exists(temp_save_dir):
            if os.path.exists(manifest_path):
                resume_manifest = _validate_resume_manifest(
                    manifest_path, expected_schema, input_identity,
                )
            else:
                warnings.warn(
                    "Resuming a legacy temporary feature directory without a manifest; "
                    "its input and generator identity cannot be verified.", RuntimeWarning,
                )

    if not os.path.exists(temp_save_dir):
        makedirs(temp_save_dir)
    temp_num, completed, persisted_vector, persisted_dimension, persisted_dtype = (
        _scan_temp_chunks(temp_save_dir)
    )
    if persisted_vector is not None:
        feature_vector = persisted_vector
        if dimension is not None and persisted_dimension != dimension:
            raise ValueError(
                f'Temporary feature dimension {persisted_dimension} does not match '
                f'generator dimension {dimension}. Use --restart.'
            )
        if dtype is not None and persisted_dtype != dtype:
            raise ValueError(
                f'Temporary feature dtype {persisted_dtype} does not match '
                f'generator dtype {dtype}. Use --restart.'
            )
        dimension = persisted_dimension
        dtype = persisted_dtype

    if completed > len(all_smiles):
        raise ValueError(
            f"Temporary features contain {completed} rows but the input has only "
            f"{len(all_smiles)} molecules. Use --restart."
        )
    if resume_manifest is not None:
        recorded_completed = resume_manifest.get('num_molecules_completed')
        if recorded_completed is not None and recorded_completed > completed:
            raise ValueError(
                f"Resume manifest records {recorded_completed} completed molecules but the "
                f"temporary archives contain {completed}. Use --restart."
            )
        if feature_vector is not None:
            resumed_schema = get_features_generator_schema(
                args.features_generator, feature_vector=feature_vector,
            )
            for key in ('dimension', 'dtype'):
                recorded = resume_manifest.get(key)
                if recorded is not None and recorded != resumed_schema[key]:
                    raise ValueError(
                        f"Resume manifest {key}={recorded!r} does not match temporary "
                        f"features ({resumed_schema[key]!r}). Use --restart."
                    )

    current_manifest = _save_manifest(
        args, (), storage='chunk_directory', temporary_file_count=temp_num,
        input_identity=input_identity, total_molecules=len(all_smiles), status='in_progress',
        prior_schema=resume_manifest,
        feature_vector=feature_vector,
        num_molecules_completed=completed,
    )

    # Build features map function
    smiles = all_smiles[completed:]  # restrict to molecules whose features are not persisted

    def consume(features_iterator: Iterable) -> None:
        nonlocal current_manifest, completed, dimension, dtype, feature_vector, temp_num
        temp_features = []
        for i, feats in enumerate(tqdm(features_iterator, total=len(smiles))):
            temp_features.append(feats)

            # Save temporary features every save_frequency.
            if (i + 1) % args.save_frequency == 0 or i == len(smiles) - 1:
                chunk_vector, chunk_dimension, chunk_dtype = _validate_feature_chunk(
                    temp_features, dimension=dimension, dtype=dtype,
                )
                _atomic_save_feature_chunk(
                    os.path.join(temp_save_dir, f'{temp_num}.npz'), temp_features,
                )
                if feature_vector is None:
                    feature_vector = chunk_vector
                dimension = chunk_dimension
                dtype = chunk_dtype
                completed += len(temp_features)
                temp_features = []
                temp_num += 1
                current_manifest = _save_manifest(
                    args, (), storage='chunk_directory', temporary_file_count=temp_num,
                    input_identity=input_identity, total_molecules=len(all_smiles), status='in_progress',
                    prior_schema=current_manifest,
                    feature_vector=feature_vector,
                    num_molecules_completed=completed,
                )

    batch_transform = getattr(features_generator, 'batch_transform', None)
    use_requested_process_pool = _prefer_requested_process_pool(
        args, features_generator,
    )
    if batch_transform is not None and not use_requested_process_pool:
        effective_batch_size = args.batch_size or getattr(features_generator, 'preferred_batch_size', 256)
        consume(_iter_batched_features(smiles, batch_transform, effective_batch_size))
    elif args.sequential:
        consume(map(features_generator, smiles))
    else:
        worker_count = args.num_workers or _default_num_workers()
        context = get_context()
        with context.Pool(
            processes=worker_count,
            initializer=_initialize_features_worker,
            initargs=(args.features_generator,),
        ) as pool:
            consume(pool.imap(_generate_features_worker, smiles, chunksize=args.chunksize))

    if completed != len(all_smiles):
        raise RuntimeError(
            f'Feature generator returned {completed} rows for {len(all_smiles)} molecules.'
        )

    try:
        _atomic_consolidate_chunks(
            args.save_path,
            temp_save_dir,
            temporary_file_count=temp_num,
            total_rows=completed,
            dimension=dimension,
            dtype=dtype,
        )

        # Write the sidecar only after the archive is complete.
        _save_manifest(
            args, (), storage='npz', temporary_file_count=temp_num,
            input_identity=input_identity, total_molecules=len(all_smiles), status='complete',
            prior_schema=current_manifest,
            feature_vector=feature_vector,
            num_molecules_completed=completed,
        )

        # Remove temporary features
        shutil.rmtree(temp_save_dir)
    except OverflowError:
        _save_manifest(
            args, (), storage='chunk_directory', temporary_file_count=temp_num,
            input_identity=input_identity, total_molecules=len(all_smiles), status='complete',
            prior_schema=current_manifest,
            feature_vector=feature_vector,
            num_molecules_completed=completed,
        )
        print('Features array is too large to save as a single file. Instead keeping features as a directory of files.')


if __name__ == '__main__':
    generate_and_save_features(Args().parse_args())
