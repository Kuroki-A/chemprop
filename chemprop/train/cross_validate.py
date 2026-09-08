from collections import defaultdict
import csv
import hashlib
import inspect
import json
from logging import Logger
import os
import stat
import sys
import tempfile
from typing import Any, Callable, Dict, List, Tuple
import subprocess

import numpy as np
import pandas as pd
from packaging.version import Version

import torch

from .run_training import run_training

from chemprop.args import TrainArgs
from chemprop.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from chemprop.data import get_data, get_task_names, load_selected_feature_columns, MoleculeDataset, validate_dataset_type
from chemprop.utils import create_logger, makedirs, timeit, multitask_mean
from chemprop.features import get_features_generator, get_features_generators_metadata, set_extra_atom_fdim, set_extra_bond_fdim, set_explicit_h, set_adding_hs, set_keeping_atom_map, set_reaction, reset_featurization_parameters

try:
    from importlib import metadata
except ImportError:  # Python 3.7
    metadata = None


DATASET_CACHE_SCHEMA_VERSION = 2
RESUME_MANIFEST_SCHEMA_VERSION = 3


def _file_manifest(path: str):
    """Returns a content fingerprint for a file used to construct a dataset."""
    if path is None:
        return None

    absolute_path = os.path.abspath(os.path.expanduser(path))
    digest = hashlib.sha256()
    with open(absolute_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)

    return {
        'path': absolute_path,
        'size': os.path.getsize(absolute_path),
        'sha256': digest.hexdigest(),
    }


def _feature_file_manifest(path: str):
    """Fingerprints a feature matrix together with its optional sidecar."""
    if path is None:
        return None

    sidecar_path = f'{path}.manifest.json'
    return {
        'data': _file_manifest(path),
        'sidecar': (
            _file_manifest(sidecar_path)
            if os.path.isfile(os.path.abspath(os.path.expanduser(sidecar_path)))
            else None
        ),
    }


def _source_digest(obj):
    """Fingerprints local implementation code so dirty-tree changes invalidate caches."""
    try:
        source_path = inspect.getsourcefile(obj)
    except (OSError, TypeError):
        return None
    if source_path is None or not os.path.isfile(source_path):
        return None
    return _file_manifest(source_path)['sha256']


def _dependency_versions() -> Dict[str, str]:
    """Collects versions which can change parsing or generated features."""
    versions = {
        'numpy': np.__version__,
        'torch': str(torch.__version__),
    }
    for distribution in [
        'chemprop',
        'rdkit',
        'pandas',
        'scipy',
        'scikit-learn',
        'descriptastorus',
        'mordred',
        'mordredcommunity',
        'padelpy',
        'map4',
        'mhfp',
        'dgllife',
        'datamol',
        'molfeat',
        'transformers',
    ]:
        if metadata is None:
            versions[distribution] = None
            continue
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _dataset_cache_manifest(args: TrainArgs) -> Dict:
    """Builds a complete manifest for inputs which affect get_data output."""
    feature_paths = getattr(args, 'features_path', None)
    if feature_paths is not None:
        if isinstance(feature_paths, str):
            feature_paths = [feature_paths]
        feature_paths = [_feature_file_manifest(path) for path in feature_paths]

    feature_generators = list(getattr(args, 'features_generator', None) or [])
    generator_sources = {
        name: _source_digest(get_features_generator(name)) for name in feature_generators
    }
    selected_feature_columns = (
        load_selected_feature_columns(args.selected_features_path)
        if feature_generators and getattr(args, 'selected_features_path', None) is not None
        else {}
    )
    generator_metadata = (
        get_features_generators_metadata(
            feature_generators,
            selected_feature_columns=selected_feature_columns,
            total_dimension=None,
        )
        if feature_generators
        else None
    )

    input_path_fields = [
        'selected_features_path',
        'atom_descriptors_path',
        'bond_descriptors_path',
        'constraints_path',
        'data_weights_path',
    ]
    input_files = {
        field: _file_manifest(getattr(args, field, None)) for field in input_path_fields
    }
    input_files['data_path'] = _file_manifest(args.data_path)
    input_files['features_path'] = feature_paths
    input_files['phase_features_path'] = _feature_file_manifest(
        getattr(args, 'phase_features_path', None)
    )

    config_fields = [
        'smiles_columns',
        'number_of_molecules',
        'target_columns',
        'ignore_columns',
        'features_generator',
        'max_data_size',
        'atom_descriptors',
        'bond_descriptors',
        'overwrite_default_atom_features',
        'overwrite_default_bond_features',
        'loss_function',
        'is_atom_bond_targets',
        'reaction',
        'reaction_mode',
        'reaction_solvent',
        'explicit_h',
        'adding_h',
        'keeping_atom_map',
    ]
    config = {field: getattr(args, field, None) for field in config_fields}

    return {
        'schema_version': DATASET_CACHE_SCHEMA_VERSION,
        'python_version': list(sys.version_info[:3]),
        'dependency_versions': _dependency_versions(),
        'source_digests': {
            'get_data': _source_digest(get_data),
            'MoleculeDataset': _source_digest(MoleculeDataset),
            'feature_generators': generator_sources,
        },
        'features_generator_metadata': generator_metadata,
        'input_files': input_files,
        'config': config,
        'loader_options': {
            'skip_none_targets': True,
        },
    }


def _dataset_cache_path(args: TrainArgs, manifest: Dict) -> str:
    """Returns the content-addressed path in the dedicated dataset cache."""
    serialized = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
    cache_key = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    cache_dir = os.environ.get('CHEMPROP_CACHE_DIR')
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.data_path)), '.chemprop_cache'
        )
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    return os.path.join(cache_dir, f'dataset-{cache_key}.pt')


def _validate_private_cache_directory(cache_dir: str, create: bool) -> None:
    """Requires a non-symlink, user-private directory for pickle caches."""
    cache_dir = os.path.abspath(cache_dir)
    if os.path.lexists(cache_dir) and os.path.islink(cache_dir):
        raise ValueError(f'Dataset cache directory {cache_dir} must not be a symbolic link.')
    if create:
        os.makedirs(cache_dir, mode=0o700, exist_ok=True)
    directory_stat = os.lstat(cache_dir)
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise ValueError(f'Dataset cache path {cache_dir} is not a directory.')
    if os.name == 'posix':
        if directory_stat.st_uid != os.geteuid():
            raise ValueError(f'Dataset cache directory {cache_dir} is not owned by the current user.')
        if stat.S_IMODE(directory_stat.st_mode) & 0o077:
            raise ValueError(
                f'Dataset cache directory {cache_dir} must be private (mode 0700); '
                'dataset caches contain trusted Python pickle data.'
            )


def _validate_cache_file_stat(path: str, file_stat: os.stat_result) -> None:
    """Rejects cache files that another user could have supplied or modified."""
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f'Dataset cache {path} must be a regular file.')
    if os.name == 'posix':
        if file_stat.st_uid != os.geteuid():
            raise ValueError(f'Dataset cache {path} is not owned by the current user.')
        if stat.S_IMODE(file_stat.st_mode) & 0o022:
            raise ValueError(
                f'Dataset cache {path} must not be writable by group or other users.'
            )


def _load_dataset_cache(path: str, manifest: Dict):
    """Loads a trusted private cache only when its manifest exactly matches."""
    cache_dir = os.path.dirname(os.path.abspath(path))
    _validate_private_cache_directory(cache_dir, create=False)
    if os.path.islink(path):
        raise ValueError(f'Dataset cache {path} must not be a symbolic link.')

    flags = os.O_RDONLY
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, 'rb') as cache_file:
        _validate_cache_file_stat(path, os.fstat(cache_file.fileno()))
        if Version(str(torch.__version__)) >= Version("2.6"):
            payload = torch.load(cache_file, map_location='cpu', weights_only=False)
        else:
            payload = torch.load(cache_file, map_location='cpu')

    if not isinstance(payload, dict) or payload.get('manifest') != manifest:
        return None
    data = payload.get('data')
    return data if isinstance(data, MoleculeDataset) else None


def _save_dataset_cache(path: str, manifest: Dict, data: MoleculeDataset) -> None:
    """Atomically saves a dataset cache so readers never observe partial files."""
    cache_dir = os.path.dirname(path)
    _validate_private_cache_directory(cache_dir, create=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=cache_dir, prefix='.dataset-', suffix='.tmp'
    )
    os.close(file_descriptor)
    try:
        torch.save({'manifest': manifest, 'data': data}, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _json_default(value):
    """Converts NumPy score values into JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def _canonical_manifest_value(value: Any) -> Any:
    """Converts argument values into deterministic, JSON-compatible data."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else repr(value)
    if isinstance(value, np.generic):
        return _canonical_manifest_value(value.item())
    if isinstance(value, np.ndarray):
        return _canonical_manifest_value(value.tolist())
    if isinstance(value, dict):
        return {
            str(key): _canonical_manifest_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_manifest_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_canonical_manifest_value(item) for item in value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return str(value)


def _directory_manifest(path: str):
    """Fingerprints every regular file in a directory in stable order."""
    if path is None:
        return None
    absolute_path = os.path.abspath(os.path.expanduser(path))
    files = []
    for root, directories, filenames in os.walk(absolute_path):
        directories.sort()
        for filename in sorted(filenames):
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                manifest = _file_manifest(file_path)
                manifest['relative_path'] = os.path.relpath(file_path, absolute_path)
                manifest.pop('path', None)
                files.append(manifest)
    return {'path': absolute_path, 'files': files}


def _resume_input_manifest(args: TrainArgs) -> Dict[str, Any]:
    """Fingerprints secondary datasets and pretrained inputs used by training."""
    path_fields = [
        'config_path',
        'separate_val_path',
        'separate_test_path',
        'separate_val_atom_descriptors_path',
        'separate_test_atom_descriptors_path',
        'separate_val_bond_descriptors_path',
        'separate_test_bond_descriptors_path',
        'separate_val_constraints_path',
        'separate_test_constraints_path',
        'spectra_phase_mask_path',
        'folds_file',
        'crossval_index_file',
        'checkpoint_frzn',
    ]
    feature_list_path_fields = [
        'separate_val_features_path',
        'separate_test_features_path',
    ]
    list_path_fields = [
        'checkpoint_paths',
    ]
    inputs = {
        field: _file_manifest(getattr(args, field, None))
        for field in path_fields
    }
    for field in list_path_fields:
        paths = getattr(args, field, None)
        if isinstance(paths, str):
            paths = [paths]
        inputs[field] = (
            [_file_manifest(path) for path in paths] if paths is not None else None
        )
    for field in feature_list_path_fields:
        paths = getattr(args, field, None)
        if isinstance(paths, str):
            paths = [paths]
        inputs[field] = (
            [_feature_file_manifest(path) for path in paths]
            if paths is not None
            else None
        )
    for field in [
        'separate_val_phase_features_path',
        'separate_test_phase_features_path',
    ]:
        inputs[field] = _feature_file_manifest(getattr(args, field, None))
    inputs['crossval_index_dir'] = _directory_manifest(
        getattr(args, 'crossval_index_dir', None)
    )
    return inputs


def _training_config_manifest(args: TrainArgs) -> Dict[str, Any]:
    """Returns stable training arguments, excluding runtime/resume state."""
    excluded_fields = {
        'resume_experiment',
        'save_dir',
        'spectra_phase_mask',
        'train_class_sizes',
        'train_data_size',
        'use_cache',
    }
    return {
        key: _canonical_manifest_value(value)
        for key, value in sorted(args.as_dict().items())
        if key not in excluded_fields
    }


def _callable_manifest(train_func: Callable) -> Dict[str, Any]:
    """Identifies a training callback and fingerprints its implementation file."""
    try:
        signature = str(inspect.signature(train_func))
    except (TypeError, ValueError):
        signature = None
    try:
        source = inspect.getsource(train_func).encode('utf-8')
        implementation_sha256 = hashlib.sha256(source).hexdigest()
    except (OSError, TypeError):
        implementation_sha256 = None
    return {
        'module': getattr(train_func, '__module__', None),
        'qualname': getattr(train_func, '__qualname__', None),
        'signature': signature,
        'implementation_sha256': implementation_sha256,
        'source_file_sha256': _source_digest(train_func),
        'defaults': _canonical_manifest_value(
            getattr(train_func, '__defaults__', None)
        ),
        'keyword_defaults': _canonical_manifest_value(
            getattr(train_func, '__kwdefaults__', None)
        ),
    }


def _package_code_manifest() -> Dict[str, Any]:
    """Fingerprints all Python implementation files in the local package."""
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    digest = hashlib.sha256()
    file_count = 0
    for root, directories, filenames in os.walk(package_root):
        directories[:] = sorted(
            directory for directory in directories if directory != '__pycache__'
        )
        for filename in sorted(filenames):
            if not filename.endswith('.py'):
                continue
            path = os.path.join(root, filename)
            relative_path = os.path.relpath(path, package_root).replace(os.sep, '/')
            digest.update(relative_path.encode('utf-8'))
            digest.update(b'\0')
            with open(path, 'rb') as source_file:
                for chunk in iter(lambda: source_file.read(1024 * 1024), b''):
                    digest.update(chunk)
            digest.update(b'\0')
            file_count += 1
    return {'python_file_count': file_count, 'sha256': digest.hexdigest()}


def _fold_resume_manifest(
    args: TrainArgs,
    dataset_manifest: Dict[str, Any],
    resume_inputs: Dict[str, Any],
    package_code_manifest: Dict[str, Any],
    train_func: Callable,
    fold_num: int,
) -> Dict[str, Any]:
    """Builds the exact contract under which fold scores may be reused."""
    return {
        'schema_version': RESUME_MANIFEST_SCHEMA_VERSION,
        'fold_num': fold_num,
        'dataset': dataset_manifest,
        'secondary_inputs': resume_inputs,
        'config': _training_config_manifest(args),
        'code': {
            'chemprop_package': package_code_manifest,
            'train_func': _callable_manifest(train_func),
        },
    }


def _score_payload_digest(scores: Dict[str, List[float]]) -> str:
    serialized = json.dumps(
        scores,
        sort_keys=True,
        separators=(',', ':'),
        default=_json_default,
    )
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _atomic_json_dump(payload: Any, path: str) -> None:
    """Writes JSON via same-directory replacement so partial files are never visible."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=directory, prefix='.json-', suffix='.tmp'
    )
    try:
        with os.fdopen(descriptor, 'w') as output_file:
            json.dump(
                payload,
                output_file,
                indent=4,
                sort_keys=True,
                default=_json_default,
            )
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _validate_score_payload(
    scores: Dict[str, List[float]], args: TrainArgs, allow_empty: bool = False
) -> None:
    """Validates the common callback score contract.

    Both validation and test payloads contain every requested metric and one
    numeric value per task. Spectra metrics are the sole exception: they score
    the complete spectrum and therefore contain one value. Validation-only
    runs may use an empty test payload, but a non-empty payload is never
    allowed to omit an extra metric or substitute the ensemble-member axis.
    """
    if allow_empty and scores == {}:
        return
    if not isinstance(scores, dict) or set(scores) != set(args.metrics):
        raise ValueError('Score metric names do not match the current arguments.')
    expected_width = 1 if args.dataset_type == 'spectra' else args.num_tasks
    for metric, values in scores.items():
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f'Scores for metric "{metric}" are not numeric.') from error
        if array.shape != (expected_width,):
            raise ValueError(
                f'Scores for metric "{metric}" have shape {array.shape}; '
                f'expected ({expected_width},).'
            )


def _load_resume_scores(
    manifest_path: str,
    valid_scores_path: str,
    test_scores_path: str,
    expected_manifest: Dict[str, Any],
    args: TrainArgs,
    skip_test_evaluation: bool,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Loads a complete, matching resume transaction or raises."""
    with open(manifest_path) as manifest_file:
        record = json.load(manifest_file)
    if not isinstance(record, dict) or record.get('manifest') != expected_manifest:
        raise ValueError('Resume manifest does not match the current run.')

    with open(valid_scores_path) as valid_file:
        valid_scores = json.load(valid_file)
    if skip_test_evaluation:
        test_scores = {}
    else:
        with open(test_scores_path) as test_file:
            test_scores = json.load(test_file)

    _validate_score_payload(valid_scores, args)
    _validate_score_payload(test_scores, args, allow_empty=skip_test_evaluation)
    expected_digests = {
        'valid': _score_payload_digest(valid_scores),
        'test': None if skip_test_evaluation else _score_payload_digest(test_scores),
    }
    if record.get('score_digests') != expected_digests:
        raise ValueError('Resume score files do not match their manifest digests.')
    return valid_scores, test_scores


def _save_resume_scores(
    manifest_path: str,
    valid_scores_path: str,
    test_scores_path: str,
    manifest: Dict[str, Any],
    valid_scores: Dict[str, List[float]],
    test_scores: Dict[str, List[float]],
    skip_test_evaluation: bool,
) -> None:
    """Commits score files first and their digest-bearing manifest last."""
    _atomic_json_dump(valid_scores, valid_scores_path)
    if not skip_test_evaluation:
        _atomic_json_dump(test_scores, test_scores_path)
    record = {
        'manifest': manifest,
        'score_digests': {
            'valid': _score_payload_digest(valid_scores),
            'test': None if skip_test_evaluation else _score_payload_digest(test_scores),
        },
    }
    _atomic_json_dump(record, manifest_path)


@timeit(logger_name=TRAIN_LOGGER_NAME)
def cross_validate(args: TrainArgs,
                   train_func: Callable[
                       [TrainArgs, MoleculeDataset, int, Logger],
                       Tuple[Dict[str, List[float]], Dict[str, List[float]]],
                   ]
                   ) -> Tuple[float, float]:
    """
    Runs k-fold cross-validation.

    For each of k splits (folds) of the data, trains and tests a model on that split
    and aggregates the performance across folds.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param train_func: Function which runs training.
    :return: A tuple containing the mean and standard deviation performance across folds.
    """
    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    skip_test_evaluation = bool(getattr(args, 'skip_test_evaluation', False))
    if skip_test_evaluation and args.data_type != 'validation':
        raise ValueError('skip_test_evaluation requires data_type="validation".')
    args.task_names = get_task_names(
        path=args.data_path,
        smiles_columns=args.smiles_columns,
        target_columns=args.target_columns,
        ignore_columns=args.ignore_columns,
        loss_function=args.loss_function,
    )

    args.quantiles = [args.quantile_loss_alpha / 2] * (args.num_tasks // 2) + [1 - args.quantile_loss_alpha / 2] * (
        args.num_tasks // 2
    )

    # Print command line
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')

    # Print args
    debug('Args')
    debug(args)

    # Save args
    makedirs(args.save_dir)
    try:
        args.save(os.path.join(args.save_dir, 'args.json'))
    except subprocess.CalledProcessError:
        debug('Could not write the reproducibility section of the arguments to file, thus omitting this section.')
        args.save(os.path.join(args.save_dir, 'args.json'), with_reproducibility=False)

    # set explicit H option and reaction option
    reset_featurization_parameters(logger=logger)
    set_explicit_h(args.explicit_h)
    set_adding_hs(args.adding_h)
    set_keeping_atom_map(args.keeping_atom_map)
    if args.reaction:
        set_reaction(args.reaction, args.reaction_mode)
    elif args.reaction_solvent:
        set_reaction(True, args.reaction_mode)

    # Fingerprint the primary inputs even when the optional dataset cache is
    # disabled. This is also the data half of the resume contract below.
    dataset_manifest = _dataset_cache_manifest(args)
    resume_inputs = _resume_input_manifest(args)
    package_code_manifest = _package_code_manifest()

    # Get data
    if args.use_cache:
        cache_manifest = dataset_manifest
        cache_path = _dataset_cache_path(args, cache_manifest)
        data = None
        if os.path.isfile(cache_path):
            try:
                data = _load_dataset_cache(cache_path, cache_manifest)
            except Exception as error:
                debug(f'Ignoring unreadable dataset cache {cache_path}: {error}')

        if data is None:
            debug('Loading data and building dataset cache')
            data = get_data(
                path=args.data_path,
                args=args,
                logger=logger,
                skip_none_targets=True,
                data_weights_path=args.data_weights_path
            )
            _save_dataset_cache(cache_path, cache_manifest, data)
        else:
            debug(f'Loaded dataset cache {cache_path}')
    else:
        debug('Loading data')
        data = get_data(
            path=args.data_path,
            args=args,
            logger=logger,
            skip_none_targets=True,
            data_weights_path=args.data_weights_path
        )
    
    validate_dataset_type(data, dataset_type=args.dataset_type)
    args.features_size = data.features_size()
    # Preserve the concrete feature-source schema in checkpoints and in the
    # fold resume manifest. This distinguishes identically sized features that
    # were generated from different implementations or external files.
    args.features_source_metadata = getattr(
        data, '_features_source_metadata', None
    )
    if args.features_generator:
        selected_feature_columns = (
            load_selected_feature_columns(args.selected_features_path)
            if args.selected_features_path is not None
            else {}
        )
        args.features_generator_metadata = get_features_generators_metadata(
            args.features_generator,
            selected_feature_columns=selected_feature_columns,
            total_dimension=args.features_size,
        )
    else:
        args.features_generator_metadata = None

    if args.atom_descriptors == 'descriptor':
        args.atom_descriptors_size = data.atom_descriptors_size()
    elif args.atom_descriptors == 'feature':
        args.atom_features_size = data.atom_features_size()
        set_extra_atom_fdim(args.atom_features_size)
    if args.bond_descriptors == 'descriptor':
        args.bond_descriptors_size = data.bond_descriptors_size()
    elif args.bond_descriptors == 'feature':
        args.bond_features_size = data.bond_features_size()
        set_extra_bond_fdim(args.bond_features_size)

    debug(f'Number of tasks = {args.num_tasks}')

    if args.target_weights is not None and len(args.target_weights) != args.num_tasks:
        raise ValueError('The number of provided target weights must match the number and order of the prediction tasks')

    # Run training on different random seeds for each fold
    all_valid_scores = defaultdict(list)
    all_test_scores = defaultdict(list)
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        data.reset_features_and_targets()

        # If resuming experiment, load results from trained models
        valid_scores_path = os.path.join(args.save_dir, 'valid_scores.json')
        test_scores_path = os.path.join(args.save_dir, 'test_scores.json')
        resume_manifest_path = os.path.join(args.save_dir, 'resume_manifest.json')
        fold_manifest = _fold_resume_manifest(
            args=args,
            dataset_manifest=dataset_manifest,
            resume_inputs=resume_inputs,
            package_code_manifest=package_code_manifest,
            train_func=train_func,
            fold_num=fold_num,
        )
        resumed = False
        if args.resume_experiment:
            try:
                model_valid_scores, model_test_scores = _load_resume_scores(
                    manifest_path=resume_manifest_path,
                    valid_scores_path=valid_scores_path,
                    test_scores_path=test_scores_path,
                    expected_manifest=fold_manifest,
                    args=args,
                    skip_test_evaluation=skip_test_evaluation,
                )
                resumed = True
                info(
                    'Loading validation scores'
                    if skip_test_evaluation
                    else 'Loading validation and test scores'
                )
            except Exception as error:
                debug(
                    f'Cannot safely resume fold {fold_num}; retraining it: {error}'
                )

        # Missing, corrupt, or stale resume artifacts are never trusted.
        if not resumed:
            model_valid_scores, model_test_scores = train_func(args, data, fold_num, logger)
            if skip_test_evaluation:
                # Defensively discard results from callbacks which have not
                # yet adopted the validation-only contract.
                model_test_scores = {}

            _validate_score_payload(model_valid_scores, args)
            _validate_score_payload(
                model_test_scores, args, allow_empty=skip_test_evaluation
            )
            # Commit score files first and the digest-bearing manifest last,
            # making the manifest the transaction completion record.
            _save_resume_scores(
                manifest_path=resume_manifest_path,
                valid_scores_path=valid_scores_path,
                test_scores_path=test_scores_path,
                manifest=fold_manifest,
                valid_scores=model_valid_scores,
                test_scores=model_test_scores,
                skip_test_evaluation=skip_test_evaluation,
            )

        for metric, scores in model_valid_scores.items():
            all_valid_scores[metric].append(scores)
        for metric, scores in model_test_scores.items():
            all_test_scores[metric].append(scores)
    
    all_valid_scores = dict(all_valid_scores)
    all_test_scores = dict(all_test_scores)

    # Convert scores to numpy arrays
    for metric, scores in all_valid_scores.items():
        all_valid_scores[metric] = np.array(scores)
    for metric, scores in all_test_scores.items():
        all_test_scores[metric] = np.array(scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    contains_nan_scores = False
    for fold_num in range(args.num_folds):
        for metric, scores in all_test_scores.items():
            info(f'\tSeed {init_seed + fold_num} ==> test {metric} = '
                 f'{multitask_mean(scores=scores[fold_num], metric=metric, ignore_nan_metrics=args.ignore_nan_metrics):.6f}')

            if args.show_individual_scores:
                if args.loss_function == "quantile_interval" and metric == "quantile":
                    num_tasks = len(args.task_names) // 2
                    task_names = args.task_names[:num_tasks]
                    task_names = [f"{task_name} lower" for task_name in task_names] + [
                                  f"{task_name} upper" for task_name in task_names]
                else:
                    task_names = args.task_names

                for task_name, score in zip(task_names, scores[fold_num]):
                    info(f'\t\tSeed {init_seed + fold_num} ==> test {task_name} {metric} = {score:.6f}')
                    if np.isnan(score):
                        contains_nan_scores = True

    # Report scores across folds
    for metric, scores in all_valid_scores.items():
        avg_scores = multitask_mean(
            scores=scores,
            axis=1,
            metric=metric,
            ignore_nan_metrics=args.ignore_nan_metrics
        )  # average score for each model across tasks
        mean_score, std_score = np.mean(avg_scores), np.std(avg_scores)
        info(f'Overall valid {metric} = {mean_score:.6f} +/- {std_score:.6f}')

        if args.show_individual_scores and args.dataset_type != 'spectra':
            for task_num, task_name in enumerate(args.task_names):
                info(f'\tOverall valid {task_name} {metric} = '
                     f'{np.mean(scores[:, task_num]):.6f} +/- {np.std(scores[:, task_num]):.6f}')

    for metric, scores in all_test_scores.items():
        avg_scores = multitask_mean(
            scores=scores,
            axis=1,
            metric=metric,
            ignore_nan_metrics=args.ignore_nan_metrics
        )  # average score for each model across tasks
        mean_score, std_score = np.mean(avg_scores), np.std(avg_scores)
        info(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')

        if args.show_individual_scores and args.dataset_type != 'spectra':
            for task_num, task_name in enumerate(task_names):
                info(f'\tOverall test {task_name} {metric} = '
                     f'{np.mean(scores[:, task_num]):.6f} +/- {np.std(scores[:, task_num]):.6f}')

    if contains_nan_scores:
        info("The metric scores observed for some fold test splits contain 'nan' values. \
            This can occur when the test set does not meet the requirements \
            for a particular metric, such as having no valid instances of one \
            task in the test set or not having positive examples for some classification metrics. \
            Before v1.5.1, the default behavior was to ignore nan values in individual folds or tasks \
            and still return an overall average for the remaining folds or tasks. The behavior now \
            is to include them in the average, converting overall average metrics to 'nan' as well.")

    # Validation-only hyperparameter trials deliberately create no held-out
    # score artifact.
    if not skip_test_evaluation:
        with open(os.path.join(save_dir, TEST_SCORES_FILE_NAME), 'w') as f:
            writer = csv.writer(f)

            header = ['Task']
            for metric in args.metrics:
                header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
                          [f'Fold {i} {metric}' for i in range(args.num_folds)]
            writer.writerow(header)

            if args.dataset_type == 'spectra':  # spectra data type has only one score to report
                row = ['spectra']
                for metric in args.metrics:
                    task_scores = all_test_scores[metric][:, 0]
                    mean, std = np.mean(task_scores), np.std(task_scores)
                    row += [mean, std] + task_scores.tolist()
                writer.writerow(row)
            else:  # all other data types, separate scores by task
                if args.loss_function == "quantile_interval":
                    num_tasks = len(args.task_names) // 2
                    task_names = args.task_names[:num_tasks]
                    task_names = [f"{task_name} (lower quantile)" for task_name in task_names] + [
                                    f"{task_name} (upper quantile)" for task_name in task_names]
                else:
                    task_names = args.task_names

                for task_num, task_name in enumerate(task_names):
                    row = [task_name]
                    for metric in args.metrics:
                        task_scores = all_test_scores[metric][:, task_num]
                        mean, std = np.mean(task_scores), np.std(task_scores)
                        row += [mean, std] + task_scores.tolist()
                    writer.writerow(row)
    
    # Determine mean and std score of main metric
    if args.data_type == 'validation':
        avg_scores = multitask_mean(
            scores=all_valid_scores[args.metric],
            metric=args.metric,
            axis=1,
            ignore_nan_metrics=args.ignore_nan_metrics,
        )
        mean_score, std_score = np.mean(avg_scores), np.std(avg_scores)
    elif args.data_type == 'test':
        avg_scores = multitask_mean(
            scores=all_test_scores[args.metric],
            metric=args.metric,
            axis=1,
            ignore_nan_metrics=args.ignore_nan_metrics,
        )
        mean_score, std_score = np.mean(avg_scores), np.std(avg_scores)
    else:
        raise ValueError(f'"{args.data_type}" is not a supported score split.')

    # Optionally merge and save test preds
    if args.save_preds and not skip_test_evaluation:
        all_preds = pd.concat([pd.read_csv(os.path.join(save_dir, f'fold_{fold_num}', 'test_preds.csv'))
                                  for fold_num in range(args.num_folds)])
        all_preds.to_csv(os.path.join(save_dir, 'test_preds.csv'), index=False)

    return mean_score, std_score


def chemprop_train() -> None:
    """Parses Chemprop training arguments and trains (cross-validates) a Chemprop model.

    This is the entry point for the command line command :code:`chemprop_train`.
    """
    args = TrainArgs().parse_args()
    
    if args.model_type == 'FFN':
        cross_validate(args=args, train_func=run_training)
    elif args.model_type == 'lgbm':
        # Keep the optional LightGBM backend out of standard FFN imports.
        from .run_training_lgbm import run_training_lgbm

        cross_validate(args=args, train_func=run_training_lgbm)
