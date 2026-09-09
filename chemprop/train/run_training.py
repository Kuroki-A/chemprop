from logging import Logger
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from .loss_functions import get_loss_func
from chemprop.spectra_utils import normalize_spectra, load_phase_mask
from chemprop.args import TrainArgs
from chemprop.constants import MODEL_FILE_NAME
from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count, param_count_all
from chemprop.utils import build_optimizer, build_lr_scheduler, load_checkpoint, \
    load_args, load_checkpoint_for_training, load_scalers, makedirs, \
    save_checkpoint, save_smiles_splits, load_frzn_model, multitask_mean


_TEST_EVALUATION_COMPATIBILITY_FIELDS = (
    'model_type',
    'dataset_type',
    'task_names',
    'loss_function',
    'multiclass_num_classes',
    'number_of_molecules',
    'reaction',
    'reaction_solvent',
    'reaction_mode',
    'explicit_h',
    'adding_h',
    'keeping_atom_map',
    'features_generator',
    'features_size',
    'use_input_features',
    'atom_descriptors',
    'atom_descriptors_size',
    'atom_features_size',
    'bond_descriptors',
    'bond_descriptors_size',
    'bond_features_size',
    'overwrite_default_atom_features',
    'overwrite_default_bond_features',
    'is_atom_bond_targets',
    'atom_targets',
    'bond_targets',
    'atom_constraints',
    'bond_constraints',
    'adding_bond_types',
    'spectra_activation',
    'spectra_target_floor',
    'quantile_loss_alpha',
    'quantiles',
)


def _checkpoint_values_equal(expected: Any, actual: Any) -> bool:
    """Compares scalar, sequence, mapping, and array checkpoint metadata."""
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        try:
            return np.array_equal(
                np.asarray(expected), np.asarray(actual), equal_nan=True,
            )
        except TypeError:
            return np.array_equal(np.asarray(expected), np.asarray(actual))
    return expected == actual


def _validate_test_checkpoint_semantics(
    args: TrainArgs,
) -> List[TrainArgs]:
    """Rejects test data/model combinations which would produce bogus scores."""
    checkpoint_args = [load_args(path) for path in args.checkpoint_paths]

    # Reuse the prediction path's exhaustive model-ensemble contract.  The
    # import is intentionally lazy to avoid a module initialization cycle.
    from chemprop.train.make_predictions import (
        _FFN_ENSEMBLE_COMPATIBILITY_FIELDS,
        _validate_ensemble_train_args,
    )
    _validate_ensemble_train_args(
        checkpoint_args,
        _FFN_ENSEMBLE_COMPATIBILITY_FIELDS,
        'FFN --test',
    )

    reference = checkpoint_args[0]
    incompatible = [
        field
        for field in _TEST_EVALUATION_COMPATIBILITY_FIELDS
        if not _checkpoint_values_equal(
            getattr(reference, field, None), getattr(args, field, None),
        )
    ]
    for metadata_field in (
        'features_generator_metadata', 'features_source_metadata',
    ):
        expected = getattr(reference, metadata_field, None)
        if expected is not None and not _checkpoint_values_equal(
            expected, getattr(args, metadata_field, None),
        ):
            incompatible.append(metadata_field)
    if getattr(reference, 'dataset_type', None) == 'spectra':
        current_phase_mask = load_phase_mask(
            getattr(args, 'spectra_phase_mask_path', None),
        )
        if not _checkpoint_values_equal(
            getattr(reference, 'spectra_phase_mask', None), current_phase_mask,
        ):
            incompatible.append('spectra_phase_mask')
    if incompatible:
        raise ValueError(
            '--test data/feature semantics do not match the supplied checkpoint: '
            f'{", ".join(incompatible)}. Architecture flags such as hidden size '
            'need not be repeated, but the dataset, ordered targets, molecular '
            'inputs, and feature/descriptor configuration must match training.'
        )
    return checkpoint_args


def _scalers_equal(first: Any, second: Any) -> bool:
    """Returns whether two loaded checkpoint scalers are semantically equal."""
    if first is None or second is None:
        return first is second
    if type(first) is not type(second):
        return False
    for attribute in ('means', 'stds'):
        try:
            first_values = np.asarray(getattr(first, attribute), dtype=float)
            second_values = np.asarray(getattr(second, attribute), dtype=float)
        except (AttributeError, TypeError, ValueError):
            return False
        if (
            first_values.shape != second_values.shape
            or not np.array_equal(first_values, second_values, equal_nan=True)
        ):
            return False
    for attribute in ('n_atom_targets', 'n_bond_targets'):
        if getattr(first, attribute, None) != getattr(second, attribute, None):
            return False
    return True


def _validate_checkpoint_scaler_contract(
    checkpoint_path: str,
    checkpoint_args: TrainArgs,
    scalers: Tuple[Any, ...],
) -> None:
    """Validates scaler presence against a checkpoint's saved data semantics.

    ``load_args`` reconstructs old checkpoints through :class:`TrainArgs`, so
    attributes which predate a checkpoint receive their v1 defaults (for
    example, feature and descriptor scaling default to enabled).  A missing
    scaler which those saved semantics require cannot be recovered from
    evaluation labels without changing the model's inputs or output units.
    """
    (
        target_scaler,
        features_scaler,
        atom_descriptor_scaler,
        bond_descriptor_scaler,
        atom_bond_target_scaler,
    ) = scalers

    uses_molecular_features = bool(
        getattr(checkpoint_args, 'use_input_features', False)
    )
    scales_molecular_features = bool(
        getattr(checkpoint_args, 'features_scaling', True)
    )
    atom_descriptor_channel = getattr(
        checkpoint_args, 'atom_descriptors', None,
    )
    bond_descriptor_channel = getattr(
        checkpoint_args, 'bond_descriptors', None,
    )
    scales_atom_descriptors = bool(
        getattr(checkpoint_args, 'atom_descriptor_scaling', True)
    )
    scales_bond_descriptors = bool(
        getattr(checkpoint_args, 'bond_descriptor_scaling', True)
    )
    dataset_type = getattr(checkpoint_args, 'dataset_type', None)
    is_atom_bond_targets = bool(
        getattr(checkpoint_args, 'is_atom_bond_targets', False)
    )

    expected_presence = (
        dataset_type == 'regression' and not is_atom_bond_targets,
        uses_molecular_features and scales_molecular_features,
        atom_descriptor_channel is not None and scales_atom_descriptors,
        bond_descriptor_channel is not None and scales_bond_descriptors,
        dataset_type == 'regression' and is_atom_bond_targets,
    )
    scaler_names = (
        'target', 'molecular feature', 'atom descriptor', 'bond descriptor',
        'atom/bond target',
    )
    for scaler_name, scaler, expected in zip(
        scaler_names, scalers, expected_presence,
    ):
        present = scaler is not None
        if present == expected:
            continue
        state = 'missing' if expected else 'unexpectedly present'
        raise ValueError(
            f'--test checkpoint {checkpoint_path!r} has a {scaler_name} '
            f'scaler which is {state} for its saved dataset, feature, and '
            'descriptor configuration. The checkpoint cannot reproduce its '
            'training-time preprocessing safely.'
        )


def _load_test_checkpoint_scalers(
    checkpoint_paths: List[str],
    checkpoint_args: List[TrainArgs],
) -> Tuple[Any, ...]:
    """Loads one compatible scaler set for label-independent ``--test`` runs."""
    scaler_names = (
        'target', 'molecular feature', 'atom descriptor', 'bond descriptor',
        'atom/bond target',
    )
    reference = load_scalers(checkpoint_paths[0])
    _validate_checkpoint_scaler_contract(
        checkpoint_paths[0], checkpoint_args[0], reference,
    )
    for checkpoint_path, member_args in zip(
        checkpoint_paths[1:], checkpoint_args[1:],
    ):
        candidate = load_scalers(checkpoint_path)
        _validate_checkpoint_scaler_contract(
            checkpoint_path, member_args, candidate,
        )
        for scaler_name, first, second in zip(
            scaler_names, reference, candidate,
        ):
            if not _scalers_equal(first, second):
                raise ValueError(
                    '--test checkpoint scalers do not match across the ensemble: '
                    f'{scaler_name} scaler differs in {checkpoint_path!r}.'
                )
    return reference


def _apply_checkpoint_feature_scaler(
    datasets: List[Tuple[str, MoleculeDataset]],
    scaler: Any,
    label: str,
    **normalization_kwargs,
) -> None:
    """Applies a saved feature scaler and rejects missing required inputs."""
    if scaler is None:
        return
    for dataset_name, dataset in datasets:
        if len(dataset) == 0:
            continue
        applied_scaler = dataset.normalize_features(
            scaler, **normalization_kwargs,
        )
        if applied_scaler is None:
            raise ValueError(
                f'--test checkpoint requires {label}, but the {dataset_name} '
                'dataset did not provide them.'
            )


def _first_feature_schema_difference(expected: Any,
                                     actual: Any,
                                     path: str = 'features') -> Optional[str]:
    """Returns the first semantic feature-schema difference, if any."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            return f'{path} keys differ (missing={missing}, extra={extra})'
        for key in sorted(expected_keys):
            difference = _first_feature_schema_difference(
                expected[key], actual[key], f'{path}.{key}'
            )
            if difference is not None:
                return difference
        return None

    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return f'{path} length differs ({len(expected)} != {len(actual)})'
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            difference = _first_feature_schema_difference(
                expected_value, actual_value, f'{path}[{index}]'
            )
            if difference is not None:
                return difference
        return None

    if expected != actual:
        expected_text = repr(expected)
        actual_text = repr(actual)
        if len(expected_text) > 160:
            expected_text = expected_text[:157] + '...'
        if len(actual_text) > 160:
            actual_text = actual_text[:157] + '...'
        return f'{path} differs ({expected_text} != {actual_text})'
    return None


def validate_features_source_metadata(reference_data: MoleculeDataset,
                                      candidate_data: MoleculeDataset,
                                      candidate_name: str) -> None:
    """Ensures a separate dataset uses the training data's feature schema.

    Width-only checks cannot distinguish reordered descriptor columns, a
    feature file from phase features, or incompatible ``save_features``
    manifests. ``_features_source_metadata`` deliberately contains only
    row-independent semantic information, so it is safe to compare strictly.
    """
    expected = getattr(reference_data, '_features_source_metadata', None)
    actual = getattr(candidate_data, '_features_source_metadata', None)
    difference = _first_feature_schema_difference(expected, actual)
    if difference is not None:
        raise ValueError(
            f'The molecular feature schema for {candidate_name} does not match '
            f'the main training data: {difference}. Use the same ordered '
            'feature columns, phase-feature layout, generator configuration, '
            'and feature manifest for every split.'
        )


def _validate_training_split(args: TrainArgs,
                             train_data: MoleculeDataset,
                             val_data: MoleculeDataset,
                             require_training_labels: bool = True) -> None:
    """Rejects unusable validation or training splits.

    Evaluation-only ``--test`` runs do not optimize a model and therefore do
    not depend on the size or labels of the otherwise unused training split.
    """
    if len(val_data) == 0:
        raise ValueError(
            'The validation data split is empty. Chemprop FFN training '
            'requires validation data for model selection and early stopping.'
        )
    if not require_training_labels:
        return
    if len(train_data) == 0:
        raise ValueError(
            'The training data split is empty. Increase the data set size, '
            'change --split_sizes, or provide a non-empty training file.'
        )

    task_masks = train_data.mask()
    if len(task_masks) != args.num_tasks:
        raise ValueError(
            'The training target schema is inconsistent: '
            f'the data contain {len(task_masks)} task columns but the model '
            f'expects {args.num_tasks}.'
        )

    missing_tasks = [
        task_index
        for task_index, task_mask in enumerate(task_masks)
        if not any(bool(is_observed) for is_observed in task_mask)
    ]
    if missing_tasks:
        task_names = list(args.task_names or [])
        labels = [
            task_names[index] if index < len(task_names) else f'index {index}'
            for index in missing_tasks
        ]
        raise ValueError(
            'The training split has no observed labels for the following '
            f'task(s): {", ".join(labels)}. A prediction head with no labels '
            'would remain randomly initialized.'
        )

    if args.class_balance:
        if args.num_tasks != 1:
            raise ValueError(
                '--class_balance is supported only for single-task binary '
                'classification data.'
            )
        observed_classes = {
            datapoint.targets[0]
            for datapoint in train_data
            if datapoint.targets[0] is not None
        }
        if observed_classes != {0, 1}:
            raise ValueError(
                '--class_balance requires both binary classes in the training '
                f'split; observed classes were {sorted(observed_classes)}.'
            )


def _validate_primary_validation_score(metric: str, score: float) -> None:
    """Prevents an unusable validation split from selecting random weights."""
    if not np.isfinite(score):
        raise ValueError(
            f'The primary validation metric {metric!r} is not finite, so '
            'Chemprop cannot select a trained checkpoint. Check that the '
            'validation split has labels and enough class/sample diversity '
            'for this metric; for multitask data, --ignore_nan_metrics may '
            'be used only when at least one task still has a finite score.'
        )


def run_training(args: TrainArgs,
                 data: MoleculeDataset,
                 fold_num: int = None,
                 logger: Logger = None) -> Union[
                     Dict[str, List[float]],
                     Tuple[Dict[str, List[float]], Dict[str, List[float]]],
                 ]:
    """
    Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
    :param logger: A logger to record output.
    :return: For cross-validation callers, a ``(validation, test)`` tuple in
             which each dictionary maps every metric in :code:`args.metrics`
             to task-axis scores (one aggregate value for spectra). Legacy
             callers receive only the test dictionary.

    """
    # Backwards-compatible adapter for callers which historically passed the
    # logger as the third positional argument and expected test scores only
    # (notably the web application). Cross-validation always supplies an
    # integer fold number and receives the uniform (valid, test) result tuple.
    legacy_call = fold_num is None or not isinstance(fold_num, int)
    if legacy_call and fold_num is not None and logger is None:
        logger = fold_num

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    test_mode = bool(getattr(args, 'test', False))
    checkpoint_args = (
        _validate_test_checkpoint_semantics(args) if test_mode else None
    )

    # Hyperparameter optimization must select configurations exclusively on
    # validation scores. A separate held-out file is not even loaded in this
    # mode; an in-memory split is constructed only long enough to preserve the
    # train/validation split semantics and is then discarded.
    skip_test_evaluation = bool(getattr(args, 'skip_test_evaluation', False))

    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    test_data = MoleculeDataset([])
    if args.separate_test_path and not skip_test_evaluation:
        test_data = get_data(path=args.separate_test_path,
                             args=args,
                             target_columns=args.task_names,
                             features_path=args.separate_test_features_path,
                             atom_descriptors_path=args.separate_test_atom_descriptors_path,
                             bond_descriptors_path=args.separate_test_bond_descriptors_path,
                             phase_features_path=args.separate_test_phase_features_path,
                             constraints_path=args.separate_test_constraints_path,
                             smiles_columns=args.smiles_columns,
                             loss_function=args.loss_function,
                             logger=logger)
        validate_features_source_metadata(data, test_data, 'separate test data')
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path,
                            args=args,
                            target_columns=args.task_names,
                            features_path=args.separate_val_features_path,
                            atom_descriptors_path=args.separate_val_atom_descriptors_path,
                            bond_descriptors_path=args.separate_val_bond_descriptors_path,
                            phase_features_path=args.separate_val_phase_features_path,
                            constraints_path=args.separate_val_constraints_path,
                            smiles_columns=args.smiles_columns,
                            loss_function=args.loss_function,
                            logger=logger)
        validate_features_source_metadata(data, val_data, 'separate validation data')

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data,
                                              split_type=args.split_type,
                                              sizes=args.split_sizes,
                                              key_molecule_index=args.split_key_molecule,
                                              seed=args.seed,
                                              num_folds=args.num_folds,
                                              args=args,
                                              logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data,
                                             split_type=args.split_type,
                                             sizes=args.split_sizes,
                                             key_molecule_index=args.split_key_molecule,
                                             seed=args.seed,
                                             num_folds=args.num_folds,
                                             args=args,
                                             logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data,
                                                     split_type=args.split_type,
                                                     sizes=args.split_sizes,
                                                     key_molecule_index=args.split_key_molecule,
                                                     seed=args.seed,
                                                     num_folds=args.num_folds,
                                                     args=args,
                                                     logger=logger)

    if skip_test_evaluation:
        test_data = MoleculeDataset([])

    _validate_training_split(
        args,
        train_data,
        val_data,
        require_training_labels=not test_mode,
    )

    if args.dataset_type == 'classification' and not test_mode:
        class_sizes = get_class_sizes(train_data)
        debug('Training class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')
        train_class_sizes = get_class_sizes(train_data, proportion=False)
        args.train_class_sizes = train_class_sizes

    if args.save_smiles_splits and not skip_test_evaluation:
        save_smiles_splits(
            data_path=args.data_path,
            save_dir=args.save_dir,
            task_names=args.task_names,
            features_path=args.features_path,
            constraints_path=args.constraints_path,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            smiles_columns=args.smiles_columns,
            loss_function=args.loss_function,
            logger=logger,
        )

    checkpoint_scalers = (
        _load_test_checkpoint_scalers(args.checkpoint_paths, checkpoint_args)
        if test_mode
        else None
    )
    datasets_to_scale = [('training', train_data), ('validation', val_data)]
    if not skip_test_evaluation:
        datasets_to_scale.append(('test', test_data))

    if test_mode:
        features_scaler = checkpoint_scalers[1]
        _apply_checkpoint_feature_scaler(
            datasets_to_scale,
            features_scaler,
            'molecular features',
        )
    elif args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        if not skip_test_evaluation:
            test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    if test_mode:
        atom_descriptor_scaler = checkpoint_scalers[2]
        _apply_checkpoint_feature_scaler(
            datasets_to_scale,
            atom_descriptor_scaler,
            'atom descriptors/features',
            scale_atom_descriptors=True,
        )
    elif args.atom_descriptor_scaling and args.atom_descriptors is not None:
        atom_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_atom_descriptors=True)
        val_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        if not skip_test_evaluation:
            test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
    else:
        atom_descriptor_scaler = None

    if test_mode:
        bond_descriptor_scaler = checkpoint_scalers[3]
        _apply_checkpoint_feature_scaler(
            datasets_to_scale,
            bond_descriptor_scaler,
            'bond descriptors/features',
            scale_bond_descriptors=True,
        )
    elif args.bond_descriptor_scaling and args.bond_descriptors is not None:
        bond_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_bond_descriptors=True)
        val_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
        if not skip_test_evaluation:
            test_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
    else:
        bond_descriptor_scaler = None

    args.train_data_size = len(train_data)

    if skip_test_evaluation:
        debug(f'Total size = {len(data):,} | '
              f'train size = {len(train_data):,} | val size = {len(val_data):,}')
    else:
        debug(f'Total size = {len(data):,} | '
              f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    empty_test_set = len(test_data) == 0
    evaluate_test = not skip_test_evaluation and not empty_test_set
    if not skip_test_evaluation and empty_test_set:
        debug('The test data split is empty. This may be either because splitting with no test set was selected, \
            such as with `cv-no-test`, or because test data provided with `--separate_test_path` was empty or contained only invalid molecules. \
            Performance on the test set will not be evaluated and metric scores will return `nan` for each task.')


    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        if test_mode:
            debug('Using target scaler stored in the supplied checkpoint')
            scaler = checkpoint_scalers[0]
            atom_bond_scaler = checkpoint_scalers[4]
            if args.is_atom_bond_targets:
                if scaler is not None or atom_bond_scaler is None:
                    raise ValueError(
                        '--test checkpoint target scaler is incompatible with '
                        'atom/bond regression mode.'
                    )
            elif scaler is None or atom_bond_scaler is not None:
                raise ValueError(
                    '--test checkpoint target scaler is incompatible with '
                    'molecule-level regression mode.'
                )
        else:
            debug('Fitting scaler')
            if args.is_atom_bond_targets:
                scaler = None
                atom_bond_scaler = train_data.normalize_atom_bond_targets()
            else:
                scaler = train_data.normalize_targets()
                atom_bond_scaler = None
        args.spectra_phase_mask = None
    elif args.dataset_type == 'spectra':
        debug('Normalizing spectra and excluding spectra regions based on phase')
        args.spectra_phase_mask = load_phase_mask(args.spectra_phase_mask_path)
        datasets_to_normalize = [train_data, val_data]
        if evaluate_test:
            datasets_to_normalize.append(test_data)
        for dataset in datasets_to_normalize:
            data_targets = normalize_spectra(
                spectra=dataset.targets(),
                phase_features=dataset.phase_features(),
                phase_mask=args.spectra_phase_mask,
                excluded_sub_value=None,
                threshold=args.spectra_target_floor,
            )
            dataset.set_targets(data_targets)
        scaler = None
        atom_bond_scaler = None
    else:
        args.spectra_phase_mask = None
        scaler = None
        atom_bond_scaler = None

    # Get loss function
    loss_func = get_loss_func(args)

    # Accumulate predictions from each member's best checkpoint. Validation
    # scores must describe the final ensemble on the task axis, just like test
    # scores from every other training backend; per-member early-stopping
    # scores are not interchangeable with per-task scores.
    val_smiles, val_targets = val_data.smiles(), val_data.targets()
    if args.dataset_type == 'multiclass':
        sum_val_preds = np.zeros(
            (len(val_smiles), args.num_tasks, args.multiclass_num_classes)
        )
    elif args.is_atom_bond_targets:
        sum_val_preds = np.array(
            [
                np.zeros((np.concatenate(task_targets).shape[0], 1))
                for task_targets in zip(*val_data.targets())
            ],
            dtype=object,
        )
    else:
        sum_val_preds = np.zeros((len(val_smiles), args.num_tasks))

    # Set up held-out set evaluation only when explicitly enabled. Avoid even
    # materializing held-out targets during hyperparameter trials.
    test_targets = None
    sum_test_preds = None
    if evaluate_test:
        test_smiles, test_targets = test_data.smiles(), test_data.targets()
        if args.dataset_type == 'multiclass':
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        elif args.is_atom_bond_targets:
            sum_test_preds = []
            for tb in zip(*test_data.targets()):
                tb = np.concatenate(tb)
                sum_test_preds.append(np.zeros((tb.shape[0], 1)))
            sum_test_preds = np.array(sum_test_preds, dtype=object)
        else:
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Automatically determine whether to cache
    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

    # Create data loaders
    training_class_balance = bool(args.class_balance and not test_mode)
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        class_balance=training_class_balance,
        shuffle=not test_mode,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=num_workers
    )
    test_data_loader = None
    if evaluate_test:
        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            num_workers=num_workers
        )

    if training_class_balance:
        debug(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except TypeError:
            # tensorboardX historically used ``logdir`` while newer releases
            # accept the PyTorch-compatible ``log_dir`` spelling.
            writer = SummaryWriter(logdir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            if test_mode:
                # Test-only runs must reconstruct the architecture stored in
                # the checkpoint.  CLI architecture defaults (for example
                # hidden_size=300) are intentionally not required to repeat
                # the training configuration.
                model = load_checkpoint(
                    args.checkpoint_paths[model_idx],
                    device=args.device,
                    logger=logger,
                )
            else:
                # Continued training is a warm start into the *current*
                # architecture, which may have a different task/readout shape.
                model = load_checkpoint_for_training(
                    args.checkpoint_paths[model_idx],
                    current_args=args,
                    device=args.device,
                    logger=logger,
                )
        else:
            debug(f'Building model {model_idx}')
            model = MoleculeModel(args)

        # Optionally, overwrite weights:
        if args.checkpoint_frzn is not None:
            debug(f'Loading and freezing parameters from {args.checkpoint_frzn}.')
            model = load_frzn_model(model=model, path=args.checkpoint_frzn, current_args=args, logger=logger)

        debug(model)

        if args.checkpoint_frzn is not None:
            debug(f'Number of unfrozen parameters = {param_count(model):,}')
            debug(f'Total number of parameters = {param_count_all(model):,}')
        else:
            debug(f'Number of parameters = {param_count_all(model):,}')

        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)

        # Ensure that a checkpoint exists in the result directory even for a
        # zero-epoch run.  In --test mode the supplied checkpoint must be
        # preserved byte-for-byte: serializing its model with current CLI
        # defaults would attach incompatible architecture metadata.
        model_checkpoint_path = os.path.join(save_dir, MODEL_FILE_NAME)
        if test_mode:
            source_checkpoint_path = args.checkpoint_paths[model_idx]
            same_checkpoint = (
                os.path.exists(model_checkpoint_path)
                and os.path.samefile(source_checkpoint_path, model_checkpoint_path)
            )
            if not same_checkpoint:
                shutil.copy2(source_checkpoint_path, model_checkpoint_path)
        else:
            save_checkpoint(model_checkpoint_path, model, scaler,
                            features_scaler, atom_descriptor_scaler, bond_descriptor_scaler,
                            atom_bond_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        early_stopping_count = 0
        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')
            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                atom_bond_scaler=atom_bond_scaler,
                logger=logger,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                scaler=scaler,
                quantiles=args.quantiles,
                atom_bond_scaler=atom_bond_scaler,
                logger=logger
            )

            for metric, scores in val_scores.items():
                # Average validation score\
                mean_val_score = multitask_mean(
                    scores=scores,
                    metric=metric,
                    ignore_nan_metrics=args.ignore_nan_metrics
                )
                debug(f'Validation {metric} = {mean_val_score:.6f}')
                writer.add_scalar(f'validation_{metric}', mean_val_score, n_iter)

                if args.show_individual_scores:
                    if args.loss_function == "quantile_interval" and metric == "quantile":
                        num_tasks = len(args.task_names) // 2
                        task_names = args.task_names[:num_tasks]
                        task_names = [f"{task_name} lower" for task_name in task_names] + [
                                        f"{task_name} upper" for task_name in task_names]
                    else:
                        task_names = args.task_names
                    # Individual validation scores
                    for task_name, val_score in zip(task_names, scores):
                        debug(f'Validation {task_name} {metric} = {val_score:.6f}')
                        writer.add_scalar(f'validation_{task_name}_{metric}', val_score, n_iter)

            # Save model checkpoint if improved validation score
            mean_val_score = multitask_mean(
                scores=val_scores[args.metric],
                metric=args.metric,
                ignore_nan_metrics=args.ignore_nan_metrics
            )
            _validate_primary_validation_score(args.metric, mean_val_score)
            if args.minimize_score and mean_val_score < best_score or \
                    not args.minimize_score and mean_val_score > best_score:
                best_score, best_epoch = mean_val_score, epoch
                save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, features_scaler,
                                atom_descriptor_scaler, bond_descriptor_scaler, atom_bond_scaler, args)
                early_stopping_count = 0
            else:
                early_stopping_count += 1
                if early_stopping_count == args.early_stopping:
                    debug(f'Early stopped at epoch {epoch}')
                    break

        # Evaluate validation and held-out data with this member's best
        # checkpoint. Accumulating validation predictions here makes the
        # returned validation payload independent of ensemble size and gives
        # extra metrics the same semantics as test metrics.
        if test_mode:
            info(f'Model {model_idx}: skipped optimization and retained the supplied checkpoint.')
        else:
            info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
            model = load_checkpoint(
                model_checkpoint_path,
                device=args.device,
                logger=logger,
            )
        val_preds = predict(
            model=model,
            data_loader=val_data_loader,
            scaler=scaler,
            atom_bond_scaler=atom_bond_scaler,
        )
        if args.is_atom_bond_targets:
            sum_val_preds += np.array(val_preds, dtype=object)
        else:
            sum_val_preds += np.array(val_preds)

        # Evaluate on the held-out set using the same best checkpoint.
        if evaluate_test:
            test_preds = predict(
                model=model,
                data_loader=test_data_loader,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler
            )
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                is_atom_bond_targets=args.is_atom_bond_targets,
                gt_targets=test_data.gt_targets(),
                lt_targets=test_data.lt_targets(),
                quantiles=args.quantiles,
                logger=logger
            )

            if len(test_preds) != 0:
                if args.is_atom_bond_targets:
                    sum_test_preds += np.array(test_preds, dtype=object)
                else:
                    sum_test_preds += np.array(test_preds)

            # Average test score
            for metric, scores in test_scores.items():
                avg_test_score = np.nanmean(scores)
                info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
                writer.add_scalar(f'test_{metric}', avg_test_score, 0)

                if args.show_individual_scores and args.dataset_type != 'spectra':
                    # Individual test scores
                    for task_name, test_score in zip(task_names, scores):
                        info(f'Model {model_idx} test {task_name} {metric} = {test_score:.6f}')
                        writer.add_scalar(f'test_{task_name}_{metric}', test_score, n_iter)
        elif not skip_test_evaluation:
            info(f'Model {model_idx} provided with no test set, no metric evaluation will be performed.')
        writer.close()

    # Evaluate the final ensemble on validation data. The result is always a
    # complete ``args.metrics`` mapping whose values use the task axis (or the
    # single aggregate spectra axis), matching LightGBM and sklearn callbacks.
    avg_val_preds = (sum_val_preds / args.ensemble_size).tolist()
    ensemble_valid_scores = evaluate_predictions(
        preds=avg_val_preds,
        targets=val_targets,
        num_tasks=args.num_tasks,
        metrics=args.metrics,
        dataset_type=args.dataset_type,
        is_atom_bond_targets=args.is_atom_bond_targets,
        gt_targets=val_data.gt_targets(),
        lt_targets=val_data.lt_targets(),
        quantiles=args.quantiles,
        logger=logger,
    )
    for metric, scores in ensemble_valid_scores.items():
        mean_ensemble_valid_score = multitask_mean(
            scores=scores,
            metric=metric,
            ignore_nan_metrics=args.ignore_nan_metrics,
        )
        info(
            f'Ensemble validation {metric} = '
            f'{mean_ensemble_valid_score:.6f}'
        )

    # Evaluate ensemble on test set
    if skip_test_evaluation:
        ensemble_test_scores = {}
    elif empty_test_set:
        score_width = 1 if args.dataset_type == 'spectra' else args.num_tasks
        ensemble_test_scores = {
            metric: [np.nan] * score_width for metric in args.metrics
        }
    else:
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

        ensemble_test_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            is_atom_bond_targets=args.is_atom_bond_targets,
            gt_targets=test_data.gt_targets(),
            lt_targets=test_data.lt_targets(),
            quantiles=args.quantiles,
            logger=logger,
        )

    for metric, scores in ensemble_test_scores.items():
        # Average ensemble score
        mean_ensemble_test_score = multitask_mean(
            scores=scores,
            metric=metric,
            ignore_nan_metrics=args.ignore_nan_metrics
        )
        info(f'Ensemble test {metric} = {mean_ensemble_test_score:.6f}')

        # Individual ensemble scores
        if args.show_individual_scores:
            for task_name, ensemble_score in zip(task_names, scores):
                info(f'Ensemble test {task_name} {metric} = {ensemble_score:.6f}')

    # Optionally save test preds
    if args.save_preds and evaluate_test:
        test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})

        if args.is_atom_bond_targets:
            n_atoms, n_bonds = test_data.number_of_atoms, test_data.number_of_bonds

            for i, atom_target in enumerate(args.atom_targets):
                values = np.split(np.array(avg_test_preds[i]).flatten(), np.cumsum(np.array(n_atoms)))[:-1]
                values = [list(v) for v in values]
                test_preds_dataframe[atom_target] = values
            for i, bond_target in enumerate(args.bond_targets):
                values = np.split(np.array(avg_test_preds[i+len(args.atom_targets)]).flatten(), np.cumsum(np.array(n_bonds)))[:-1]
                values = [list(v) for v in values]
                test_preds_dataframe[bond_target] = values
        else:
            if args.loss_function == "quantile_interval":
                num_tasks = len(args.task_names) // 2
                task_names = args.task_names[:num_tasks]
                avg_test_preds = np.array(avg_test_preds)
                num_data = avg_test_preds.shape[0]
                preds = avg_test_preds.reshape(num_data, 2, num_tasks).mean(axis=1)
                intervals = abs(np.diff(avg_test_preds.reshape(num_data, 2, num_tasks), axis=1) / 2)
                intervals = intervals.reshape(num_data, num_tasks)
                for i, task_name in enumerate(task_names):
                    test_preds_dataframe[task_name] = [pred[i] for pred in preds]
                for i, task_name in enumerate(task_names):
                    task_name = f"{task_name}_{args.quantile_loss_alpha}_half_interval"
                    test_preds_dataframe[task_name] = [interval[i] for interval in intervals]
            else:
                for i, task_name in enumerate(args.task_names):
                    test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]

        test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)

    if legacy_call:
        return ensemble_test_scores

    return ensemble_valid_scores, ensemble_test_scores
