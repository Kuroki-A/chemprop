"""Training utilities for a frozen Chemprop encoder plus LightGBM heads.

The MPN is initialized once per fold, evaluated once per split, and shared by
all members of the LightGBM ensemble.  Every saved ensemble member contains
the exact MPN state and one LightGBM Booster per prediction task.
"""

from logging import Logger
import os
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .evaluate import evaluate_predictions
from .metrics import prc_auc
from .run_training import (
    _validate_primary_validation_score,
    validate_features_source_metadata,
)
from chemprop.args import TrainArgs
from chemprop.data import (
    MoleculeDataLoader,
    MoleculeDataset,
    get_class_sizes,
    get_data,
    set_cache_graph,
    split_data,
)
from chemprop.models import MoleculeModelEncoder
from chemprop.utils import (
    makedirs,
    multitask_mean,
    save_checkpoint_lgbm,
    save_smiles_splits,
)


LIGHTGBM_MODEL_FILE_NAME = "model.pkl"


def _seed_lgbm_pipeline(seed: int) -> None:
    """Seeds every RNG used while constructing the frozen encoder."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_frozen_lgbm_encoder(args: TrainArgs) -> MoleculeModelEncoder:
    """Builds the single deterministic MPN encoder used by a fold."""
    if not getattr(args, "features_only", False):
        raise ValueError(
            "LightGBM requires a deterministic --features_only representation."
        )
    if (
        getattr(args, "reaction", False)
        or getattr(args, "reaction_solvent", False)
    ) and getattr(args, "features_generator", None):
        raise NotImplementedError(
            "LightGBM molecular feature generators encode only the reactant. "
            "Use an explicitly reaction-aware --features_path for reaction data."
        )
    if (
        getattr(args, "atom_descriptors", None) is not None
        or getattr(args, "bond_descriptors", None) is not None
    ):
        raise NotImplementedError(
            "LightGBM features-only encoding does not consume atom or bond "
            "descriptors/features; provide molecule-level --features_path data."
        )
    _seed_lgbm_pipeline(args.pytorch_seed)
    encoder = MoleculeModelEncoder(args).to(args.device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def encode_lgbm_features(
    encoder: MoleculeModelEncoder,
    data: MoleculeDataset,
    batch_size: int,
    num_workers: int,
    show_progress: bool = False,
) -> np.ndarray:
    """Runs a frozen encoder over a dataset without building autograd graphs."""
    if len(data) == 0:
        return np.empty((0, 0), dtype=np.float32)

    # MPN.forward returns molecule-level input features immediately in
    # features-only mode. Avoid DataLoader collation, RDKit graph construction,
    # device transfer, and a no-op model call for this recommended LightGBM
    # configuration. The float32 cast exactly matches torch.Tensor.float().
    if getattr(encoder.encoder, "features_only", False):
        features = data.features()
        if features is None:
            raise ValueError(
                "LightGBM features-only encoding requires molecular features."
            )
        try:
            encoded = np.stack(features).astype(np.float32, copy=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "LightGBM molecular features must have a consistent numeric width."
            ) from error
        if encoded.ndim != 2 or encoded.shape[1] < 1:
            raise ValueError(
                "LightGBM molecular features must be a non-empty 2-D matrix."
            )
        return encoded

    data_loader = MoleculeDataLoader(
        dataset=data,
        batch_size=max(1, min(batch_size, len(data))),
        num_workers=num_workers,
        shuffle=False,
    )
    encoded_batches = []
    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), leave=False, disable=not show_progress):
            encoded = encoder(
                batch.batch_graph(),
                batch.features(),
                batch.atom_descriptors(),
                batch.atom_features(),
                batch.bond_descriptors(),
                batch.bond_features(),
            )
            encoded_batches.append(encoded.detach().cpu().numpy())

    return np.concatenate(encoded_batches, axis=0)


def _targets_to_array(
    targets: Sequence[Sequence[float]], num_tasks: int
) -> np.ndarray:
    """Converts Chemprop targets (including ``None``) to a 2-D float array."""
    if len(targets) == 0:
        return np.empty((0, num_tasks), dtype=float)
    try:
        target_array = np.asarray(targets, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("LightGBM targets must form a rectangular numeric matrix.") from error
    if target_array.ndim == 1:
        target_array = target_array.reshape(-1, 1)
    if target_array.ndim != 2 or target_array.shape[1] != num_tasks:
        actual_columns = target_array.shape[1] if target_array.ndim == 2 else None
        raise ValueError(
            f"Expected {num_tasks} LightGBM target columns, got {actual_columns}."
        )
    return target_array


def _validate_lgbm_targets(
    targets: Sequence[Sequence[float]],
    args: TrainArgs,
    split_name: str,
    require_each_task: bool = False,
) -> np.ndarray:
    """Validates target values before scaling or handing them to LightGBM."""
    target_array = _targets_to_array(targets, args.num_tasks)
    if np.any(np.isinf(target_array)):
        raise ValueError(
            f"LightGBM {split_name} targets contain an infinite value; "
            "use a finite value or leave a missing target blank."
        )

    task_names = list(
        getattr(args, "task_names", None)
        or [f"task_{task_index}" for task_index in range(args.num_tasks)]
    )
    if len(task_names) != args.num_tasks:
        raise ValueError("LightGBM task names do not match the target width.")
    for task_index, task_name in enumerate(task_names):
        present = np.isfinite(target_array[:, task_index])
        if require_each_task and not np.any(present):
            raise ValueError(
                f'LightGBM task "{task_name}" has no {split_name} targets.'
            )
        if args.dataset_type == "classification" and np.any(present):
            labels = set(target_array[present, task_index].tolist())
            if not labels.issubset({0.0, 1.0}):
                raise ValueError(
                    f'LightGBM classification task "{task_name}" contains '
                    f"{split_name} labels other than 0 and 1."
                )
    return target_array


def _validate_lgbm_feature_matrices(
    train_features: np.ndarray, val_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validates the dense encoder matrices used to construct LightGBM datasets."""
    train_features = np.asarray(train_features)
    val_features = np.asarray(val_features)
    if train_features.ndim != 2 or val_features.ndim != 2:
        raise ValueError("LightGBM encoded features must be two-dimensional matrices.")
    if train_features.shape[1] < 1:
        raise ValueError("LightGBM training features contain no columns.")
    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError(
            "LightGBM training and validation feature widths do not match."
        )
    if not np.all(np.isfinite(train_features)) or not np.all(np.isfinite(val_features)):
        raise ValueError(
            "LightGBM encoded features contain NaN or infinite values. Check "
            "external features/descriptors and their scaling settings."
        )
    return train_features, val_features


def _lightgbm_metric(args: TrainArgs) -> str:
    """Returns the exact native metric, or disables it for a custom metric."""
    if args.dataset_type == "classification":
        return {
            "auc": "auc",
            "binary_cross_entropy": "binary_logloss",
            "prc-auc": "None",
        }[args.metric]
    return {"rmse": "rmse", "mae": "l1", "mse": "l2"}[args.metric]


def _lightgbm_feval(args: TrainArgs) -> Optional[Callable]:
    """Builds an exact Chemprop metric callback when no native metric matches."""
    if args.dataset_type != "classification" or args.metric != "prc-auc":
        return None

    # LightGBM's native ``average_precision`` is the step-integrated average
    # precision score. Chemprop's historical ``prc-auc`` contract instead uses
    # trapezoidal integration over the precision-recall curve, so it needs an
    # explicit callback rather than an approximately named native metric.
    def chemprop_prc_auc(predictions: np.ndarray, dataset: lgb.Dataset):
        labels = dataset.get_label()
        if len(np.unique(labels)) < 2:
            score = float("nan")
        else:
            score = float(prc_auc(labels, predictions))
        return "prc-auc", score, True

    return chemprop_prc_auc


def _lightgbm_params(args: TrainArgs, seed: int) -> Dict[str, object]:
    """Creates deterministic parameters for one task Booster."""
    learning_rate = getattr(args, "lgbm_learning_rate", 0.05)
    num_leaves = getattr(args, "lgbm_num_leaves", 31)
    feature_fraction = getattr(args, "lgbm_feature_fraction", 0.8)
    bagging_fraction = getattr(args, "lgbm_bagging_fraction", 0.8)
    bagging_freq = getattr(args, "lgbm_bagging_freq", 1)
    min_data_in_leaf = getattr(args, "lgbm_min_data_in_leaf", 20)
    num_threads = getattr(args, "lgbm_num_threads", None) or max(1, args.num_workers)
    finite_number = lambda value: (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )
    if not finite_number(learning_rate) or learning_rate <= 0:
        raise ValueError("lgbm_learning_rate must be finite and greater than 0.")
    if not isinstance(num_leaves, int) or isinstance(num_leaves, bool) or num_leaves < 2:
        raise ValueError("lgbm_num_leaves must be an integer of at least 2.")
    if not finite_number(feature_fraction) or not 0 < feature_fraction <= 1:
        raise ValueError("lgbm_feature_fraction must be finite and in (0, 1].")
    if not finite_number(bagging_fraction) or not 0 < bagging_fraction <= 1:
        raise ValueError("lgbm_bagging_fraction must be finite and in (0, 1].")
    if not isinstance(bagging_freq, int) or isinstance(bagging_freq, bool) or bagging_freq < 0:
        raise ValueError("lgbm_bagging_freq must be a non-negative integer.")
    if not isinstance(min_data_in_leaf, int) or isinstance(min_data_in_leaf, bool) or min_data_in_leaf < 1:
        raise ValueError("lgbm_min_data_in_leaf must be a positive integer.")
    if not isinstance(num_threads, int) or isinstance(num_threads, bool) or num_threads < 1:
        raise ValueError("lgbm_num_threads must be a positive integer.")

    return {
        "objective": "binary" if args.dataset_type == "classification" else "regression",
        "metric": _lightgbm_metric(args),
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "min_data_in_leaf": min_data_in_leaf,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": num_threads,
        "verbosity": -1 if args.quiet else 0,
    }


def train_task_boosters(
    args: TrainArgs,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    seed: int,
    train_weights: Sequence[float] = None,
) -> List[lgb.Booster]:
    """Trains one Booster per task while masking each task's missing labels."""
    num_boost_round = getattr(args, "lgbm_num_boost_round", 500)
    early_stopping_rounds = getattr(args, "lgbm_early_stopping_rounds", 30)
    if not isinstance(num_boost_round, int) or num_boost_round < 1:
        raise ValueError("lgbm_num_boost_round must be a positive integer.")
    if not isinstance(early_stopping_rounds, int) or early_stopping_rounds < 0:
        raise ValueError("lgbm_early_stopping_rounds must be a non-negative integer.")
    train_features, val_features = _validate_lgbm_feature_matrices(
        train_features, val_features
    )
    train_targets = _validate_lgbm_targets(
        train_targets, args, "training", require_each_task=True
    )
    val_targets = _validate_lgbm_targets(val_targets, args, "validation")
    if train_features.shape[0] != train_targets.shape[0]:
        raise ValueError("LightGBM training features and targets have different lengths.")
    if val_features.shape[0] != val_targets.shape[0]:
        raise ValueError("LightGBM validation features and targets have different lengths.")

    base_weights = (
        np.ones(train_features.shape[0], dtype=float)
        if train_weights is None
        else np.asarray(train_weights, dtype=float)
    )
    if base_weights.shape != (train_features.shape[0],):
        raise ValueError("LightGBM data weights must contain one value per training row.")
    if not np.all(np.isfinite(base_weights)):
        raise ValueError("LightGBM data weights must be finite.")
    if np.any(base_weights < 0):
        raise ValueError("LightGBM data weights must be non-negative.")
    if float(base_weights.sum()) <= 0:
        raise ValueError("At least one LightGBM data weight must be positive.")

    boosters = []
    for task_index in range(args.num_tasks):
        train_mask = np.isfinite(train_targets[:, task_index])
        task_labels = train_targets[train_mask, task_index]
        task_weights = base_weights[train_mask].copy()
        if float(task_weights.sum()) <= 0:
            raise ValueError(
                f'LightGBM task "{args.task_names[task_index]}" has no '
                "positive-weight training targets."
            )
        if args.dataset_type == "classification":
            unique_labels = set(np.unique(task_labels).tolist())
            if args.class_balance and unique_labels == {0.0, 1.0}:
                negative_count = np.count_nonzero(task_labels == 0)
                positive_count = np.count_nonzero(task_labels == 1)
                task_weights[task_labels == 0] *= len(task_labels) / (2 * negative_count)
                task_weights[task_labels == 1] *= len(task_labels) / (2 * positive_count)

        train_set = lgb.Dataset(
            train_features[train_mask],
            label=task_labels,
            weight=task_weights,
            free_raw_data=False,
        )
        val_mask = np.isfinite(val_targets[:, task_index])
        valid_sets = []
        callbacks = [lgb.log_evaluation(period=0)]
        # AUC and PRC-AUC are undefined for a single-class validation task.
        # LightGBM reports a misleading native AUC of 1.0 in that situation,
        # while a NaN custom PRC-AUC causes meaningless early stopping. Train
        # the requested fixed number of rounds instead, matching Chemprop's
        # post-training convention of reporting NaN for those task metrics.
        rank_metric_without_both_classes = (
            args.dataset_type == "classification"
            and args.metric in {"auc", "prc-auc"}
            and np.any(val_mask)
            and len(np.unique(val_targets[val_mask, task_index])) < 2
        )
        if np.any(val_mask) and not rank_metric_without_both_classes:
            valid_sets.append(
                lgb.Dataset(
                    val_features[val_mask],
                    label=val_targets[val_mask, task_index],
                    reference=train_set,
                    free_raw_data=False,
                )
            )
            if early_stopping_rounds > 0:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=early_stopping_rounds,
                        verbose=not args.quiet,
                    )
                )

        boosters.append(
            lgb.train(
                params=_lightgbm_params(args, seed + task_index),
                train_set=train_set,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets or None,
                valid_names=["validation"] if valid_sets else None,
                feval=_lightgbm_feval(args),
                callbacks=callbacks,
            )
        )

    return boosters


def predict_task_boosters(
    boosters: Sequence[lgb.Booster], features: np.ndarray
) -> np.ndarray:
    """Predicts an ``(examples, tasks)`` matrix from per-task Boosters."""
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError("LightGBM prediction features must be a two-dimensional matrix.")
    if not boosters:
        raise ValueError("LightGBM prediction requires at least one task Booster.")
    if features.shape[0] == 0:
        return np.empty((0, len(boosters)), dtype=float)
    try:
        features_are_finite = np.all(np.isfinite(features))
    except TypeError as error:
        raise ValueError("LightGBM prediction features must be numeric.") from error
    if not features_are_finite:
        raise ValueError(
            "LightGBM prediction features contain NaN or infinite values. Check "
            "external features/descriptors and their scaling settings."
        )
    expected_widths = {booster.num_feature() for booster in boosters}
    if expected_widths != {features.shape[1]}:
        raise ValueError(
            "LightGBM prediction feature width does not match the trained Booster."
        )
    predictions = []
    for booster in boosters:
        best_iteration = booster.best_iteration if booster.best_iteration > 0 else None
        predictions.append(booster.predict(features, num_iteration=best_iteration))
    return np.column_stack(predictions)


def _inverse_target_scaling(
    predictions: np.ndarray, scaler
) -> np.ndarray:
    return predictions if scaler is None else np.asarray(scaler.inverse_transform(predictions), dtype=float)


def evaluate_lgbm_predictions(
    predictions: np.ndarray,
    targets: Sequence[Sequence[float]],
    args: TrainArgs,
    logger: Logger = None,
    gt_targets: Sequence[Sequence[bool]] = None,
    lt_targets: Sequence[Sequence[bool]] = None,
) -> Dict[str, List[float]]:
    """Evaluates every requested metric and preserves all task positions."""
    results = {metric: [] for metric in args.metrics}
    if predictions.shape[0] == 0:
        return {metric: [float("nan")] * args.num_tasks for metric in args.metrics}

    target_array = _validate_lgbm_targets(targets, args, "evaluation")
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != target_array.shape:
        raise ValueError(
            "LightGBM prediction and target matrices must have the same shape."
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError("LightGBM predictions contain NaN or infinite values.")
    for task_index in range(args.num_tasks):
        task_targets = target_array[:, task_index]
        present = np.isfinite(task_targets)
        if not np.any(present):
            for metric in args.metrics:
                results[metric].append(float("nan"))
            continue

        task_gt = None
        task_lt = None
        if gt_targets is not None:
            task_gt = [[row[task_index]] for row, keep in zip(gt_targets, present) if keep]
        if lt_targets is not None:
            task_lt = [[row[task_index]] for row, keep in zip(lt_targets, present) if keep]
        task_results = evaluate_predictions(
            preds=predictions[present, task_index].reshape(-1, 1).tolist(),
            targets=task_targets[present].reshape(-1, 1).tolist(),
            num_tasks=1,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            gt_targets=task_gt,
            lt_targets=task_lt,
            logger=logger,
        )
        for metric in args.metrics:
            values = task_results.get(metric, [])
            results[metric].append(values[0] if values else float("nan"))
    return results


def _load_split_data(
    args: TrainArgs, data: MoleculeDataset, logger: Logger
) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    """Applies Chemprop's standard split and separate-set rules."""
    skip_test_evaluation = bool(getattr(args, "skip_test_evaluation", False))
    val_data = None
    test_data = MoleculeDataset([])
    if args.separate_test_path and not skip_test_evaluation:
        test_data = get_data(
            path=args.separate_test_path,
            args=args,
            target_columns=args.task_names,
            features_path=args.separate_test_features_path,
            atom_descriptors_path=args.separate_test_atom_descriptors_path,
            bond_descriptors_path=args.separate_test_bond_descriptors_path,
            phase_features_path=args.separate_test_phase_features_path,
            constraints_path=args.separate_test_constraints_path,
            smiles_columns=args.smiles_columns,
            loss_function=args.loss_function,
            logger=logger,
        )
        validate_features_source_metadata(data, test_data, "separate test data")
    if args.separate_val_path:
        val_data = get_data(
            path=args.separate_val_path,
            args=args,
            target_columns=args.task_names,
            features_path=args.separate_val_features_path,
            atom_descriptors_path=args.separate_val_atom_descriptors_path,
            bond_descriptors_path=args.separate_val_bond_descriptors_path,
            phase_features_path=args.separate_val_phase_features_path,
            constraints_path=args.separate_val_constraints_path,
            smiles_columns=args.smiles_columns,
            loss_function=args.loss_function,
            logger=logger,
        )
        validate_features_source_metadata(data, val_data, "separate validation data")

    if args.separate_val_path and args.separate_test_path:
        return data, val_data, test_data
    if args.separate_val_path:
        train_data, _, split_test = split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            key_molecule_index=args.split_key_molecule,
            seed=args.seed,
            num_folds=args.num_folds,
            args=args,
            logger=logger,
        )
        return train_data, val_data, (
            MoleculeDataset([]) if skip_test_evaluation else split_test
        )
    if args.separate_test_path:
        train_data, split_val, _ = split_data(
            data=data,
            split_type=args.split_type,
            sizes=args.split_sizes,
            key_molecule_index=args.split_key_molecule,
            seed=args.seed,
            num_folds=args.num_folds,
            args=args,
            logger=logger,
        )
        return train_data, split_val, test_data
    train_data, val_data, test_data = split_data(
        data=data,
        split_type=args.split_type,
        sizes=args.split_sizes,
        key_molecule_index=args.split_key_molecule,
        seed=args.seed,
        num_folds=args.num_folds,
        args=args,
        logger=logger,
    )
    if skip_test_evaluation:
        test_data = MoleculeDataset([])
    return train_data, val_data, test_data


def run_training_lgbm(
    args: TrainArgs,
    data: MoleculeDataset,
    fold_num: int,
    logger: Logger = None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Trains and saves a deterministic, multitask LightGBM ensemble."""
    debug = logger.debug if logger is not None else print
    info = logger.info if logger is not None else print

    if args.dataset_type not in {"classification", "regression"}:
        raise ValueError("LightGBM supports only classification and regression datasets.")
    if not getattr(args, "features_only", False):
        raise ValueError(
            "LightGBM requires --features_only because its MPN encoder is not "
            "trained. Provide deterministic molecular features, for example "
            "--features_generator morgan --features_only, or use --features_path "
            "together with --features_only."
        )
    if args.is_atom_bond_targets:
        raise NotImplementedError("LightGBM does not support atom/bond target mode.")

    debug(f"Splitting data with seed {args.seed}")
    train_data, val_data, test_data = _load_split_data(args, data, logger)
    if len(train_data) == 0:
        raise ValueError("The LightGBM training data split is empty.")
    if len(val_data) == 0:
        raise ValueError("The LightGBM validation data split is empty.")
    empty_test_set = len(test_data) == 0
    skip_test_evaluation = getattr(args, "skip_test_evaluation", False)

    if skip_test_evaluation:
        debug(
            f"Total size = {len(data):,} | train size = {len(train_data):,} | "
            f"val size = {len(val_data):,} | test evaluation skipped"
        )
    else:
        debug(
            f"Total size = {len(data):,} | train size = {len(train_data):,} | "
            f"val size = {len(val_data):,} | test size = {len(test_data):,}"
        )
    if empty_test_set and not skip_test_evaluation:
        info("LightGBM was provided with no test set; test metrics will be NaN.")

    if args.dataset_type == "classification":
        debug("Class sizes")
        class_size_data = train_data if skip_test_evaluation else data
        for task_name, task_sizes in zip(args.task_names, get_class_sizes(class_size_data)):
            debug(
                f'{task_name} '
                + ", ".join(
                    f"{label}: {size * 100:.2f}%" for label, size in enumerate(task_sizes)
                )
            )
        args.train_class_sizes = get_class_sizes(train_data, proportion=False)

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
            logger=logger,
        )

    # Preserve unscaled evaluation targets before normalizing regression targets.
    train_targets = train_data.targets()
    val_targets = val_data.targets()
    val_gt_targets, val_lt_targets = val_data.gt_targets(), val_data.lt_targets()
    if skip_test_evaluation:
        test_targets = []
        test_gt_targets = test_lt_targets = None
    else:
        test_targets = test_data.targets()
        test_gt_targets, test_lt_targets = test_data.gt_targets(), test_data.lt_targets()

    # Check raw values before regression scaling can turn a single infinity
    # into NaNs across an entire task column.
    _validate_lgbm_targets(
        train_targets, args, "training", require_each_task=True
    )
    _validate_lgbm_targets(val_targets, args, "validation")
    if not skip_test_evaluation:
        _validate_lgbm_targets(test_targets, args, "test")

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        if not skip_test_evaluation and len(test_data) > 0:
            test_data.normalize_features(features_scaler)
    else:
        features_scaler = None
    if args.atom_descriptor_scaling and args.atom_descriptors is not None:
        atom_descriptor_scaler = train_data.normalize_features(
            replace_nan_token=0, scale_atom_descriptors=True
        )
        val_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        if not skip_test_evaluation and len(test_data) > 0:
            test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
    else:
        atom_descriptor_scaler = None
    if args.bond_descriptor_scaling and args.bond_descriptors is not None:
        bond_descriptor_scaler = train_data.normalize_features(
            replace_nan_token=0, scale_bond_descriptors=True
        )
        val_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
        if not skip_test_evaluation and len(test_data) > 0:
            test_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
    else:
        bond_descriptor_scaler = None

    if args.dataset_type == "regression":
        scaler = train_data.normalize_targets()
        val_data.set_targets(scaler.transform(val_targets).tolist())
        if not skip_test_evaluation and len(test_data) > 0:
            test_data.set_targets(scaler.transform(test_targets).tolist())
    else:
        scaler = None
    atom_bond_scaler = None
    args.spectra_phase_mask = None
    args.train_data_size = len(train_data)

    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

    # The expensive MPN pass is done exactly once per fold/split and reused by
    # every ensemble member.  Only the inexpensive LightGBM heads vary by seed.
    encoder = build_frozen_lgbm_encoder(args)
    train_features = encode_lgbm_features(encoder, train_data, args.batch_size, num_workers)
    val_features = encode_lgbm_features(encoder, val_data, args.batch_size, num_workers)
    test_features = (
        np.empty((0, train_features.shape[1]), dtype=train_features.dtype)
        if skip_test_evaluation
        else encode_lgbm_features(encoder, test_data, args.batch_size, num_workers)
    )
    train_target_array = _targets_to_array(train_data.targets(), args.num_tasks)
    val_target_array = _targets_to_array(val_data.targets(), args.num_tasks)
    train_weights = train_data.data_weights()

    val_prediction_sum = np.zeros((len(val_data), args.num_tasks), dtype=float)
    test_prediction_sum = np.zeros(
        (0 if skip_test_evaluation else len(test_data), args.num_tasks), dtype=float
    )
    for model_index in range(args.ensemble_size):
        model_seed = args.seed + model_index
        model_dir = os.path.abspath(os.path.join(args.save_dir, f"model_{model_index}"))
        makedirs(model_dir)
        boosters = train_task_boosters(
            args=args,
            train_features=train_features,
            train_targets=train_target_array,
            val_features=val_features,
            val_targets=val_target_array,
            seed=model_seed,
            train_weights=train_weights,
        )
        val_prediction_sum += _inverse_target_scaling(
            predict_task_boosters(boosters, val_features), scaler
        )
        if not empty_test_set and not skip_test_evaluation:
            test_prediction_sum += _inverse_target_scaling(
                predict_task_boosters(boosters, test_features), scaler
            )

        save_checkpoint_lgbm(
            path=os.path.join(model_dir, LIGHTGBM_MODEL_FILE_NAME),
            encoder=encoder,
            task_boosters=boosters,
            scaler=scaler,
            features_scaler=features_scaler,
            atom_descriptor_scaler=atom_descriptor_scaler,
            bond_descriptor_scaler=bond_descriptor_scaler,
            atom_bond_scaler=atom_bond_scaler,
            args=args,
            model_index=model_index,
            seed=model_seed,
        )
    val_predictions = val_prediction_sum / args.ensemble_size
    test_predictions = test_prediction_sum / args.ensemble_size
    valid_scores = evaluate_lgbm_predictions(
        val_predictions,
        val_targets,
        args,
        logger,
        gt_targets=val_gt_targets,
        lt_targets=val_lt_targets,
    )
    test_scores = (
        {}
        if skip_test_evaluation
        else evaluate_lgbm_predictions(
            test_predictions,
            test_targets,
            args,
            logger,
            gt_targets=test_gt_targets,
            lt_targets=test_lt_targets,
        )
    )

    for metric in args.metrics:
        valid_mean = multitask_mean(
            valid_scores[metric],
            metric,
            ignore_nan_metrics=args.ignore_nan_metrics,
        )
        if metric == args.metric:
            _validate_primary_validation_score(metric, valid_mean)
        info(f"Ensemble validation {metric} = {valid_mean:.6f}")
        if not skip_test_evaluation:
            test_mean = multitask_mean(
                test_scores[metric],
                metric,
                ignore_nan_metrics=args.ignore_nan_metrics,
            )
            info(f"Ensemble test {metric} = {test_mean:.6f}")

    if args.save_preds and not skip_test_evaluation:
        prediction_columns = {
            column: [smiles[row_index] for smiles in test_data.smiles()]
            for row_index, column in enumerate(args.smiles_columns)
        }
        prediction_columns.update(
            {
                task_name: test_predictions[:, task_index]
                for task_index, task_name in enumerate(args.task_names)
            }
        )
        pd.DataFrame(prediction_columns).to_csv(
            os.path.join(args.save_dir, "test_preds.csv"), index=False
        )

    return valid_scores, test_scores
