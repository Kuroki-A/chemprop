from collections import defaultdict
import logging
from numbers import Integral
from typing import Dict, List

import numpy as np

from .predict import predict
from chemprop.data import MoleculeDataLoader, StandardScaler, AtomBondScaler
from chemprop.models import MoleculeModel
from chemprop.train import get_metric_func


def _require_finite_prediction(value, location: str, allow_empty: bool = False) -> None:
    """Rejects malformed/non-finite model output before metric computation."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{location} must be numeric.') from exc
    if (array.size == 0 and not allow_empty) or not np.isfinite(array).all():
        raise ValueError(f'{location} is empty or contains NaN/infinity.')


def _validate_optional_target(value, location: str) -> None:
    """Validates one scalar target while preserving missing ``None`` labels."""
    if value is None:
        return
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{location} must be a numeric scalar or None.') from exc
    if not np.isfinite(numeric_value):
        raise ValueError(f'{location} contains NaN or infinity.')


def _transpose_atom_bond_rows(rows, num_tasks: int, name: str):
    """Validates and concatenates per-molecule atom/bond task arrays."""
    task_values = [[] for _ in range(num_tasks)]
    for row_index, row in enumerate(rows):
        if len(row) != num_tasks:
            raise ValueError(
                f'{name} row {row_index} has {len(row)} tasks; expected {num_tasks}.'
            )
        for task_index, values in enumerate(row):
            task_values[task_index].append(np.asarray(values, dtype=object).reshape(-1))
    return [
        np.concatenate(values).reshape(-1, 1)
        if values
        else np.empty((0, 1), dtype=object)
        for values in task_values
    ]


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metrics: List[str],
                         dataset_type: str,
                         is_atom_bond_targets: bool = False,
                         gt_targets: List[List[bool]] = None,
                         lt_targets: List[List[bool]] = None,
                         quantiles: List[float] = None,
                         logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates predictions using a metric function after filtering out invalid targets.

    :param preds: A list of lists of shape :code:`(data_size, num_tasks)` with model predictions.
    :param targets: A list of lists of shape :code:`(data_size, num_tasks)` with targets.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param is_atom_bond_targets: Boolean whether this is atomic/bond properties prediction.
    :param gt_targets: A list of lists of booleans indicating whether the target is an inequality rather than a single value.
    :param lt_targets: A list of lists of booleans indicating whether the target is an inequality rather than a single value.
    :param quantiles: A list of quantiles for use in pinball evaluation of quantile_interval metric.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.
    """
    if not isinstance(num_tasks, Integral) or isinstance(num_tasks, bool) or num_tasks <= 0:
        raise ValueError(f'num_tasks must be a positive integer; got {num_tasks!r}.')
    if not metrics:
        raise ValueError('At least one evaluation metric is required.')
    if 'quantile' in metrics:
        if quantiles is None or len(quantiles) != num_tasks:
            raise ValueError(
                f'quantile metric evaluation requires exactly {num_tasks} quantiles.'
            )
        if not np.isfinite(np.asarray(quantiles, dtype=float)).all():
            raise ValueError('Evaluation quantiles must be finite.')

    info = logger.info if logger is not None else print

    metric_to_func = {metric: get_metric_func(metric) for metric in metrics}

    if len(preds) == 0:
        if len(targets) != 0:
            raise ValueError('Predictions are empty but targets are not.')
        return {metric: [float('nan')] * num_tasks for metric in metrics}

    if is_atom_bond_targets:
        if len(preds) != num_tasks:
            raise ValueError(
                f'Atom/bond predictions contain {len(preds)} tasks; expected '
                f'{num_tasks}.'
            )
        targets = _transpose_atom_bond_rows(targets, num_tasks, 'Target')
        if gt_targets is not None:
            gt_targets = _transpose_atom_bond_rows(
                gt_targets, num_tasks, 'Greater-than target mask'
            )
        if lt_targets is not None:
            lt_targets = _transpose_atom_bond_rows(
                lt_targets, num_tasks, 'Less-than target mask'
            )
        for task_index, (task_preds, task_targets) in enumerate(zip(preds, targets)):
            pred_array = np.asarray(task_preds)
            if pred_array.ndim != 2 or pred_array.shape[0] != len(task_targets):
                raise ValueError(
                    f'Atom/bond prediction task {task_index} has shape '
                    f'{pred_array.shape}; expected a 2D array with '
                    f'{len(task_targets)} rows.'
                )
            _require_finite_prediction(
                pred_array,
                f'Atom/bond prediction task {task_index}',
                allow_empty=len(task_targets) == 0,
            )
            for target_index, target in enumerate(task_targets[:, 0]):
                _validate_optional_target(
                    target, f'Atom/bond target task {task_index}, row {target_index}'
                )
    else:
        if len(preds) != len(targets):
            raise ValueError(
                f'Prediction row count {len(preds)} does not match target row '
                f'count {len(targets)}.'
            )
        for row_index, (pred_row, target_row) in enumerate(zip(preds, targets)):
            if len(pred_row) != num_tasks:
                raise ValueError(
                    f'Prediction row {row_index} has {len(pred_row)} tasks; '
                    f'expected {num_tasks}.'
                )
            if len(target_row) != num_tasks:
                raise ValueError(
                    f'Target row {row_index} has {len(target_row)} tasks; '
                    f'expected {num_tasks}.'
                )
            for task_index, (prediction, target) in enumerate(
                zip(pred_row, target_row)
            ):
                location = f'Prediction row {row_index}, task {task_index}'
                if dataset_type == 'spectra' and target is None:
                    try:
                        masked_prediction = np.asarray(prediction, dtype=float)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f'{location} must be numeric.') from exc
                    if masked_prediction.size == 0 or np.isinf(masked_prediction).any():
                        raise ValueError(
                            f'{location} is empty or contains infinity.'
                        )
                else:
                    _require_finite_prediction(prediction, location)
                _validate_optional_target(
                    target, f'Target row {row_index}, task {task_index}'
                )

        for mask_name, mask_rows in (
            ('Greater-than target mask', gt_targets),
            ('Less-than target mask', lt_targets),
        ):
            if mask_rows is None:
                continue
            if len(mask_rows) != len(targets) or any(
                len(row) != num_tasks for row in mask_rows
            ):
                raise ValueError(
                    f'{mask_name} shape must match the target matrix.'
                )

    # Filter out empty targets for most data types, excluding dataset_type spectra
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    valid_gt_targets = [[] for _ in range(num_tasks)]
    valid_lt_targets = [[] for _ in range(num_tasks)]
    if dataset_type != 'spectra':
        for i in range(num_tasks):
            if is_atom_bond_targets:
                for j in range(len(preds[i])):
                    if targets[i][j][0] is not None:  # Skip those without targets
                        valid_preds[i].append(list(preds[i][j]))
                        valid_targets[i].append(list(targets[i][j]))
                        if gt_targets is not None:
                            valid_gt_targets[i].append(list(gt_targets[i][j]))
                        if lt_targets is not None:
                            valid_lt_targets[i].append(list(lt_targets[i][j]))
            else:
                for j in range(len(preds)):
                    if targets[j][i] is not None:  # Skip those without targets
                        valid_preds[i].append(preds[j][i])
                        valid_targets[i].append(targets[j][i])
                        if gt_targets is not None:
                            valid_gt_targets[i].append(gt_targets[j][i])
                        if lt_targets is not None:
                            valid_lt_targets[i].append(lt_targets[j][i])

    # Compute metric. Spectra loss calculated for all tasks together, others calculated for tasks individually.
    results = defaultdict(list)
    if dataset_type == 'spectra':
        for metric, metric_func in metric_to_func.items():
            results[metric].append(metric_func(preds, targets))
    elif is_atom_bond_targets:
        for metric, metric_func in metric_to_func.items():
            if metric == 'quantile':
                for i, (valid_target, valid_pred) in enumerate(zip(valid_targets, valid_preds)):
                    if len(valid_target) == 0:
                        results[metric].append(float('nan'))
                        continue
                    valid_target = np.concatenate(valid_target)
                    valid_pred = np.concatenate(valid_pred)
                    results[metric].append(metric_func(valid_target, valid_pred, quantiles[i]))
            else:
                for valid_target, valid_pred in zip(valid_targets, valid_preds):
                    if len(valid_target) == 0:
                        results[metric].append(float('nan'))
                    else:
                        results[metric].append(metric_func(valid_target, valid_pred))
    else:
        for i in range(num_tasks):
            if len(valid_targets[i]) == 0:
                # Preserve the task axis even when this split has no labels for
                # a task. Downstream fold aggregation and CSV column ordering
                # rely on every metric containing exactly ``num_tasks`` values.
                for metric in metrics:
                    results[metric].append(float('nan'))
                continue

            # Only rank-based binary metrics require both target classes.
            # Accuracy and loss metrics remain well-defined for a single-class
            # split and must not be discarded along with AUC metrics. Constant
            # predictions are likewise valid inputs for every supported metric.
            single_class = (
                dataset_type == 'classification'
                and len(set(valid_targets[i])) < 2
            )
            if single_class:
                info('Warning: Found a classification task with only one target class; '
                     'AUC metrics will be reported as nan')

            for metric, metric_func in metric_to_func.items():
                if single_class and metric in {'auc', 'prc-auc'}:
                    results[metric].append(float('nan'))
                elif dataset_type == 'classification' and metric == 'cross_entropy':
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i], labels=[0, 1]))
                elif dataset_type == 'multiclass' and metric == 'cross_entropy':
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i],
                                                    labels=list(range(len(valid_preds[i][0])))))
                elif metric in ['bounded_rmse', 'bounded_mse', 'bounded_mae']:
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i], valid_gt_targets[i], valid_lt_targets[i]))
                elif metric == 'quantile':
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i], quantiles[i]))
                else:
                    results[metric].append(metric_func(valid_targets[i], valid_preds[i]))

    results = dict(results)

    return results


def evaluate(model: MoleculeModel,
             data_loader: MoleculeDataLoader,
             num_tasks: int,
             metrics: List[str],
             dataset_type: str,
             scaler: StandardScaler = None,
             quantiles: List[float] = None,
             atom_bond_scaler: AtomBondScaler = None,
             logger: logging.Logger = None) -> Dict[str, List[float]]:
    """
    Evaluates an ensemble of models on a dataset by making predictions and then evaluating the predictions.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param num_tasks: Number of tasks.
    :param metrics: A list of names of metric functions.
    :param dataset_type: Dataset type.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :param quantiles: A list of quantiles for use in pinball evaluation of quantile_interval metric.
    :param atom_bond_scaler: A :class:`~chemprop.data.scaler.AtomBondScaler` fitted on the atomic/bond targets.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.

    """
    # Inequality targets only need for evaluation of certain regression metrics
    if any(m in metrics for m in ['bounded_rmse', 'bounded_mse', 'bounded_mae']):
        gt_targets = data_loader.gt_targets
        lt_targets = data_loader.lt_targets
    else:
        gt_targets = None
        lt_targets = None

    preds = predict(
        model=model,
        data_loader=data_loader,
        scaler=scaler,
        atom_bond_scaler=atom_bond_scaler,
    )

    results = evaluate_predictions(
        preds=preds,
        targets=data_loader.targets,
        num_tasks=num_tasks,
        metrics=metrics,
        dataset_type=dataset_type,
        is_atom_bond_targets=model.is_atom_bond_targets,
        logger=logger,
        gt_targets=gt_targets,
        lt_targets=lt_targets,
        quantiles=quantiles,
    )

    return results
