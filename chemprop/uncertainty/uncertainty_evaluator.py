from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy.stats import t, spearmanr
from scipy.special import erfinv

from chemprop.multitask_utils import (
    flatten_atom_bond_value_sets,
    flatten_atom_bond_values,
    validate_task_masks,
)
from chemprop.uncertainty.uncertainty_calibrator import UncertaintyCalibrator


def evaluate_predictions(*args, **kwargs):
    """Imports the evaluator lazily to avoid ``train``/``uncertainty`` cycles."""
    from chemprop.train.evaluate import evaluate_predictions as train_evaluate_predictions

    return train_evaluate_predictions(*args, **kwargs)


def _atom_bond_evaluation_arrays(mask, **row_major_values):
    """Returns aligned task-major arrays for variable-size atom/bond rows."""
    task_masks = validate_task_masks(mask)
    lengths = [len(task_mask) for task_mask in task_masks]
    task_values = flatten_atom_bond_value_sets(
        row_major_values,
        num_tasks=len(task_masks),
        expected_lengths=lengths,
    )
    return task_masks, task_values


class UncertaintyEvaluator(ABC):
    """
    A class for evaluating the effectiveness of uncertainty estimates with metrics.
    """

    def __init__(
        self,
        evaluation_method: str,
        calibration_method: str,
        uncertainty_method: str,
        dataset_type: str,
        loss_function: str,
        calibrator: UncertaintyCalibrator,
        is_atom_bond_targets: bool,
    ):
        self.evaluation_method = evaluation_method
        self.calibration_method = calibration_method
        self.uncertainty_method = uncertainty_method
        self.dataset_type = dataset_type
        self.loss_function = loss_function
        self.calibrator = calibrator
        self.is_atom_bond_targets = is_atom_bond_targets

        self.raise_argument_errors()

    def raise_argument_errors(self):
        """
        Raise errors for incompatibilities between dataset type and uncertainty method, or similar.
        """
        if self.dataset_type == "spectra":
            raise ValueError(
                "No uncertainty evaluators implemented for spectra dataset type."
            )
        if self.uncertainty_method in ["ensemble", "dropout"] and self.dataset_type in [
            "classification",
            "multiclass",
        ]:
            raise ValueError(
                "Though ensemble and dropout uncertainty methods are available for classification \
                    multiclass dataset types, their outputs are not confidences and are not \
                    compatible with any implemented evaluation methods for classification."
            )
        if self.uncertainty_method == "dirichlet":
            raise ValueError(
                "The Dirichlet uncertainty method returns an evidential uncertainty value rather than a \
                    class confidence. It is not compatible with any implemented evaluation methods. \
                    To evaluate the performance of a model trained using the Dirichlet loss function, \
                    use the classification uncertainty method in a separate job."
            )

    @abstractmethod
    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ) -> List[float]:
        """
        Evaluate the performance of uncertainty predictions against the model target values.

        :param targets:  The target values for prediction.
        :param preds: The prediction values of a model on the test set.
        :param uncertainties: The estimated uncertainty values, either calibrated or uncalibrated, of a model on the test set.
        :param mask: Whether the values in targets were provided.

        :return: A list of metric values for each model task.
        """


class MetricEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating confidence estimates of classification and multiclass datasets using builtin evaluation metrics.
    """

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        task_masks = validate_task_masks(mask)
        if self.is_atom_bond_targets:
            task_masks, task_values = _atom_bond_evaluation_arrays(
                task_masks,
                targets=targets,
                uncertainties=uncertainties,
            )
            task_uncertainties = task_values['uncertainties']
            metric_preds = [values.reshape(-1, 1) for values in task_uncertainties]
        else:
            metric_preds = uncertainties
        return evaluate_predictions(
            preds=metric_preds,
            targets=targets,
            num_tasks=len(task_masks),
            metrics=[self.evaluation_method],
            dataset_type=self.dataset_type,
            is_atom_bond_targets=self.is_atom_bond_targets,
        )[self.evaluation_method]


class NLLRegressionEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating regression uncertainty values using the mean negative-log-likelihood
    of the actual targets given the probability distributions estimated by the model.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "NLL Regression Evaluator is only for regression dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        if self.calibrator is None:  # uncalibrated regression uncertainties are variances
            if self.is_atom_bond_targets:
                mask, task_values = _atom_bond_evaluation_arrays(
                    mask,
                    uncertainties=uncertainties,
                    preds=preds,
                    targets=targets,
                )
                uncertainties = task_values['uncertainties']
                preds = task_values['preds']
                targets = task_values['targets']
            else:
                uncertainties = np.asarray(uncertainties, dtype=float)
                preds = np.asarray(preds, dtype=float)
                targets = np.asarray(targets, dtype=float)
                mask = np.asarray(mask, dtype=bool)
                uncertainties = np.array(list(zip(*uncertainties)))
                preds = np.array(list(zip(*preds)))
                targets = np.array(list(zip(*targets)))
            num_tasks = len(mask)
            nll = []
            for i in range(num_tasks):
                task_mask = mask[i]
                task_unc = uncertainties[i][task_mask]
                task_preds = preds[i][task_mask]
                task_targets = targets[i][task_mask]
                if task_unc.size == 0:
                    nll.append(float('nan'))
                    continue
                if (
                    not np.all(np.isfinite(task_unc))
                    or not np.all(np.isfinite(task_preds))
                    or not np.all(np.isfinite(task_targets))
                    or np.any(task_unc < 0)
                ):
                    raise ValueError(
                        'Regression NLL expects finite predictions and targets '
                        'and finite non-negative variances.'
                    )
                task_unc = np.maximum(task_unc, np.finfo(float).eps)
                task_nll = np.log(2 * np.pi * task_unc) / 2 \
                    + (task_preds - task_targets) ** 2 / (2 * task_unc)
                nll.append(task_nll.mean())
            return nll
        else:
            nll = self.calibrator.nll(
                preds=preds, unc=uncertainties, targets=targets, mask=mask
            )  # shape(data, task)
            return nll


class NLLClassEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating classification uncertainty values using the mean negative-log-likelihood
    of the actual targets given the probabilities assigned to them by the model.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "classification":
            raise ValueError(
                "NLL Classification Evaluator is only for classification dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        if self.is_atom_bond_targets:
            mask, task_values = _atom_bond_evaluation_arrays(
                mask,
                uncertainties=uncertainties,
                targets=targets,
            )
            uncertainties = task_values['uncertainties']
            targets = task_values['targets']
        else:
            targets = np.asarray(targets, dtype=float)
            mask = np.asarray(mask, dtype=bool)
            uncertainties = np.asarray(uncertainties, dtype=float)
            uncertainties = np.array(list(zip(*uncertainties)))
            targets = np.array(list(zip(*targets)))
        num_tasks = len(mask)
        nll = []
        for i in range(num_tasks):
            task_mask = mask[i]
            task_unc = uncertainties[i][task_mask]
            task_targets = targets[i][task_mask]
            if task_unc.size == 0:
                nll.append(float('nan'))
                continue
            if (
                not np.all(np.isfinite(task_unc))
                or not np.all(np.isfinite(task_targets))
                or np.any((task_unc < 0) | (task_unc > 1))
                or np.any((task_targets != 0) & (task_targets != 1))
            ):
                raise ValueError(
                    'Classification NLL expects finite binary targets and '
                    'probabilities in [0, 1].'
                )
            task_likelihood = task_unc * task_targets + (1 - task_unc) * (1 - task_targets)
            task_likelihood = np.maximum(task_likelihood, np.finfo(float).eps)
            task_nll = -1 * np.log(task_likelihood)
            nll.append(task_nll.mean())
        return nll


class NLLMultiEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating multiclass uncertainty values using the mean negative-log-likelihood
    of the actual targets given the probabilities assigned to them by the model.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "multiclass":
            raise ValueError(
                "NLL Multiclass Evaluator is only for multiclass dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        targets = np.asarray(targets, dtype=float)  # shape(data, tasks)
        mask = np.asarray(mask, dtype=bool)  # shape(tasks, data)
        uncertainties = np.asarray(uncertainties, dtype=float)
        if targets.ndim != 2 or uncertainties.ndim != 3:
            raise ValueError(
                'Multiclass NLL expects targets with shape (data, tasks) and '
                'probabilities with shape (data, tasks, classes).'
            )
        if uncertainties.shape[:2] != targets.shape or mask.shape != targets.T.shape:
            raise ValueError('Multiclass NLL target, probability, and mask shapes do not match.')
        num_tasks = targets.shape[1]
        nll = []
        for i in range(num_tasks):
            task_mask = mask[i]
            task_preds = uncertainties[task_mask, i]
            task_target_values = targets[task_mask, i]
            if task_target_values.size == 0:
                nll.append(float('nan'))
                continue
            if (
                not np.all(np.isfinite(task_target_values))
                or np.any(task_target_values != np.floor(task_target_values))
            ):
                raise ValueError('Multiclass targets must be finite integer class indices.')
            task_targets = task_target_values.astype(int)
            if (
                not np.all(np.isfinite(task_preds))
                or np.any((task_preds < 0) | (task_preds > 1))
            ):
                raise ValueError('Multiclass probabilities must be finite and in [0, 1].')
            if np.any(task_targets < 0) or np.any(task_targets >= task_preds.shape[1]):
                raise ValueError('Multiclass targets must be valid class indices.')
            bin_targets = np.zeros_like(task_preds)  # shape(data, classes)
            bin_targets[np.arange(task_targets.shape[0]), task_targets] = 1
            task_likelihood = np.sum(bin_targets * task_preds, axis=1)
            task_likelihood = np.maximum(task_likelihood, np.finfo(float).eps)
            task_nll = -1 * np.log(task_likelihood)
            nll.append(task_nll.mean())
        return nll


class CalibrationAreaEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating regression uncertainty values based on how they deviate from perfect
    calibration on an observed-probability versus expected-probability plot.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Miscalibration area is only implemented for regression dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        if self.is_atom_bond_targets:
            mask, task_values = _atom_bond_evaluation_arrays(
                mask,
                uncertainties=uncertainties,
                targets=targets,
                preds=preds,
            )
            uncertainties = task_values['uncertainties']
            targets = task_values['targets']
            preds = task_values['preds']
        else:
            targets = np.asarray(targets, dtype=float)
            mask = np.asarray(mask, dtype=bool)
            uncertainties = np.asarray(uncertainties, dtype=float)
            preds = np.asarray(preds, dtype=float)
            uncertainties = np.array(list(zip(*uncertainties)))
            targets = np.array(list(zip(*targets)))
            preds = np.array(list(zip(*preds)))
        num_tasks = len(mask)
        # using 101 bin edges, hardcoded
        fractions = np.zeros([num_tasks, 101])  # shape(tasks, 101)
        fractions[:, 100] = 1

        if self.calibrator is not None:
            original_metric = self.calibrator.regression_calibrator_metric
            original_scaling = self.calibrator.scaling
            original_interval = self.calibrator.interval_percentile

            bin_scaling = [0]

            try:
                for i in range(1, 100):
                    self.calibrator.regression_calibrator_metric = "interval"
                    self.calibrator.interval_percentile = i
                    self.calibrator.calibrate()
                    bin_scaling.append(self.calibrator.scaling)

                for j in range(num_tasks):
                    task_mask = mask[j]
                    task_targets = targets[j][task_mask]
                    task_preds = preds[j][task_mask]
                    task_error = np.abs(task_preds - task_targets)
                    task_unc = uncertainties[j][task_mask]
                    if task_unc.size == 0:
                        fractions[j] = np.nan
                        continue

                    for i in range(1, 100):
                        bin_unc = task_unc / original_scaling[j] * bin_scaling[i][j]
                        bin_fraction = np.mean(bin_unc >= task_error)
                        fractions[j, i] = bin_fraction
            finally:
                # Evaluation must never leave the shared calibrator configured
                # for the last temporary percentile if a calculation fails.
                self.calibrator.regression_calibrator_metric = original_metric
                self.calibrator.scaling = original_scaling
                self.calibrator.interval_percentile = original_interval

        else:  # uncertainties are uncalibrated variances
            bin_scaling = [0]
            for i in range(1, 100):
                bin_scaling.append(erfinv(i / 100) * np.sqrt(2))
            for j in range(num_tasks):
                task_mask = mask[j]
                task_targets = targets[j][task_mask]
                task_preds = preds[j][task_mask]
                task_error = np.abs(task_preds - task_targets)
                task_unc = uncertainties[j][task_mask]
                if task_unc.size == 0:
                    fractions[j] = np.nan
                    continue
                if (
                    not np.all(np.isfinite(task_unc))
                    or not np.all(np.isfinite(task_error))
                    or np.any(task_unc < 0)
                ):
                    raise ValueError(
                        'Miscalibration area expects finite errors and '
                        'finite non-negative variances.'
                    )
                for i in range(1, 100):
                    bin_unc = np.sqrt(task_unc) * bin_scaling[i]
                    bin_fraction = np.mean(bin_unc >= task_error)
                    fractions[j, i] = bin_fraction

        # trapezoid rule
        auce = np.sum(
            0.01 * np.abs(fractions - np.expand_dims(np.arange(101) / 100, axis=0)),
            axis=1,
        )
        return auce.tolist()


class ExpectedNormalizedErrorEvaluator(UncertaintyEvaluator):
    """
    A class that evaluates uncertainty performance by binning together clusters of predictions
    and comparing the average predicted variance of the clusters against the RMSE of the cluster.
    Method discussed in https://doi.org/10.1021/acs.jcim.9b00975.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Expected normalized error is only appropriate for regression dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        if self.is_atom_bond_targets:
            mask, task_values = _atom_bond_evaluation_arrays(
                mask,
                uncertainties=uncertainties,
                targets=targets,
                preds=preds,
            )
            uncertainties = task_values['uncertainties']
            targets = task_values['targets']
            preds = task_values['preds']
        else:
            targets = np.asarray(targets, dtype=float)
            mask = np.asarray(mask, dtype=bool)
            uncertainties = np.asarray(uncertainties, dtype=float)
            preds = np.asarray(preds, dtype=float)
            uncertainties = np.array(list(zip(*uncertainties)))
            targets = np.array(list(zip(*targets)))
            preds = np.array(list(zip(*preds)))
        num_tasks = len(mask)
        # get stdev scaling then revert if interval
        if self.calibrator is not None:
            original_metric = self.calibrator.regression_calibrator_metric
            original_scaling = self.calibrator.scaling
            if (
                self.calibration_method != "tscaling"
                and self.calibrator.regression_calibrator_metric == "interval"
            ):
                self.calibrator.regression_calibrator_metric = "stdev"
                self.calibrator.calibrate()
                stdev_scaling = self.calibrator.scaling
                self.calibrator.regression_calibrator_metric = original_metric
                self.calibrator.scaling = original_scaling

        ence = []

        for i in range(num_tasks):
            task_mask = mask[i]  # shape(data)
            task_targets = targets[i][task_mask]
            task_preds = preds[i][task_mask]
            task_error = np.abs(task_preds - task_targets)
            task_unc = uncertainties[i][task_mask]

            if task_unc.size == 0:
                ence.append(float('nan'))
                continue
            if not np.all(np.isfinite(task_unc)) or not np.all(np.isfinite(task_error)):
                raise ValueError('ENCE inputs must contain only finite observed values.')
            if np.any(task_unc < 0):
                raise ValueError('ENCE uncertainties must be non-negative.')

            sort_idx = np.argsort(task_unc)
            task_unc = task_unc[sort_idx]
            task_error = task_error[sort_idx]

            # Use at most one bin per observation. The historical fixed 100
            # bins produced 0/0 and NaN whenever a task had fewer than 100
            # labelled rows because most bins were empty.
            num_bins = min(100, task_unc.size)
            split_unc = np.array_split(task_unc, num_bins)
            split_error = np.array_split(task_error, num_bins)
            root_mean_vars = np.empty(num_bins, dtype=float)
            rmses = np.empty(num_bins, dtype=float)

            for j in range(num_bins):
                if self.calibrator is None:  # starts as a variance
                    root_mean_vars[j] = np.sqrt(np.mean(split_unc[j]))
                    rmses[j] = np.sqrt(np.mean(np.square(split_error[j])))
                elif self.calibration_method == "tscaling":  # convert back to sample stdev
                    bin_unc = split_unc[j] / original_scaling[i]
                    bin_var = t.var(df=self.calibrator.num_models - 1, scale=bin_unc)
                    root_mean_vars[j] = np.sqrt(np.mean(bin_var))
                    rmses[j] = np.sqrt(np.mean(np.square(split_error[j])))
                else:
                    bin_unc = split_unc[j]
                    if self.calibrator.regression_calibrator_metric == "interval":
                        bin_unc = bin_unc / original_scaling[i] * stdev_scaling[i]  # convert from interval to stdev as needed
                    root_mean_vars[j] = np.sqrt(np.mean(np.square(bin_unc)))
                    rmses[j] = np.sqrt(np.mean(np.square(split_error[j])))

            if not np.all(np.isfinite(root_mean_vars)) or not np.all(np.isfinite(rmses)):
                raise ValueError('ENCE calculation produced non-finite bin statistics.')
            denominator = np.maximum(root_mean_vars, np.finfo(float).eps)
            ence.append(float(np.mean(np.abs(root_mean_vars - rmses) / denominator)))

        return ence


class SpearmanEvaluator(UncertaintyEvaluator):
    """
    Class evaluating uncertainty performance using the spearman rank correlation. Method produces
    better scores (closer to 1 in the [-1, 1] range) when the uncertainty values are predictive
    of the ranking of prediciton errors.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Spearman rank correlation is only appropriate for regression dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        spearman_coeffs = []
        if self.is_atom_bond_targets:
            mask, task_values = _atom_bond_evaluation_arrays(
                mask,
                uncertainties=uncertainties,
                targets=targets,
                preds=preds,
            )
            uncertainties = task_values['uncertainties']
            targets = task_values['targets']
            preds = task_values['preds']
        else:
            targets = np.asarray(targets, dtype=float)
            uncertainties = np.asarray(uncertainties, dtype=float)
            mask = np.asarray(mask, dtype=bool)
            preds = np.asarray(preds, dtype=float)
            uncertainties = np.array(list(zip(*uncertainties)))
            targets = np.array(list(zip(*targets)))
            preds = np.array(list(zip(*preds)))
        num_tasks = len(mask)
        for i in range(num_tasks):
            task_mask = mask[i]
            task_unc = uncertainties[i][task_mask]
            task_targets = targets[i][task_mask]
            task_preds = preds[i][task_mask]
            task_error = np.abs(task_preds - task_targets)
            if task_unc.size < 2 or np.ptp(task_unc) == 0 or np.ptp(task_error) == 0:
                spmn = float('nan')
            elif not np.all(np.isfinite(task_unc)) or not np.all(np.isfinite(task_error)):
                raise ValueError('Spearman inputs must contain only finite observed values.')
            else:
                spmn = spearmanr(task_unc, task_error).correlation
            spearman_coeffs.append(spmn)
        return spearman_coeffs


class ConformalRegressionEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating the coverage of conformal regression intervals.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Conformal Regression Evaluator is only for regression dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        """
        Args:
            targets: shape(data, tasks)
            preds: shape(data, tasks)
            uncertainties: shape(data, tasks)
            mask: shape(data, tasks)

        Returns:
            Conformal coverage for each task
        """
        if self.is_atom_bond_targets:
            mask, task_values = _atom_bond_evaluation_arrays(
                mask,
                uncertainties=uncertainties,
                targets=targets,
                preds=preds,
            )
            uncertainties = task_values['uncertainties']
            targets = task_values['targets']
            preds = task_values['preds']
        else:
            uncertainties = np.asarray(uncertainties, dtype=float)
            targets = np.asarray(targets, dtype=float)
            preds = np.asarray(preds, dtype=float)
            mask = np.asarray(mask, dtype=bool)
            uncertainties = np.array(list(zip(*uncertainties)))
            targets = np.array(list(zip(*targets)))
            preds = np.array(list(zip(*preds)))
        num_tasks = len(mask)

        results = []
        for i in range(num_tasks):
            task_mask = mask[i]
            task_unc = uncertainties[i][task_mask]
            task_targets = targets[i][task_mask]
            task_preds = preds[i][task_mask]
            unc_task_lower = task_preds - task_unc
            unc_task_upper = task_preds + task_unc
            task_results = np.logical_and(unc_task_lower <= task_targets, task_targets <= unc_task_upper)
            results.append(
                float(np.mean(task_results)) if task_results.size else float('nan')
            )

        return results


class ConformalMulticlassEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating the coverage of conformal prediction on multiclass datasets.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "multiclass":
            raise ValueError(
                "Conformal Multiclass Evaluator is only for multiclass dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        """
        Args:
            targets: shape(data, tasks)
            preds: shape(data, tasks, num_classes)
            uncertainties: shape(data, tasks, num_classes)
            mask: shape(data, tasks)

        Returns:
            Conformal coverage for each task
        """
        targets = np.array(targets, dtype=float)
        mask = np.array(mask, dtype=bool)
        uncertainties = np.array(uncertainties)
        num_tasks = targets.shape[1]
        results = []

        for i in range(num_tasks):
            task_mask = mask[i]
            task_results = np.take_along_axis(
                uncertainties[task_mask, i], targets[task_mask, i].reshape(-1, 1).astype(int), axis=1
            ).squeeze(1)
            results.append(
                float(np.mean(task_results)) if task_results.size else float('nan')
            )

        return results


class ConformalMultilabelEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating the coverage of conformal prediction on multilabel datasets.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "classification":
            raise ValueError(
                "Conformal Multilabel Evaluator is only for classification dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        """
        Args:
            targets: shape(data, tasks)
            preds: shape(data, tasks)
            uncertainties: shape(data, tasks)
            mask: shape(data, tasks)

        Returns:
            Conformal coverage for each task
        """
        if self.is_atom_bond_targets:
            task_masks = validate_task_masks(mask)
            num_tasks = len(task_masks)
            lengths = [len(task_mask) for task_mask in task_masks]
            targets = flatten_atom_bond_values(
                targets,
                num_tasks=num_tasks,
                label='targets',
                expected_lengths=lengths,
            )
            uncertainties = flatten_atom_bond_values(
                uncertainties,
                num_tasks=2 * num_tasks,
                label='uncertainties',
                expected_lengths=lengths + lengths,
            )
            mask = task_masks
        else:
            targets = np.asarray(targets, dtype=float)
            uncertainties = np.asarray(uncertainties, dtype=float)
            mask = np.asarray(mask, dtype=bool)
            num_tasks = len(mask)
            uncertainties = np.array(list(zip(*uncertainties)))
            targets = np.array(list(zip(*targets)))
        results = []
        for i in range(num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_unc_in = uncertainties[i][task_mask]
            task_unc_out = uncertainties[i + num_tasks][task_mask]
            task_results = np.logical_and(task_unc_in <= task_targets, task_targets <= task_unc_out)
            results.append(
                float(np.mean(task_results)) if task_results.size else float('nan')
            )

        return results


def build_uncertainty_evaluator(
    evaluation_method: str,
    calibration_method: str,
    uncertainty_method: str,
    dataset_type: str,
    loss_function: str,
    calibrator: UncertaintyCalibrator,
    is_atom_bond_targets: bool,
) -> UncertaintyEvaluator:
    """
    Function that chooses and returns the appropriate :class: `UncertaintyEvaluator` subclass
    for the provided arguments.
    """
    supported_evaluators = {
        "nll": {
            "regression": NLLRegressionEvaluator,
            "classification": NLLClassEvaluator,
            "multiclass": NLLMultiEvaluator,
            "spectra": None,
        }[dataset_type],
        "miscalibration_area": CalibrationAreaEvaluator,
        "ence": ExpectedNormalizedErrorEvaluator,
        "spearman": SpearmanEvaluator,
        "conformal_coverage": {
            "regression": ConformalRegressionEvaluator,
            "multiclass": ConformalMulticlassEvaluator,
            "classification": ConformalMultilabelEvaluator,
        }.get(dataset_type),
    }

    classification_metrics = [
        "auc",
        "prc-auc",
        "accuracy",
        "binary_cross_entropy",
        "f1",
        "mcc",
    ]
    multiclass_metrics = ["cross_entropy", "accuracy", "f1", "mcc"]
    if dataset_type == "classification" and evaluation_method in classification_metrics:
        evaluator_class = MetricEvaluator
    elif dataset_type == "multiclass" and evaluation_method in multiclass_metrics:
        evaluator_class = MetricEvaluator
    else:
        evaluator_class = supported_evaluators.get(evaluation_method, None)

    if evaluator_class is None:
        raise NotImplementedError(
            f"Evaluator type {evaluation_method} is not supported. Available options are all calibration/multiclass metrics and {list(supported_evaluators.keys())}"
        )
    else:
        evaluator = evaluator_class(
            evaluation_method=evaluation_method,
            calibration_method=calibration_method,
            uncertainty_method=uncertainty_method,
            dataset_type=dataset_type,
            loss_function=loss_function,
            calibrator=calibrator,
            is_atom_bond_targets=is_atom_bond_targets,
        )
        return evaluator
