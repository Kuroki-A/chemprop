from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np
from chemprop.data.data import MoleculeDataLoader
from scipy.special import erfinv, softmax, logit, expit
from scipy.optimize import fmin
from scipy.stats import t
from sklearn.isotonic import IsotonicRegression

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel
from chemprop.uncertainty.uncertainty_predictor import build_uncertainty_predictor, UncertaintyPredictor
from chemprop.multitask_utils import (
    flatten_atom_bond_value_sets,
    flatten_atom_bond_values,
    reshape_values,
    validate_task_masks,
)


def _atom_bond_calibration_arrays(
    calibration_data: MoleculeDataset, **row_major_values
):
    """Returns aligned flattened task arrays for variable-size targets."""
    masks = validate_task_masks(calibration_data.mask())
    lengths = [len(task_mask) for task_mask in masks]
    task_values = flatten_atom_bond_value_sets(
        row_major_values,
        num_tasks=len(masks),
        expected_lengths=lengths,
    )
    return masks, task_values


def _observed_regression_calibration_task(
    method: str,
    task_index: int,
    predictions,
    targets,
    variances,
    mask,
):
    """Validates and selects one task used to fit a variance calibrator."""
    task_predictions = np.asarray(predictions, dtype=float).reshape(-1)
    task_targets = np.asarray(targets, dtype=float).reshape(-1)
    task_variances = np.asarray(variances, dtype=float).reshape(-1)
    task_mask = np.asarray(mask, dtype=bool).reshape(-1)
    lengths = {
        len(task_predictions), len(task_targets), len(task_variances), len(task_mask)
    }
    if len(lengths) != 1:
        raise ValueError(
            f'{method} calibration task {task_index} has misaligned '
            'predictions, targets, variances, and mask.'
        )

    task_predictions = task_predictions[task_mask]
    task_targets = task_targets[task_mask]
    task_variances = task_variances[task_mask]
    if task_targets.size == 0:
        raise ValueError(
            f'{method} calibration task {task_index} has no observed targets.'
        )
    if (
        not np.all(np.isfinite(task_predictions))
        or not np.all(np.isfinite(task_targets))
        or not np.all(np.isfinite(task_variances))
        or np.any(task_variances <= 0)
    ):
        raise ValueError(
            f'{method} calibration task {task_index} requires finite '
            'predictions and targets and finite, strictly positive variances.'
        )
    return task_predictions, task_targets, task_variances


def _conformal_quantile(scores, alpha: float) -> float:
    """Returns the finite-sample corrected conformal quantile."""
    if (
        not isinstance(alpha, (int, float, np.integer, np.floating))
        or isinstance(alpha, (bool, np.bool_))
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError('conformal_alpha must be finite and in the range (0, 1).')
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError('Every conformal calibration task needs observed targets.')
    if not np.all(np.isfinite(scores)):
        raise ValueError('Conformal calibration scores must be finite.')
    quantile_level = min(
        1.0,
        np.ceil((scores.size + 1) * (1 - alpha)) / scores.size,
    )
    return float(np.quantile(scores, quantile_level, method='higher'))


class UncertaintyCalibrator(ABC):
    """
    Uncertainty calibrator class. Subclasses for each uncertainty calibration
    method. Subclasses should override the calibrate and apply functions for
    implemented metrics.
    """

    def __init__(
        self,
        uncertainty_method: str,
        calibration_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        calibration_data_loader: MoleculeDataLoader,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        num_models: int,
        dataset_type: str,
        loss_function: str,
        uncertainty_dropout_p: float,
        conformal_alpha: float,
        dropout_sampling_size: int,
        spectra_phase_mask: List[List[bool]],
    ):
        self.calibration_data = calibration_data
        self.calibration_data_loader = calibration_data_loader
        self.regression_calibrator_metric = regression_calibrator_metric
        self.interval_percentile = interval_percentile
        self.dataset_type = dataset_type
        self.uncertainty_method = uncertainty_method
        self.calibration_method = calibration_method
        self.loss_function = loss_function
        self.num_models = num_models
        self.conformal_alpha = conformal_alpha

        self.raise_argument_errors()

        self.calibration_predictor = build_uncertainty_predictor(
            test_data=calibration_data,
            test_data_loader=calibration_data_loader,
            models=models,
            scalers=scalers,
            num_models=num_models,
            dataset_type=dataset_type,
            loss_function=loss_function,
            uncertainty_method=uncertainty_method,
            uncertainty_dropout_p=uncertainty_dropout_p,
            conformal_alpha=conformal_alpha,
            dropout_sampling_size=dropout_sampling_size,
            individual_ensemble_predictions=False,
            spectra_phase_mask=spectra_phase_mask,
        )

        self.calibrate()

    @property
    @abstractmethod
    def label(self):
        """
        The string in saved results indicating the uncertainty method used.
        """

    def raise_argument_errors(self):
        """
        Raise errors for incompatibilities between dataset type and uncertainty method, or similar.
        """
        if self.dataset_type == "spectra":
            raise ValueError(
                "No uncertainty calibrators are implemented for the spectra dataset type."
            )
        if self.uncertainty_method in ["ensemble", "dropout"] and self.dataset_type in ["classification", "multiclass"]:
            raise ValueError(
                "Though ensemble and dropout uncertainty methods are available for classification \
                    multiclass dataset types, their outputs are not confidences and are not \
                    compatible with any implemented calibration methods for classification."
            )
        if self.uncertainty_method == "dirichlet":
            raise ValueError(
                "The Dirichlet uncertainty method returns an evidential uncertainty value rather than a \
                    class confidence. It is not compatible with any implemented calibration methods. \
                    To calibrate a model trained using the Dirichlet loss function, \
                    use the classification uncertainty method."
            )

    @abstractmethod
    def calibrate(self):
        """
        Fit calibration method for the calibration data.
        """

    @abstractmethod
    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        """
        Take in predictions and uncertainty parameters from a model and apply the calibration method using fitted parameters.
        """

    @abstractmethod
    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ) -> List[float]:
        """
        Takes in calibrated predictions and uncertainty parameters and returns the log probability density of that result.
        """


class ZScalingCalibrator(UncertaintyCalibrator):
    """
    A class that calibrates regression uncertainty models by applying
    a scaling value to the uncalibrated standard deviation, fitted by minimizing the
    negative log likelihood of a normal distribution around each prediction
    with scaling given by the uncalibrated variance. Method is described
    in https://arxiv.org/abs/1905.11659.
    """

    @property
    def label(self):
        if self.regression_calibrator_metric == "stdev":
            label = f"{self.uncertainty_method}_zscaling_stdev"
        else:  # interval
            label = (
                f"{self.uncertainty_method}_zscaling_{self.interval_percentile}interval"
            )
        return label

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError("Z Score Scaling is only compatible with regression datasets.")

    def calibrate(self):
        if self.calibration_data.is_atom_bond_targets:
            mask, arrays = _atom_bond_calibration_arrays(
                self.calibration_data,
                predictions=self.calibration_predictor.get_uncal_preds(),
                variances=self.calibration_predictor.get_uncal_vars(),
                targets=self.calibration_data.targets(),
            )
            uncal_preds = arrays['predictions']
            uncal_vars = arrays['variances']
            targets = arrays['targets']
        else:
            uncal_preds = np.asarray(
                self.calibration_predictor.get_uncal_preds(), dtype=float,
            ).T
            uncal_vars = np.asarray(
                self.calibration_predictor.get_uncal_vars(), dtype=float,
            ).T
            targets = np.asarray(self.calibration_data.targets(), dtype=float).T
            mask = np.asarray(self.calibration_data.mask(), dtype=bool)
        self.num_tasks = len(mask)
        self.scaling = np.zeros(self.num_tasks)

        for i in range(self.num_tasks):
            task_preds, task_targets, task_vars = (
                _observed_regression_calibration_task(
                    'Z-scaling', i, uncal_preds[i], targets[i], uncal_vars[i], mask[i]
                )
            )
            task_errors = task_preds - task_targets
            task_zscore = task_errors / np.sqrt(task_vars)

            def objective(scaler_value: float):
                scaler_value = float(np.asarray(scaler_value).reshape(-1)[0])
                if not np.isfinite(scaler_value) or scaler_value <= 0:
                    return np.inf
                scaled_vars = task_vars * scaler_value**2
                nll = np.log(2 * np.pi * scaled_vars) / 2 + (task_errors) ** 2 / (2 * scaled_vars)
                return nll.sum()

            initial_guess = max(float(np.std(task_zscore)), np.sqrt(np.finfo(float).eps))
            sol = float(fmin(objective, initial_guess, disp=False)[0])

            if self.regression_calibrator_metric == "stdev":
                self.scaling[i] = sol
            else:  # interval
                self.scaling[i] = sol * erfinv(self.interval_percentile / 100) * np.sqrt(2)

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())
        uncal_vars = np.array(uncal_predictor.get_uncal_vars())
        if self.calibration_data.is_atom_bond_targets:
            cal_stdev = []
            sqrt_uncal_vars = [
                [np.sqrt(var) for var in uncal_var] for uncal_var in uncal_vars
            ]
            for sqrt_uncal_var in sqrt_uncal_vars:
                scaled_stdev = [var * s for var, s in zip(sqrt_uncal_var, self.scaling)]
                cal_stdev.append(scaled_stdev)
            return uncal_preds, cal_stdev
        else:
            cal_stdev = np.sqrt(uncal_vars) * np.expand_dims(self.scaling, axis=0)
            return uncal_preds.tolist(), cal_stdev.tolist()

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ):
        if self.calibration_data.is_atom_bond_targets:
            task_masks = validate_task_masks(mask, num_tasks=self.num_tasks)
            lengths = [len(task_mask) for task_mask in task_masks]
            preds = flatten_atom_bond_values(
                preds, self.num_tasks, 'Predictions', lengths,
            )
            unc_var = [
                np.square(values) for values in flatten_atom_bond_values(
                    unc, self.num_tasks, 'Uncertainties', lengths,
                )
            ]
            targets = flatten_atom_bond_values(
                targets, self.num_tasks, 'Targets', lengths,
            )
            mask = task_masks
        else:
            unc_var = np.square(np.asarray(unc, dtype=float)).T
            preds = np.asarray(preds, dtype=float).T
            targets = np.asarray(targets, dtype=float).T
            mask = np.asarray(mask, dtype=bool)
        nll = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_preds = preds[i][task_mask]
            task_targets = targets[i][task_mask]
            task_unc = unc_var[i][task_mask]
            task_nll = (
                np.log(2 * np.pi * task_unc) / 2
                + (task_preds - task_targets) ** 2 / (2 * task_unc)
            )
            nll.append(task_nll.mean())
        return nll


class TScalingCalibrator(UncertaintyCalibrator):
    """
    A class that calibrates regression uncertainty models using a variation of the
    ZScaling method. Instead, this method assumes that error is dominated by
    variance error as represented by the variance of the ensemble predictions.
    The scaling value is obtained by minimizing the negative log likelihood
    of the t distribution, including reductio term due to the number of ensemble models sampled.
    """

    @property
    def label(self):
        if self.regression_calibrator_metric == "stdev":
            label = f"{self.uncertainty_method}_tscaling_stdev"
        else:  # interval
            label = (
                f"{self.uncertainty_method}_tscaling_{self.interval_percentile}interval"
            )
        return label

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError("T Score Scaling is only compatible with regression datasets.")
        if self.uncertainty_method == "dropout":
            raise ValueError("T scaling not enabled with dropout variance uncertainty method.")
        if self.num_models == 1:
            raise ValueError("T scaling is intended for use with ensemble models.")

    def calibrate(self):
        if self.calibration_data.is_atom_bond_targets:
            mask, arrays = _atom_bond_calibration_arrays(
                self.calibration_data,
                predictions=self.calibration_predictor.get_uncal_preds(),
                variances=self.calibration_predictor.get_uncal_vars(),
                targets=self.calibration_data.targets(),
            )
            uncal_preds = arrays['predictions']
            uncal_vars = arrays['variances']
            targets = arrays['targets']
        else:
            uncal_preds = np.asarray(
                self.calibration_predictor.get_uncal_preds(), dtype=float,
            ).T
            uncal_vars = np.asarray(
                self.calibration_predictor.get_uncal_vars(), dtype=float,
            ).T
            targets = np.asarray(self.calibration_data.targets(), dtype=float).T
            mask = np.asarray(self.calibration_data.mask(), dtype=bool)
        self.num_tasks = len(mask)
        self.scaling = np.zeros(self.num_tasks)

        for i in range(self.num_tasks):
            task_preds, task_targets, task_vars = (
                _observed_regression_calibration_task(
                    'T-scaling', i, uncal_preds[i], targets[i], uncal_vars[i], mask[i]
                )
            )
            std_error_of_mean = np.sqrt(
                task_vars / (self.num_models - 1)
            )  # reduced for number of samples and include Bessel's correction
            task_errors = task_preds - task_targets
            task_tscore = task_errors / std_error_of_mean

            def objective(scaler_value: np.ndarray):
                scaler_value = float(np.asarray(scaler_value).reshape(-1)[0])
                if not np.isfinite(scaler_value) or scaler_value <= 0:
                    return np.inf
                scaled_std = std_error_of_mean * scaler_value
                likelihood = t.pdf(
                    x=task_errors, df=self.num_models - 1, scale=scaled_std
                )  # scipy t distribution pdf
                nll = -1 * np.sum(np.log(likelihood), axis=0)
                return nll

            initial_guess = max(float(np.std(task_tscore)), np.sqrt(np.finfo(float).eps))
            stdev_scaling = float(fmin(objective, initial_guess, disp=False)[0])
            if self.regression_calibrator_metric == "stdev":
                self.scaling[i] = stdev_scaling
            else:  # interval
                interval_scaling = stdev_scaling * t.ppf(
                    (self.interval_percentile / 100 + 1) / 2, df=self.num_models - 1
                )
                self.scaling[i] = interval_scaling

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())
        uncal_vars = np.array(uncal_predictor.get_uncal_vars())
        if self.calibration_data.is_atom_bond_targets:
            cal_stdev = []
            sqrt_uncal_vars = [
                [np.sqrt(var / (self.num_models - 1)) for var in uncal_var]
                for uncal_var in uncal_vars
            ]
            for sqrt_uncal_var in sqrt_uncal_vars:
                scaled_stdev = [var * s for var, s in zip(sqrt_uncal_var, self.scaling)]
                cal_stdev.append(scaled_stdev)
            return uncal_preds, cal_stdev
        else:
            cal_stdev = np.sqrt(uncal_vars / (self.num_models - 1)) * np.expand_dims(
                self.scaling, axis=0
            )
        return uncal_preds.tolist(), cal_stdev.tolist()

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ):
        # ``apply_calibration`` returns the Student-t scale (a calibrated
        # standard deviation), which is exactly what scipy's ``scale``
        # parameter expects. Squaring it here incorrectly treated a variance
        # as a scale and distorted every T-scaling NLL evaluation.
        if self.calibration_data.is_atom_bond_targets:
            task_masks = validate_task_masks(mask, num_tasks=self.num_tasks)
            lengths = [len(task_mask) for task_mask in task_masks]
            unc = flatten_atom_bond_values(
                unc, self.num_tasks, 'Uncertainties', lengths,
            )
            preds = flatten_atom_bond_values(
                preds, self.num_tasks, 'Predictions', lengths,
            )
            targets = flatten_atom_bond_values(
                targets, self.num_tasks, 'Targets', lengths,
            )
            mask = task_masks
        else:
            unc = np.asarray(unc, dtype=float).T
            preds = np.asarray(preds, dtype=float).T
            targets = np.asarray(targets, dtype=float).T
            mask = np.asarray(mask, dtype=bool)
        nll = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_preds = preds[i][task_mask]
            task_targets = targets[i][task_mask]
            task_unc = unc[i][task_mask]
            if task_unc.size == 0:
                nll.append(float('nan'))
                continue
            if (
                not np.all(np.isfinite(task_preds))
                or not np.all(np.isfinite(task_targets))
                or not np.all(np.isfinite(task_unc))
                or np.any(task_unc <= 0)
            ):
                raise ValueError(
                    'T-scaling NLL expects finite predictions and targets and '
                    'finite, strictly positive scale values.'
                )
            task_nll = -1 * t.logpdf(
                x=task_preds - task_targets, scale=task_unc, df=self.num_models - 1
            )
            nll.append(task_nll.mean())
        return nll


class ZelikmanCalibrator(UncertaintyCalibrator):
    """
    A calibrator for regression datasets that does not depend on a particular probability
    function form. Designed to be used with interval output. Uses the "CRUDE" method as
    described in https://arxiv.org/abs/2005.12496. As implemented here, the interval
    bounds are constrained to be symmetrical, though this is not required in the source method.
    The probability density to be used for NLL evaluator for the zelikman interval method is
    approximated here as a histogram function.
    """

    @property
    def label(self):
        if self.regression_calibrator_metric == "stdev":
            label = f"{self.uncertainty_method}_zelikman_stdev"
        else:
            label = f"{self.uncertainty_method}_zelikman_{self.interval_percentile}interval"
        return label

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError("Crude Scaling is only compatible with regression datasets.")

    def calibrate(self):
        if self.calibration_data.is_atom_bond_targets:
            mask, arrays = _atom_bond_calibration_arrays(
                self.calibration_data,
                predictions=self.calibration_predictor.get_uncal_preds(),
                variances=self.calibration_predictor.get_uncal_vars(),
                targets=self.calibration_data.targets(),
            )
            uncal_preds = arrays['predictions']
            uncal_vars = arrays['variances']
            targets = arrays['targets']
        else:
            uncal_preds = np.asarray(
                self.calibration_predictor.get_uncal_preds(), dtype=float,
            ).T
            uncal_vars = np.asarray(
                self.calibration_predictor.get_uncal_vars(), dtype=float,
            ).T
            targets = np.asarray(self.calibration_data.targets(), dtype=float).T
            mask = np.asarray(self.calibration_data.mask(), dtype=bool)
        self.num_tasks = len(mask)
        self.histogram_parameters = []
        self.scaling = np.zeros(self.num_tasks)
        for i in range(self.num_tasks):
            task_preds, task_targets, task_vars = (
                _observed_regression_calibration_task(
                    'Zelikman', i, uncal_preds[i], targets[i], uncal_vars[i], mask[i]
                )
            )
            task_preds = np.abs(task_preds - task_targets) / np.sqrt(task_vars)
            if self.regression_calibrator_metric == "interval":
                interval_scaling = np.percentile(task_preds, self.interval_percentile)
                self.scaling[i] = interval_scaling
            else:
                symmetric_z = np.concatenate([task_preds, -1 * task_preds])
                std_scaling = np.std(symmetric_z, axis=0)
                self.scaling[i] = std_scaling
            # histogram parameters for nll calculation
            h_params = np.histogram(task_preds, bins="auto", density=True)
            self.histogram_parameters.append(h_params)

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())
        uncal_vars = np.array(uncal_predictor.get_uncal_vars())
        if self.calibration_data.is_atom_bond_targets:
            cal_stdev = []
            sqrt_uncal_vars = [
                [np.sqrt(var) for var in uncal_var] for uncal_var in uncal_vars
            ]
            for sqrt_uncal_var in sqrt_uncal_vars:
                scaled_stdev = [var * s for var, s in zip(sqrt_uncal_var, self.scaling)]
                cal_stdev.append(scaled_stdev)
            return uncal_preds, cal_stdev
        else:
            cal_stdev = np.sqrt(uncal_vars) * np.expand_dims(self.scaling, axis=0)
            return uncal_preds.tolist(), cal_stdev.tolist()

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ):
        if self.calibration_data.is_atom_bond_targets:
            task_masks = validate_task_masks(mask, num_tasks=self.num_tasks)
            lengths = [len(task_mask) for task_mask in task_masks]
            preds = flatten_atom_bond_values(
                preds, self.num_tasks, 'Predictions', lengths,
            )
            unc = flatten_atom_bond_values(
                unc, self.num_tasks, 'Uncertainties', lengths,
            )
            targets = flatten_atom_bond_values(
                targets, self.num_tasks, 'Targets', lengths,
            )
            mask = task_masks
        else:
            preds = np.asarray(preds, dtype=float).T
            unc = np.asarray(unc, dtype=float).T
            targets = np.asarray(targets, dtype=float).T
            mask = np.asarray(mask, dtype=bool)
        nll = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_preds = preds[i][task_mask]
            task_targets = targets[i][task_mask]
            task_stdev = unc[i][task_mask] / self.scaling[i]
            task_abs_z = np.abs(task_preds - task_targets) / task_stdev
            bin_edges = self.histogram_parameters[i][1]
            bin_magnitudes = self.histogram_parameters[i][0]
            bin_magnitudes = np.insert(bin_magnitudes, [0, len(bin_magnitudes)], 0)
            pred_bins = np.searchsorted(bin_edges, task_abs_z)
            # magnitude adjusted by stdev scale of the distribution and symmetry assumption
            task_likelihood = bin_magnitudes[pred_bins] / task_stdev / 2
            task_nll = -1 * np.log(task_likelihood)
            nll.append(task_nll.mean())
        return nll


class MVEWeightingCalibrator(UncertaintyCalibrator):
    """
    A method of calibration for models that have ensembles of individual models that
    make variance predictions. Minimizes the negative log likelihood for the
    predictions versus the targets by applying a weighted average across the
    variance predictions of the ensemble. Discussed in https://doi.org/10.1186/s13321-021-00551-x.
    """

    @property
    def label(self):
        if self.regression_calibrator_metric == "stdev":
            label = f"{self.uncertainty_method}_mve_weighting_stdev"
        else:  # interval
            label = f"{self.uncertainty_method}_mve_weighting_{self.interval_percentile}interval"
        return label

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                f"MVE Weighting is only compatible with regression datasets! got: {self.dataset_type}"
            )
        if self.loss_function not in ["mve", "evidential"]:
            raise ValueError(
                "MVE Weighting calibration can only be carried out with MVE or Evidential loss function models."
            )
        if self.num_models == 1:
            raise ValueError(
                "MVE Weighting is only useful when weighting between results in an ensemble. Only one model was provided."
            )

    def calibrate(self):
        individual_vars = self.calibration_predictor.get_individual_vars()
        if self.calibration_data.is_atom_bond_targets:
            mask, arrays = _atom_bond_calibration_arrays(
                self.calibration_data,
                predictions=self.calibration_predictor.get_uncal_preds(),
                targets=self.calibration_data.targets(),
            )
            self.num_tasks = len(mask)
            uncal_preds = arrays['predictions']
            targets = arrays['targets']
            if len(individual_vars) != self.num_models:
                raise ValueError('MVE weighting received the wrong number of model variances.')
            task_lengths = [len(task_mask) for task_mask in mask]
            task_individual_vars = []
            for task_index, expected_length in enumerate(task_lengths):
                try:
                    model_values = [
                        np.asarray(model_vars[task_index], dtype=float).reshape(-1)
                        for model_vars in individual_vars
                    ]
                    task_values = np.stack(model_values, axis=0)
                except (IndexError, TypeError, ValueError) as error:
                    raise ValueError(
                        'Atom/bond MVE variances must have matching shapes for '
                        'every model and task.'
                    ) from error
                if task_values.shape != (self.num_models, expected_length):
                    raise ValueError(
                        f'Atom/bond MVE variance task {task_index} has shape '
                        f'{task_values.shape}; expected '
                        f'{(self.num_models, expected_length)}.'
                    )
                task_individual_vars.append(task_values)
            individual_vars = task_individual_vars
        else:
            uncal_preds = np.asarray(
                self.calibration_predictor.get_uncal_preds(), dtype=float,
            ).T
            individual_vars = np.asarray(individual_vars, dtype=float)
            targets = np.asarray(self.calibration_data.targets(), dtype=float).T
            mask = np.asarray(self.calibration_data.mask(), dtype=bool)
            self.num_tasks = len(mask)
            if (
                individual_vars.ndim != 3
                or individual_vars.shape[0] != self.num_models
                or individual_vars.shape[2] != self.num_tasks
            ):
                raise ValueError(
                    'MVE individual variances must have shape (models, data, tasks).'
                )
            individual_vars = [
                individual_vars[:, :, task_index]
                for task_index in range(self.num_tasks)
            ]
        self.var_weighting = np.zeros([self.num_models, self.num_tasks])  # shape(models, tasks)

        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_preds = uncal_preds[i][task_mask]
            task_ind_vars = individual_vars[i][:, task_mask]
            task_errors = task_preds - task_targets

            def objective(scaler_values: np.ndarray):
                scaler_values = np.reshape(softmax(scaler_values), [-1, 1])  # (models, 1)
                scaled_vars = np.sum(
                    task_ind_vars * scaler_values, axis=0, keepdims=False
                )  # shape(data)
                nll = np.log(2 * np.pi * scaled_vars) / 2 + (task_errors) ** 2 / (2 * scaled_vars)
                nll = np.sum(nll)
                return nll

            initial_guess = np.ones(self.num_models)
            sol = fmin(objective, initial_guess)
            self.var_weighting[:, i] = softmax(sol)
        if self.regression_calibrator_metric == "stdev":
            self.scaling = np.repeat(1, self.num_tasks)
        else:  # interval
            self.scaling = np.repeat(erfinv(self.interval_percentile / 100) * np.sqrt(2), self.num_tasks)

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())
        uncal_individual_vars = uncal_predictor.get_individual_vars()
        if self.calibration_data.is_atom_bond_targets:
            if len(uncal_individual_vars) != self.num_models:
                raise ValueError('MVE weighting received the wrong number of model variances.')
            weighted_vars = []
            for task_index in range(self.num_tasks):
                try:
                    task_model_vars = np.stack(
                        [
                            np.asarray(model_vars[task_index], dtype=float)
                            for model_vars in uncal_individual_vars
                        ],
                        axis=0,
                    )
                except (IndexError, TypeError, ValueError) as error:
                    raise ValueError(
                        'Atom/bond MVE variances must have matching shapes for '
                        'every model and task.'
                    ) from error
                task_weights = self.var_weighting[:, task_index].reshape(
                    (self.num_models,) + (1,) * (task_model_vars.ndim - 1)
                )
                weighted_vars.append(np.sum(task_model_vars * task_weights, axis=0))

            weighted_stdev = [
                np.sqrt(task_vars) * self.scaling[task_index]
                for task_index, task_vars in enumerate(weighted_vars)
            ]
            natom_targets = len(self.calibration_data[0].atom_targets) if self.calibration_data[0].atom_targets is not None else 0
            nbond_targets = len(self.calibration_data[0].bond_targets) if self.calibration_data[0].bond_targets is not None else 0
            weighted_stdev = reshape_values(
                weighted_stdev,
                uncal_predictor.test_data,
                natom_targets,
                nbond_targets,
            )
            return uncal_preds, weighted_stdev
        else:
            try:
                uncal_individual_vars = np.asarray(
                    uncal_individual_vars, dtype=float,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    'MVE variances must form a numeric (models, data, tasks) array.'
                ) from error
            expected_prefix = (self.num_models,)
            if (
                uncal_individual_vars.ndim != 3
                or uncal_individual_vars.shape[:1] != expected_prefix
                or uncal_individual_vars.shape[2] != self.num_tasks
                or self.var_weighting.shape != (self.num_models, self.num_tasks)
            ):
                raise ValueError(
                    'MVE variances and calibration weights have incompatible shapes.'
                )
            weighted_vars = np.sum(
                uncal_individual_vars
                * self.var_weighting[:, np.newaxis, :],
                axis=0,
            )
            weighted_stdev = np.sqrt(weighted_vars) * self.scaling
            return uncal_preds.tolist(), weighted_stdev.tolist()

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ):
        if self.calibration_data.is_atom_bond_targets:
            task_masks = validate_task_masks(mask, num_tasks=self.num_tasks)
            lengths = [len(task_mask) for task_mask in task_masks]
            unc_var = [
                np.square(values) for values in flatten_atom_bond_values(
                    unc, self.num_tasks, 'Uncertainties', lengths,
                )
            ]
            preds = flatten_atom_bond_values(
                preds, self.num_tasks, 'Predictions', lengths,
            )
            targets = flatten_atom_bond_values(
                targets, self.num_tasks, 'Targets', lengths,
            )
            mask = task_masks
        else:
            unc_var = np.square(np.asarray(unc, dtype=float)).T
            preds = np.asarray(preds, dtype=float).T
            targets = np.asarray(targets, dtype=float).T
            mask = np.asarray(mask, dtype=bool)
        nll = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_preds = preds[i][task_mask]
            task_targets = targets[i][task_mask]
            task_unc = unc_var[i][task_mask]
            task_nll = (
                np.log(2 * np.pi * task_unc) / 2
                + (task_preds - task_targets) ** 2 / (2 * task_unc)
            )
            nll.append(task_nll.mean())
        return nll


class PlattCalibrator(UncertaintyCalibrator):
    """
    A calibration method for classification datasets based on the Platt scaling algorithm.
    As discussed in https://arxiv.org/abs/1706.04599.
    """

    @property
    def label(self):
        return f"{self.uncertainty_method}_platt_confidence"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "classification":
            raise ValueError("Platt scaling is only implemented for classification dataset types.")

    def calibrate(self):
        raw_predictions = self.calibration_predictor.get_uncal_preds()
        if self.calibration_data.is_atom_bond_targets:
            mask, arrays = _atom_bond_calibration_arrays(
                self.calibration_data,
                predictions=raw_predictions,
                targets=self.calibration_data.targets(),
            )
            uncal_preds = arrays['predictions']
            targets = arrays['targets']
        else:
            uncal_preds = np.asarray(raw_predictions, dtype=float).T
            targets = np.asarray(self.calibration_data.targets(), dtype=float).T
            mask = np.asarray(self.calibration_data.mask(), dtype=bool)
        self.num_tasks = len(mask)
        # If train class sizes are available, set Bayes corrected calibration targets
        if self.calibration_predictor.train_class_sizes is not None:
            class_size_correction = True
            train_class_sizes = np.sum(
                self.calibration_predictor.train_class_sizes, axis=0
            )  # shape(tasks, 2)
            negative_target = 1 / (train_class_sizes[:, 0] + 2)
            positive_target = (train_class_sizes[:, 1] + 1) / (train_class_sizes[:, 1] + 2)
            print(
                "Platt scaling for calibration uses Bayesian correction against training set overfitting, "
                "replacing calibration targets [0,1] with adjusted values."
            )
        else:
            class_size_correction = False
            print(
                "Class sizes used in training models unavailable in checkpoints before Chemprop v1.5.0. "
                "No Bayesian correction perfomed as part of class scaling."
            )

        platt_parameters = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_preds = uncal_preds[i][task_mask]
            if class_size_correction:
                task_targets[task_targets == 0] = negative_target[i]
                task_targets[task_targets == 1] = positive_target[i]
                print(
                    f"Platt Bayesian correction for task {i} in calibration replacing [0,1] targets with {[negative_target[i], positive_target[i]]}"
                )

            def objective(parameters: np.ndarray):
                a = parameters[0]
                b = parameters[1]
                scaled_preds = expit(a * logit(task_preds) + b)
                nll = -1 * np.sum(
                    task_targets * np.log(scaled_preds)
                    + (1 - task_targets) * np.log(1 - scaled_preds)
                )
                return nll

            initial_guess = [1, 0]
            sol = fmin(objective, initial_guess)
            platt_parameters.append(sol)

        platt_parameters = np.array(platt_parameters)  # shape(task, 2)
        self.platt_a = platt_parameters[:, 0]
        self.platt_b = platt_parameters[:, 1]

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        raw_predictions = uncal_predictor.get_uncal_preds()
        if self.calibration_data.is_atom_bond_targets:
            task_predictions = flatten_atom_bond_values(
                raw_predictions,
                num_tasks=self.num_tasks,
                label='predictions',
            )
            eps = np.finfo(float).eps
            task_calibrated = [
                expit(
                    self.platt_a[task_index]
                    * logit(np.clip(values, eps, 1 - eps))
                    + self.platt_b[task_index]
                )
                for task_index, values in enumerate(task_predictions)
            ]
            calibration_example = self.calibration_data[0]
            natom_targets = (
                len(calibration_example.atom_targets)
                if calibration_example.atom_targets is not None
                else 0
            )
            nbond_targets = (
                len(calibration_example.bond_targets)
                if calibration_example.bond_targets is not None
                else 0
            )
            cal_preds = reshape_values(
                task_calibrated,
                uncal_predictor.test_data,
                natom_targets,
                nbond_targets,
            )
            return raw_predictions, cal_preds

        uncal_preds = np.asarray(raw_predictions, dtype=float)
        eps = np.finfo(float).eps
        cal_preds = expit(
            np.expand_dims(self.platt_a, axis=0)
            * logit(np.clip(uncal_preds, eps, 1 - eps))
            + np.expand_dims(self.platt_b, axis=0)
        )
        return uncal_preds.tolist(), cal_preds.tolist()

    def nll(self, preds: List[List[float]], unc: List[List[float]], targets: List[List[float]], mask: List[List[bool]]):
        if self.calibration_data.is_atom_bond_targets:
            task_masks = validate_task_masks(mask, num_tasks=self.num_tasks)
            lengths = [len(task_mask) for task_mask in task_masks]
            targets = flatten_atom_bond_values(
                targets, self.num_tasks, 'Targets', lengths,
            )
            unc = flatten_atom_bond_values(
                unc, self.num_tasks, 'Uncertainties', lengths,
            )
            mask = task_masks
        else:
            targets = np.asarray(targets, dtype=float).T
            unc = np.asarray(unc, dtype=float).T
            mask = np.asarray(mask, dtype=bool)
        nll = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_unc = unc[i][task_mask]

            likelihood = task_unc * task_targets + (1 - task_unc) * (1 - task_targets)
            task_nll = -1 * np.log(likelihood)
            nll.append(task_nll.mean())
        return nll


class IsotonicCalibrator(UncertaintyCalibrator):
    """
    A calibration method for classification datasets based on the isotonic regression algorithm.
    In effect, the method transforms incoming uncalibrated confidences using a histogram-like
    function where the range of each transforming bin and its magnitude is learned.
    As discussed in https://arxiv.org/abs/1706.04599.
    """

    @property
    def label(self):
        return f"{self.uncertainty_method}_isotonic_confidence"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "classification":
            raise ValueError(
                "Isotonic Regression is only implemented for classification dataset types."
            )

    def calibrate(self):
        raw_predictions = self.calibration_predictor.get_uncal_preds()
        if self.calibration_data.is_atom_bond_targets:
            mask, arrays = _atom_bond_calibration_arrays(
                self.calibration_data,
                predictions=raw_predictions,
                targets=self.calibration_data.targets(),
            )
            uncal_preds = arrays['predictions']
            targets = arrays['targets']
        else:
            uncal_preds = np.asarray(raw_predictions, dtype=float).T
            targets = np.asarray(self.calibration_data.targets(), dtype=float).T
            mask = np.asarray(self.calibration_data.mask(), dtype=bool)
        self.num_tasks = len(mask)

        isotonic_models = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_preds = uncal_preds[i][task_mask]

            isotonic_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            isotonic_model.fit(task_preds, task_targets)
            isotonic_models.append(isotonic_model)

        self.isotonic_models = isotonic_models

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())  # shape(data, task)
        if self.calibration_data.is_atom_bond_targets:
            cal_preds = []
            uncal_preds_list = [np.concatenate(x) for x in zip(*uncal_preds)]
            for i, iso_model in enumerate(self.isotonic_models):
                task_preds = uncal_preds_list[i]
                task_cal = iso_model.predict(task_preds)
                transpose_cal_preds = [task_cal]
                cal_preds.append(np.transpose(transpose_cal_preds))
            return uncal_preds, cal_preds
        else:
            transpose_cal_preds = []
            for i, iso_model in enumerate(self.isotonic_models):
                task_preds = uncal_preds[:, i]
                task_cal = iso_model.predict(task_preds)
                transpose_cal_preds.append(task_cal)
            cal_preds = np.transpose(transpose_cal_preds)
            return uncal_preds.tolist(), cal_preds.tolist()

    def nll(self, preds: List[List[float]], unc: List[List[float]], targets: List[List[float]], mask: List[List[bool]]):
        if self.calibration_data.is_atom_bond_targets:
            task_masks = validate_task_masks(mask, num_tasks=self.num_tasks)
            lengths = [len(task_mask) for task_mask in task_masks]
            targets = flatten_atom_bond_values(
                targets, self.num_tasks, 'Targets', lengths,
            )
            unc = flatten_atom_bond_values(
                unc, self.num_tasks, 'Uncertainties', lengths,
            )
            mask = task_masks
        else:
            targets = np.asarray(targets, dtype=float).T
            unc = np.asarray(unc, dtype=float).T
            mask = np.asarray(mask, dtype=bool)
        nll = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_unc = unc[i][task_mask]

            likelihood = task_unc * task_targets + (1 - task_unc) * (1 - task_targets)
            task_nll = -1 * np.log(likelihood)
            nll.append(task_nll.mean())
        return nll


class IsotonicMulticlassCalibrator(UncertaintyCalibrator):
    """
    A multiclass method for classification datasets based on the isotonic regression algorithm.
    In effect, the method transforms incoming uncalibrated confidences using a histogram-like
    function where the range of each transforming bin and its magnitude is learned. Uses a
    one-against-all aggregation scheme for convertering between binary and multiclass classifiers.
    As discussed in https://arxiv.org/abs/1706.04599.
    """

    @property
    def label(self):
        return f"{self.uncertainty_method}_isotonic_confidence"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "multiclass":
            raise ValueError(
                "Isotonic Multiclass Regression is only implemented for multiclass dataset types."
            )

    def calibrate(self):
        uncal_preds = np.array(
            self.calibration_predictor.get_uncal_preds()
        )  # shape(data, tasks, num_classes)
        targets = np.array(self.calibration_data.targets(), dtype=float)  # shape(data, tasks)
        mask = np.array(self.calibration_data.mask())
        self.num_tasks = len(mask)
        self.num_classes = uncal_preds.shape[2]

        isotonic_models = []
        for i in range(self.num_tasks):
            isotonic_models.append([])
            task_mask = mask[i]
            task_targets = targets[task_mask, i]  # shape(data)
            task_preds = uncal_preds[task_mask, i]
            for j in range(self.num_classes):
                class_preds = task_preds[:, j]  # shape(data)
                positive_class_targets = task_targets == j

                class_targets = np.ones_like(class_preds)
                class_targets[positive_class_targets] = 1
                class_targets[~positive_class_targets] = 0

                isotonic_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                isotonic_model.fit(class_preds, class_targets)
                isotonic_models[i].append(isotonic_model)

        self.isotonic_models = isotonic_models  # shape(tasks, classes)

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())  # shape(data, task, class)
        transpose_cal_preds = []
        for i in range(self.num_tasks):
            transpose_cal_preds.append([])
            for j in range(self.num_classes):
                class_preds = uncal_preds[:, i, j]
                class_cal = self.isotonic_models[i][j].predict(class_preds)
                transpose_cal_preds[i].append(class_cal)  # shape (task, class, data)
        cal_preds = np.transpose(transpose_cal_preds, [2, 0, 1])  # shape(data, task, class)
        cal_preds = cal_preds / np.sum(cal_preds, axis=2, keepdims=True)
        return uncal_preds.tolist(), cal_preds.tolist()

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ):
        targets = np.array(targets, dtype=float)  # shape(data, tasks)
        mask = np.array(mask)
        unc = np.array(unc)
        preds = np.array(preds)
        nll = []
        for i in range(targets.shape[1]):
            task_mask = mask[i]
            task_preds = unc[task_mask, i]
            task_target_values = targets[task_mask, i]
            if (
                not np.all(np.isfinite(task_target_values))
                or np.any(task_target_values != np.floor(task_target_values))
            ):
                raise ValueError('Multiclass targets must be finite integer class indices.')
            task_targets = task_target_values.astype(int)  # shape(data)
            if np.any(task_targets < 0) or np.any(task_targets >= task_preds.shape[1]):
                raise ValueError('Multiclass targets must be valid class indices.')
            bin_targets = np.zeros_like(task_preds)  # shape(valid_data, classes)
            bin_targets[np.arange(task_targets.shape[0]), task_targets] = 1
            task_likelihood = np.sum(bin_targets * task_preds, axis=1)
            task_nll = -1 * np.log(task_likelihood)
            nll.append(task_nll.mean())
        return nll


class ConformalMulticlassCalibrator(UncertaintyCalibrator):
    """
    Conformal Calibrator. Outputs binary values for whether each class should be included in the
    conformal set for each task.
    As discussed in https://arxiv.org/abs/2107.07511.
    """

    @property
    def label(self):
        return f"conformal_{self.conformal_alpha}"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "multiclass":
            raise ValueError("Conformal is only implemented for multiclass dataset types.")

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ):
        raise NotImplementedError(
            "The NLL uncertainty evaluation method for classification models has not been implemented for use with the conformal classification calibration method."
            )

    @staticmethod
    def nonconformity_scores(uncal_preds):
        """Fixed per class. Example is for basic conformal.

        Args:
            uncal_preds (torch.Tensor): a tensor of shape `n x t x c`, where `n`
            is the number of examples, `t` the number of tasks, and `c` the number
            of classes, containing the uncalibrated model predictions.
        
        Returns:
            scores (torch.Tensor): [num_examples, num_tasks, num_classes]
        """
        return -uncal_preds

    def calibrate(self):
        uncal_preds = np.array(
            self.calibration_predictor.get_uncal_preds()
        )  # shape(data, tasks, num_classes)
        targets = np.array(self.calibration_data.targets(), dtype=float)  # shape(data, tasks)
        mask = np.array(self.calibration_data.mask(), dtype=bool)
        _, self.num_tasks, self.num_classes = uncal_preds.shape

        all_scores = self.nonconformity_scores(uncal_preds)
        self.qhats = []

        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[task_mask, i]
            if (
                not np.all(np.isfinite(task_targets))
                or np.any(task_targets != np.floor(task_targets))
                or np.any(task_targets < 0)
                or np.any(task_targets >= self.num_classes)
            ):
                raise ValueError('Multiclass targets must be valid integer class indices.')
            task_scores = np.take_along_axis(
                all_scores[task_mask, i], task_targets.reshape(-1, 1).astype(int), axis=1
            ).squeeze(1)  # shape(valid_data)
            self.qhats.append(
                _conformal_quantile(task_scores, self.conformal_alpha)
            )

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())  # shape(data, task)
        cal_preds = np.zeros_like(uncal_preds, dtype=int)
        all_scores = self.nonconformity_scores(uncal_preds)
        for task_id, qhat in enumerate(self.qhats):
            cal_preds[:, task_id] = all_scores[:, task_id] <= qhat
        return uncal_preds.tolist(), cal_preds.tolist()


class ConformalAdaptiveMulticlassCalibrator(ConformalMulticlassCalibrator):
    """
    Adaptive Conformal Calibrator. Outputs binary values for whether each class should be
    included in the conformal set for each task.
    As discussed in https://arxiv.org/abs/2107.07511.
    """

    @property
    def label(self):
        return f"conformal_adaptive_{self.conformal_alpha}"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "multiclass":
            raise ValueError("Conformal Adaptive is only implemented for multiclass dataset types.")

    @staticmethod
    def nonconformity_scores(uncal_preds):
        """Fixed per class. Example is for adaptive conformal.

        Args:
            uncal_preds (torch.Tensor): a tensor of shape `n x t x c`, where `n`
            is the number of examples, `t` the number of tasks, and `c` the number
            of classes, containing the uncalibrated model predictions.
        
        Returns:
            scores (torch.Tensor): [num_examples, num_tasks, num_classes]
        """
        sort_inds = np.argsort(-uncal_preds, axis=2)
        sorted_preds = np.take_along_axis(uncal_preds, sort_inds, axis=2)
        sorted_scores = sorted_preds.cumsum(axis=2)
        unsort_inds = np.argsort(sort_inds, axis=2)
        unsorted_scores = np.take_along_axis(sorted_scores, unsort_inds, axis=2)
        return unsorted_scores


class ConformalMultilabelCalibrator(UncertaintyCalibrator):
    """
    Conformal Calibrator for Multilabel datasets. Creates conformal in-set and conformal out-set such that
    for 1-alpha proportion of datapoints, the set of labels is bounded by the in-set and out-set. That is,
    the conformal in-set is contained in the set of actual labels and the set of actual labels is contained
    in the conformal out-set.
    As discussed in https://arxiv.org/abs/2004.10181.
    """

    @property
    def label(self):
        return f"conformal_multilabel_{self.conformal_alpha}"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "classification":
            raise ValueError("Conformal is only implemented for classification dataset types.")

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ):
        raise NotImplementedError(
            "The NLL uncertainty evaluation method for classification models has not been implemented for use with the conformal classification calibration method."
            )

    @staticmethod
    def nonconformity_scores(uncal_preds):
        """Fixed per class. Example is for multilabel conformal.

        Args:
            uncal_preds (torch.Tensor): a tensor of shape `n x t`, where `n`
            is the number of examples, and `t` the number of tasks,
            containing the uncalibrated model predictions.
        
        Returns:
            scores (torch.Tensor): [num_examples, num_tasks]
        """
        return -uncal_preds

    def calibrate(self):
        uncal_preds = np.array(
            self.calibration_predictor.get_uncal_preds()
        )  # shape(data, tasks)
        targets = np.array(self.calibration_data.targets(), dtype=float)  # shape(data, tasks)
        mask = np.array(self.calibration_data.mask(), dtype=bool)
        self.num_data, self.num_tasks = targets.shape
        observed = mask.T
        if uncal_preds.shape != targets.shape or observed.shape != targets.shape:
            raise ValueError('Multilabel calibration predictions and targets must have matching shapes.')
        if not np.all(np.isfinite(uncal_preds)):
            raise ValueError('Multilabel calibration predictions must be finite.')
        if np.any(~np.isin(targets[observed], [0, 1])):
            raise ValueError('Observed multilabel calibration targets must be 0 or 1.')

        scores = self.nonconformity_scores(uncal_preds)
        negative_labels = observed & (targets == 0)
        positive_labels = observed & (targets == 1)
        rows_with_negatives = np.any(negative_labels, axis=1)
        rows_with_positives = np.any(positive_labels, axis=1)
        if not np.any(rows_with_negatives) or not np.any(rows_with_positives):
            raise ValueError(
                'Multilabel conformal calibration requires observed positive '
                'and negative labels.'
            )
        calibration_scores_in = np.min(
            np.where(negative_labels[rows_with_negatives], scores[rows_with_negatives], np.inf),
            axis=1,
        )
        calibration_scores_out = np.max(
            np.where(positive_labels[rows_with_positives], scores[rows_with_positives], -np.inf),
            axis=1,
        )

        self.tout = float(np.quantile(
            calibration_scores_out,
            1 - self.conformal_alpha / 2,
            method="higher",
        ))
        self.tin = float(np.quantile(
            calibration_scores_in,
            self.conformal_alpha / 2,
            method="higher",
        ))

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())  # shape(data, task)
        scores = self.nonconformity_scores(uncal_preds)

        cal_preds_in = (scores <= self.tin).astype(int)
        cal_preds_out = (scores <= self.tout).astype(int)
        cal_preds = np.concatenate((cal_preds_in, cal_preds_out), axis=1)

        return uncal_preds.tolist(), cal_preds.tolist()


class ConformalRegressionCalibrator(UncertaintyCalibrator):
    """
    Conformal Calibrator for regression datasets. Used for both conformal regression and conformal quantile regression.
    Outputs interval of variable size, centered around quantile outputs of model, for each datapoint. Intervals 
    should cover 1-alpha proportion of datapoints.
    As discussed in https://arxiv.org/abs/2107.07511.
    """

    @property
    def label(self):
        return f"conformal_regression_{self.conformal_alpha}_half_interval"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Conformal Regression is only implemented for regression dataset types."
            )

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ):
        raise NotImplementedError(
            "The NLL uncertainty evaluation method for regression models has not been implemented for use with the conformal regression calibration method."
            )

    def calibrate(self):
        if self.calibration_data.is_atom_bond_targets:
            mask, arrays = _atom_bond_calibration_arrays(
                self.calibration_data,
                predictions=self.calibration_predictor.get_uncal_preds(),
                intervals=self.calibration_predictor.get_uncal_output(),
                targets=self.calibration_data.targets(),
            )
            uncal_preds = arrays['predictions']
            uncal_interval = arrays['intervals']
            targets = arrays['targets']
        else:
            uncal_preds = np.asarray(
                self.calibration_predictor.get_uncal_preds(), dtype=float,
            ).T
            uncal_interval = np.asarray(
                self.calibration_predictor.get_uncal_output(), dtype=float,
            ).T
            targets = np.asarray(self.calibration_data.targets(), dtype=float).T
            mask = np.asarray(self.calibration_data.mask(), dtype=bool)
            shapes = {
                uncal_preds.shape,
                uncal_interval.shape,
                targets.shape,
                mask.shape,
            }
            if len(shapes) != 1:
                raise ValueError(
                    'Conformal regression calibration predictions, intervals, '
                    'targets, and masks must have matching task-by-row shapes; '
                    f'got predictions={uncal_preds.shape}, '
                    f'intervals={uncal_interval.shape}, targets={targets.shape}, '
                    f'mask={mask.shape}.'
                )
        self.num_tasks = len(mask)

        self.qhats = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_preds = uncal_preds[i][task_mask]
            task_interval = uncal_interval[i][task_mask]
            uncal_interval_lower = task_preds - task_interval
            uncal_interval_upper = task_preds + task_interval
            calibration_scores = np.maximum(
                uncal_interval_lower - task_targets, task_targets - uncal_interval_upper
            )
            self.qhats.append(
                _conformal_quantile(calibration_scores, self.conformal_alpha)
            )

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        raw_preds = uncal_predictor.get_uncal_preds()
        raw_intervals = uncal_predictor.get_uncal_output()
        if self.calibration_data.is_atom_bond_targets:
            task_values = flatten_atom_bond_value_sets(
                {'predictions': raw_preds, 'intervals': raw_intervals},
                num_tasks=self.num_tasks,
            )
            cal_task_intervals = [
                values + self.qhats[task_index]
                for task_index, values in enumerate(task_values['intervals'])
            ]
            calibration_example = self.calibration_data[0]
            natom_targets = (
                len(calibration_example.atom_targets)
                if calibration_example.atom_targets is not None
                else 0
            )
            nbond_targets = (
                len(calibration_example.bond_targets)
                if calibration_example.bond_targets is not None
                else 0
            )
            cal_intervals = reshape_values(
                cal_task_intervals,
                uncal_predictor.test_data,
                natom_targets,
                nbond_targets,
            )
            return raw_preds, cal_intervals

        uncal_preds = np.asarray(raw_preds, dtype=float)
        uncal_interval = np.asarray(raw_intervals, dtype=float)
        cal_intervals = uncal_interval + self.qhats
        return uncal_preds, cal_intervals


def build_uncertainty_calibrator(
    calibration_method: str,
    uncertainty_method: str,
    regression_calibrator_metric: str,
    interval_percentile: int,
    calibration_data: MoleculeDataset,
    calibration_data_loader: MoleculeDataLoader,
    models: Iterator[MoleculeModel],
    scalers: Iterator[StandardScaler],
    num_models: int,
    dataset_type: str,
    loss_function: str,
    uncertainty_dropout_p: float,
    conformal_alpha: float,
    dropout_sampling_size: int,
    spectra_phase_mask: List[List[bool]],
) -> UncertaintyCalibrator:
    """
    Function that chooses the subclass of :class: `UncertaintyCalibrator`
    based on the provided arguments and returns that class.
    """
    if calibration_method is None:
        if dataset_type == "regression":
            if regression_calibrator_metric == "stdev":
                calibration_method = "zscaling"
            else:
                calibration_method = "zelikman_interval"
        if dataset_type in ["classification", "multiclass"]:
            calibration_method = "isotonic"

    supported_calibrators = {
        "zscaling": ZScalingCalibrator,
        "tscaling": TScalingCalibrator,
        "zelikman_interval": ZelikmanCalibrator,
        "mve_weighting": MVEWeightingCalibrator,
        "platt": PlattCalibrator,
        "conformal": ConformalMultilabelCalibrator
        if dataset_type == "classification"
        else ConformalMulticlassCalibrator,
        "conformal_adaptive": ConformalAdaptiveMulticlassCalibrator,
        "conformal_regression": ConformalRegressionCalibrator,
        "conformal_quantile_regression": ConformalRegressionCalibrator,
        "isotonic": IsotonicCalibrator
        if dataset_type == "classification"
        else IsotonicMulticlassCalibrator,
    }

    calibrator_class = supported_calibrators.get(calibration_method, None)

    if calibrator_class is None:
        raise NotImplementedError(
            f"Calibrator type {calibration_method} is not currently supported. Avalable options are: {list(supported_calibrators.keys())}"
        )
    else:
        calibrator = calibrator_class(
            uncertainty_method=uncertainty_method,
            calibration_method=calibration_method,
            regression_calibrator_metric=regression_calibrator_metric,
            interval_percentile=interval_percentile,
            calibration_data=calibration_data,
            calibration_data_loader=calibration_data_loader,
            models=models,
            scalers=scalers,
            num_models=num_models,
            dataset_type=dataset_type,
            loss_function=loss_function,
            uncertainty_dropout_p=uncertainty_dropout_p,
            conformal_alpha=conformal_alpha,
            dropout_sampling_size=dropout_sampling_size,
            spectra_phase_mask=spectra_phase_mask,
        )
    return calibrator
