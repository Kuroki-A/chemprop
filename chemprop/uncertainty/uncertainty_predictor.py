from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np
from tqdm import tqdm

from chemprop.data import MoleculeDataset, StandardScaler, MoleculeDataLoader
from chemprop.models import MoleculeModel
from chemprop.spectra_utils import normalize_spectra, roundrobin_sid
from chemprop.multitask_utils import reshape_values, reshape_individual_preds


def _atom_bond_task_arrays(values, label: str) -> List[np.ndarray]:
    """Coerces task-major atom/bond values without forming a ragged array."""
    try:
        task_values = list(values)
    except TypeError as error:
        raise ValueError(f'{label} must contain one array per atom/bond task.') from error

    arrays = []
    for task_index, value in enumerate(task_values):
        try:
            array = np.array(value, dtype=float, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'{label} task {task_index} must be a numeric array.'
            ) from error
        if array.ndim == 0:
            raise ValueError(
                f'{label} task {task_index} must have at least one dimension.'
            )
        arrays.append(array)
    return arrays


def _numeric_values(values, is_atom_bond: bool, label: str):
    """Returns either one dense array or a list of independent task arrays."""
    if is_atom_bond:
        return _atom_bond_task_arrays(values, label)
    return np.array(values, dtype=float, copy=True)


def _accumulate_numeric_values(total, values, is_atom_bond: bool, label: str):
    """Adds a model/sample result to an accumulator with shape validation."""
    numeric = _numeric_values(values, is_atom_bond, label)
    if not is_atom_bond:
        if total.shape != numeric.shape:
            raise ValueError(
                f'{label} shape changed from {total.shape} to {numeric.shape}.'
            )
        total += numeric
        return total

    if len(total) != len(numeric):
        raise ValueError(
            f'{label} task count changed from {len(total)} to {len(numeric)}.'
        )
    for task_index, (total_task, task_values) in enumerate(zip(total, numeric)):
        if total_task.shape != task_values.shape:
            raise ValueError(
                f'{label} task {task_index} shape changed from '
                f'{total_task.shape} to {task_values.shape}.'
            )
        total_task += task_values
    return total


def _zeros_like_numeric(values, is_atom_bond: bool, label: str):
    """Creates zero-valued dense or task-major accumulators."""
    numeric = _numeric_values(values, is_atom_bond, label)
    if is_atom_bond:
        return [np.zeros_like(value, dtype=float) for value in numeric]
    return np.zeros_like(numeric, dtype=float)


def _validate_numeric_alignment(
    reference, values, is_atom_bond: bool, label: str,
) -> None:
    """Validates that uncertainty parameters align exactly with predictions."""
    reference_numeric = _numeric_values(reference, is_atom_bond, 'Predictions')
    values_numeric = _numeric_values(values, is_atom_bond, label)
    if not is_atom_bond:
        if reference_numeric.shape != values_numeric.shape:
            raise ValueError(
                f'{label} shape {values_numeric.shape} does not match prediction '
                f'shape {reference_numeric.shape}.'
            )
        return

    if len(reference_numeric) != len(values_numeric):
        raise ValueError(
            f'{label} has {len(values_numeric)} tasks but predictions have '
            f'{len(reference_numeric)}.'
        )
    for task_index, (prediction, value) in enumerate(
        zip(reference_numeric, values_numeric)
    ):
        if prediction.shape != value.shape:
            raise ValueError(
                f'{label} task {task_index} shape {value.shape} does not match '
                f'prediction shape {prediction.shape}.'
            )


def _update_running_moments(
    running_mean,
    running_m2,
    values,
    count: int,
    is_atom_bond: bool,
    label: str,
):
    """Updates population moments with Welford's numerically stable method."""
    numeric = _numeric_values(values, is_atom_bond, label)
    if not is_atom_bond:
        if running_mean.shape != numeric.shape:
            raise ValueError(
                f'{label} shape changed from {running_mean.shape} to {numeric.shape}.'
            )
        delta = numeric - running_mean
        running_mean += delta / count
        running_m2 += delta * (numeric - running_mean)
        return running_mean, running_m2

    if len(running_mean) != len(numeric):
        raise ValueError(
            f'{label} task count changed from {len(running_mean)} to {len(numeric)}.'
        )
    for task_index, (mean_task, m2_task, task_values) in enumerate(
        zip(running_mean, running_m2, numeric)
    ):
        if mean_task.shape != task_values.shape:
            raise ValueError(
                f'{label} task {task_index} shape changed from '
                f'{mean_task.shape} to {task_values.shape}.'
            )
        delta = task_values - mean_task
        mean_task += delta / count
        m2_task += delta * (task_values - mean_task)
    return running_mean, running_m2


def _atom_bond_elementwise(label: str, operation, *value_sets) -> List[np.ndarray]:
    """Applies an operation independently to aligned ragged task arrays."""
    arrays = [
        _atom_bond_task_arrays(values, label)
        for values in value_sets
    ]
    if not arrays:
        return []
    task_count = len(arrays[0])
    if any(len(values) != task_count for values in arrays[1:]):
        raise ValueError(f'{label} atom/bond parameter task counts do not match.')

    results = []
    for task_index, task_values in enumerate(zip(*arrays)):
        reference_shape = task_values[0].shape
        if any(value.shape != reference_shape for value in task_values[1:]):
            raise ValueError(
                f'{label} parameter shapes do not match for task {task_index}.'
            )
        results.append(operation(*task_values))
    return results


def predict(*args, **kwargs):
    """Imports the training predictor lazily to avoid package import cycles."""
    from chemprop.train.predict import predict as train_predict

    return train_predict(*args, **kwargs)


class UncertaintyPredictor(ABC):
    """
    A class for making model predictions and associated predictions of
    prediction uncertainty according to the chosen uncertainty method.
    """

    def __init__(
        self,
        test_data: MoleculeDataset,
        test_data_loader: MoleculeDataLoader,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        num_models: int,
        dataset_type: str,
        loss_function: str,
        uncertainty_dropout_p: float,
        conformal_alpha: float,
        dropout_sampling_size: int,
        individual_ensemble_predictions: bool = False,
        spectra_phase_mask: List[List[bool]] = None,
    ):
        self.test_data = test_data
        self.models = models
        self.scalers = scalers
        self.dataset_type = dataset_type
        self.loss_function = loss_function
        self.uncal_preds = None
        self.uncal_vars = None
        self.uncal_intervals = None
        self.uncal_confidence = None
        self.individual_vars = None
        self.num_models = num_models
        self.uncertainty_dropout_p = uncertainty_dropout_p
        self.conformal_alpha = conformal_alpha
        self.dropout_sampling_size = dropout_sampling_size
        self.individual_ensemble_predictions = individual_ensemble_predictions
        self.spectra_phase_mask = spectra_phase_mask
        self.train_class_sizes = None

        if (
            not isinstance(self.num_models, int)
            or isinstance(self.num_models, bool)
            or self.num_models < 1
        ):
            raise ValueError('num_models must be a positive integer.')
        self.raise_argument_errors()
        self.test_data_loader = test_data_loader
        self.calculate_predictions()

    def _model_scaler_pairs(self):
        """Iterates exactly ``num_models`` model/scaler pairs.

        ``zip`` silently truncates unequal iterables. That behavior is unsafe
        here because all ensemble aggregates are divided by ``num_models``.
        """
        model_iterator = iter(self.models)
        scaler_iterator = iter(self.scalers)
        sentinel = object()

        def validated_pairs():
            for _ in range(self.num_models):
                model = next(model_iterator, sentinel)
                scaler = next(scaler_iterator, sentinel)
                if model is sentinel or scaler is sentinel:
                    raise ValueError(
                        'The number of supplied models and scalers must both '
                        f'match num_models={self.num_models}.'
                    )
                yield model, scaler
            if (
                next(model_iterator, sentinel) is not sentinel
                or next(scaler_iterator, sentinel) is not sentinel
            ):
                raise ValueError(
                    'The number of supplied models and scalers must both '
                    f'match num_models={self.num_models}.'
                )

        return tqdm(validated_pairs(), total=self.num_models)

    def _record_train_class_sizes(self, model, model_index: int, predictions) -> None:
        """Validates and records classification counts for Bayesian calibration.

        Legacy checkpoints may not contain ``train_class_sizes``.  An ensemble
        must not silently mix those checkpoints with newer members because the
        Bayesian calibrator sums this metadata across models.
        """
        raw_sizes = getattr(model, 'train_class_sizes', None)
        has_sizes = raw_sizes is not None
        if model_index == 0:
            self._ensemble_has_train_class_sizes = has_sizes
        elif has_sizes != self._ensemble_has_train_class_sizes:
            first_state = (
                'contains' if self._ensemble_has_train_class_sizes else 'does not contain'
            )
            current_state = 'contains' if has_sizes else 'does not contain'
            raise ValueError(
                'Classification ensemble checkpoints have inconsistent '
                'train_class_sizes metadata: checkpoint 0 '
                f'{first_state} it, while checkpoint {model_index} '
                f'{current_state} it. Do not mix legacy and current '
                'classification checkpoints when using uncertainty calibration.'
            )

        if not has_sizes:
            return

        try:
            object_sizes = np.asarray(raw_sizes)
            sizes = np.asarray(raw_sizes, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'Checkpoint {model_index} train_class_sizes must be a '
                'rectangular two-dimensional numeric count matrix.'
            ) from error
        if object_sizes.dtype == np.bool_ or any(
            isinstance(value, (bool, np.bool_)) for value in object_sizes.flat
        ):
            raise ValueError(
                f'Checkpoint {model_index} train_class_sizes must contain '
                'integer class counts, not booleans.'
            )
        if sizes.ndim != 2 or 0 in sizes.shape:
            raise ValueError(
                f'Checkpoint {model_index} train_class_sizes must be a non-empty '
                f'two-dimensional task-by-class matrix; got shape {sizes.shape}.'
            )
        if (
            not np.all(np.isfinite(sizes))
            or np.any(sizes < 0)
            or not np.all(sizes == np.floor(sizes))
        ):
            raise ValueError(
                f'Checkpoint {model_index} train_class_sizes must contain only '
                'finite non-negative integer class counts.'
            )

        if model.is_atom_bond_targets:
            expected_tasks = len(predictions)
        else:
            prediction_values = np.asarray(predictions)
            expected_tasks = (
                prediction_values.shape[1]
                if prediction_values.ndim >= 2
                else None
            )
        expected_classes = 2 if self.dataset_type == 'classification' else None
        if self.dataset_type == 'multiclass':
            if model.is_atom_bond_targets:
                first_task = np.asarray(predictions[0]) if predictions else None
                expected_classes = (
                    first_task.shape[-1]
                    if first_task is not None and first_task.ndim >= 2
                    else None
                )
            else:
                prediction_values = np.asarray(predictions)
                expected_classes = (
                    prediction_values.shape[2]
                    if prediction_values.ndim >= 3
                    else None
                )
        expected_shape = (
            (expected_tasks, expected_classes)
            if expected_tasks is not None and expected_classes is not None
            else None
        )
        if expected_shape is not None and sizes.shape != expected_shape:
            raise ValueError(
                f'Checkpoint {model_index} train_class_sizes has shape '
                f'{sizes.shape}; predictions require task-by-class shape '
                f'{expected_shape}.'
            )

        if model_index == 0:
            self._train_class_sizes_shape = sizes.shape
            self.train_class_sizes = []
        elif sizes.shape != self._train_class_sizes_shape:
            raise ValueError(
                'Classification ensemble checkpoints have incompatible '
                'train_class_sizes shapes: checkpoint 0 has shape '
                f'{self._train_class_sizes_shape}, while checkpoint '
                f'{model_index} has shape {sizes.shape}.'
            )
        self.train_class_sizes.append(sizes.tolist())

    @property
    @abstractmethod
    def label(self):
        """
        The string in saved results indicating the uncertainty method used.
        """

    def raise_argument_errors(self):
        """
        Raise errors for incompatible dataset types or uncertainty methods, etc.
        """

    @abstractmethod
    def calculate_predictions(self):
        """
        Calculate the uncalibrated predictions and store them as attributes
        """

    def get_uncal_preds(self):
        """
        Return the predicted values for the test data.
        """
        return self.uncal_preds

    def get_uncal_vars(self):
        """
        Return the uncalibrated variances for the test data
        """
        return self.uncal_vars

    def get_uncal_confidence(self):
        """
        Return the uncalibrated confidences for the test data
        """
        return self.uncal_confidence

    def get_individual_vars(self):
        """
        Return the variances predicted by each individual model in an ensemble.
        """
        return self.individual_vars

    def get_individual_preds(self):
        """
        Return the value predicted by each individual model in an ensemble.
        """
        return self.individual_preds

    @abstractmethod
    def get_uncal_output(self):
        """
        Return the uncalibrated uncertainty outputs for the test data
        """


class NoUncertaintyPredictor(UncertaintyPredictor):
    """
    Class that is used for predictions when no uncertainty method is selected.
    Model value predictions are made as normal but uncertainty output only returns "nan".
    """

    @property
    def label(self):
        return "no_uncertainty_method"

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(self._model_scaler_pairs()):
            (
                scaler,
                features_scaler,
                atom_descriptor_scaler,
                bond_descriptor_scaler,
                atom_bond_scaler,
            ) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_descriptor_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        bond_descriptor_scaler, scale_bond_descriptors=True
                    )

            preds = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler,
                return_unc_parameters=False,
            )
            if self.dataset_type == "spectra":
                preds = normalize_spectra(
                    spectra=preds,
                    phase_features=self.test_data.phase_features(),
                    phase_mask=self.spectra_phase_mask,
                    excluded_sub_value=float("nan"),
                )
            if i == 0:
                sum_preds = _numeric_values(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )

                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        n_atoms, n_bonds = (
                            self.test_data.number_of_atoms,
                            self.test_data.number_of_bonds,
                        )
                        individual_preds = []
                        for _ in model.atom_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_atoms).sum(), 1, self.num_models))
                            )
                        for _ in model.bond_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_bonds).sum(), 1, self.num_models))
                            )
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                sum_preds = _accumulate_numeric_values(
                    sum_preds, preds, model.is_atom_bond_targets, 'Predictions',
                )

                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.append(
                            individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                        )

        if model.is_atom_bond_targets:
            uncal_preds = [pred / self.num_models for pred in sum_preds]
            self.uncal_preds = reshape_values(
                uncal_preds,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            uncal_vars = np.empty(len(sum_preds), dtype=object)
            for i, pred in enumerate(sum_preds):
                uncal_vars[i] = np.full(len(pred), np.nan)
            self.uncal_vars = reshape_values(
                uncal_vars,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            if self.individual_ensemble_predictions:
                self.individual_preds = reshape_individual_preds(
                    individual_preds,
                    self.test_data,
                    len(model.atom_targets),
                    len(model.bond_targets),
                    self.num_models,
                )
        else:
            self.uncal_preds = (sum_preds / self.num_models).tolist()
            uncal_vars = np.zeros_like(sum_preds)
            uncal_vars[:] = np.nan
            self.uncal_vars = uncal_vars
            if self.individual_ensemble_predictions:
                self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_vars


class ConformalQuantileRegressionPredictor(UncertaintyPredictor):
    """
    This class is used for conformal quantile regression. The original targets of
    the model are intervals. Here, we reformat the prediction results to be the
    midpoint of intervals and use the intervals as the `uncal_output`.
    """

    @property
    def label(self):
        return "no_uncertainty_method"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Conformal quantile regression is only compatible with regression dataset types."
            )

    @staticmethod
    def reformat_preds(preds):
        """
        Reformat predictions to the midpoint between the upper and lower quantiles.
        """
        num_data, num_tasks = preds.shape
        reshaped_preds = preds.reshape(num_data, 2, num_tasks // 2).mean(axis=1)
        return reshaped_preds

    @staticmethod
    def make_intervals(preds):
        """
        Make uncalibrated intervals from the uncalibrated predictions.
        """
        num_data, num_tasks = preds.shape
        intervals = abs(np.diff(preds.reshape(num_data, 2, num_tasks // 2), axis=1) / 2)
        intervals = intervals.reshape(num_data, num_tasks // 2)
        return intervals

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(self._model_scaler_pairs()):
            if model.is_atom_bond_targets:
                raise NotImplementedError(
                    'Conformal quantile and conformal regression uncertainty '
                    'are not supported for atom/bond property prediction.'
                )
            (
                scaler,
                features_scaler,
                atom_descriptor_scaler,
                bond_descriptor_scaler,
                atom_bond_scaler,
            ) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_descriptor_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        bond_descriptor_scaler, scale_bond_descriptors=True
                    )

            preds = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler,
                return_unc_parameters=False,
            )
            if i == 0:
                sum_preds = np.array(preds)
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        n_atoms, n_bonds = (
                            self.test_data.number_of_atoms,
                            self.test_data.number_of_bonds,
                        )
                        individual_preds = []
                        for _ in model.atom_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_atoms).sum(), 1, self.num_models))
                            )
                        for _ in model.bond_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_bonds).sum(), 1, self.num_models))
                            )
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                sum_preds += np.array(preds)
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.append(
                            individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                        )

        uncal_preds = sum_preds / self.num_models
        self.uncal_intervals = self.make_intervals(uncal_preds)
        if self.individual_ensemble_predictions:
            self.individual_preds = individual_preds.tolist()
        self.uncal_preds = self.reformat_preds(uncal_preds)

    def get_uncal_output(self):
        return self.uncal_intervals


class ConformalRegressionPredictor(ConformalQuantileRegressionPredictor):
    """
    This class is used for basic conformal regression. The prediction outputs are midpoints
    of intervals, while the uncalibrated intervals are reported as the `uncal_output`.
    """

    @staticmethod
    def reformat_preds(preds):
        """
        Reformat predictions to the midpoint between the upper and lower quantiles.
        """
        return preds

    @staticmethod
    def make_intervals(preds):
        """
        Make uncalibrated intervals from the uncalibrated predictions.
        """
        intervals = np.zeros(preds.shape)
        return intervals


class RoundRobinSpectraPredictor(UncertaintyPredictor):
    """
    A class predicting uncertainty for spectra outputs from an ensemble of models. Output is
    the average SID calculated pairwise between each of the individual spectrum predictions.
    """

    @property
    def label(self):
        return "roundrobin_sid"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "spectra":
            raise ValueError(
                "Round-robin spectral uncertainty requires the spectra dataset type."
            )
        if self.num_models < 2:
            raise ValueError(
                "Roundrobin uncertainty is only available when multiple models are provided."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(self._model_scaler_pairs()):
            (
                scaler,
                features_scaler,
                atom_descriptor_scaler,
                bond_descriptor_scaler,
                atom_bond_scaler,
            ) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_descriptor_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        bond_descriptor_scaler, scale_bond_descriptors=True
                    )

            preds = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler,
                return_unc_parameters=False,
            )
            if self.dataset_type == "spectra":
                preds = normalize_spectra(
                    spectra=preds,
                    phase_features=self.test_data.phase_features(),
                    phase_mask=self.spectra_phase_mask,
                    excluded_sub_value=float("nan"),
                )
            if i == 0:
                sum_preds = np.array(preds)
                individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                sum_preds += np.array(preds)
                individual_preds = np.append(
                    individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                )  # shape(data, tasks, ensemble)

        self.uncal_preds = (sum_preds / self.num_models).tolist()
        self.uncal_sid = roundrobin_sid(individual_preds)  # shape(data)
        if self.individual_ensemble_predictions:
            self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_sid


class MVEPredictor(UncertaintyPredictor):
    """
    Class that uses the variance output of the mve loss function (aka heteroscedastic loss)
    as a prediction uncertainty.
    """

    @property
    def label(self):
        return "mve_uncal_var"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.loss_function != "mve":
            raise ValueError(
                "In order to use mve uncertainty, trained models must have used mve loss function."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(self._model_scaler_pairs()):
            (
                scaler,
                features_scaler,
                atom_descriptor_scaler,
                bond_descriptor_scaler,
                atom_bond_scaler,
            ) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_descriptor_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        bond_descriptor_scaler, scale_bond_descriptors=True
                    )

            preds, var = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler,
                return_unc_parameters=True,
            )
            _validate_numeric_alignment(
                preds, var, model.is_atom_bond_targets, 'MVE variances',
            )
            if i == 0:
                mean_preds = _numeric_values(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
                prediction_m2 = _zeros_like_numeric(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
                sum_vars = _numeric_values(
                    var, model.is_atom_bond_targets, 'MVE variances',
                )
                individual_vars = [var]
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        n_atoms, n_bonds = (
                            self.test_data.number_of_atoms,
                            self.test_data.number_of_bonds,
                        )
                        individual_preds = []
                        for _ in model.atom_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_atoms).sum(), 1, self.num_models))
                            )
                        for _ in model.bond_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_bonds).sum(), 1, self.num_models))
                            )
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                mean_preds, prediction_m2 = _update_running_moments(
                    mean_preds,
                    prediction_m2,
                    preds,
                    i + 1,
                    model.is_atom_bond_targets,
                    'Predictions',
                )
                sum_vars = _accumulate_numeric_values(
                    sum_vars, var, model.is_atom_bond_targets, 'MVE variances',
                )
                individual_vars.append(var)
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.append(
                            individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                        )

        if model.is_atom_bond_targets:
            uncal_preds, uncal_vars = [], []
            for pred, m2, var in zip(mean_preds, prediction_m2, sum_vars):
                uncal_pred = pred
                uncal_var = np.maximum(
                    (var + m2) / self.num_models, 0,
                )
                uncal_preds.append(uncal_pred)
                uncal_vars.append(uncal_var)
            self.uncal_preds = reshape_values(
                uncal_preds,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            self.uncal_vars = reshape_values(
                uncal_vars,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            self.individual_vars = individual_vars
            if self.individual_ensemble_predictions:
                self.individual_preds = reshape_individual_preds(
                    individual_preds,
                    self.test_data,
                    len(model.atom_targets),
                    len(model.bond_targets),
                    self.num_models,
                )
        else:
            uncal_preds = mean_preds
            uncal_vars = np.maximum(
                (sum_vars + prediction_m2) / self.num_models, 0,
            )
            self.uncal_preds, self.uncal_vars = (
                uncal_preds.tolist(),
                uncal_vars.tolist(),
            )
            self.individual_vars = individual_vars
            if self.individual_ensemble_predictions:
                self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_vars


class EvidentialTotalPredictor(UncertaintyPredictor):
    """
    Uses the evidential loss function to calculate total uncertainty variance from
    ancilliary loss function outputs. As presented in https://doi.org/10.1021/acscentsci.1c00546.
    """

    @property
    def label(self):
        return "evidential_total_uncal_var"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.loss_function != "evidential":
            raise ValueError(
                "In order to use evidential uncertainty, trained models must have used evidential regression loss function."
            )
        if self.dataset_type != "regression":
            raise ValueError(
                "Evidential total uncertainty is only compatible with regression dataset types."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(self._model_scaler_pairs()):
            (
                scaler,
                features_scaler,
                atom_descriptor_scaler,
                bond_descriptor_scaler,
                atom_bond_scaler,
            ) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_descriptor_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        bond_descriptor_scaler, scale_bond_descriptors=True
                    )

            preds, lambdas, alphas, betas = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler,
                return_unc_parameters=True,
            )
            for parameter_label, parameter_values in (
                ('Evidential lambdas', lambdas),
                ('Evidential alphas', alphas),
                ('Evidential betas', betas),
            ):
                _validate_numeric_alignment(
                    preds,
                    parameter_values,
                    model.is_atom_bond_targets,
                    parameter_label,
                )
            if model.is_atom_bond_targets:
                var = _atom_bond_elementwise(
                    'Evidential parameters',
                    lambda beta, weight, alpha: (
                        beta * (1 + 1 / weight) / (alpha - 1)
                    ),
                    betas,
                    lambdas,
                    alphas,
                )
            else:
                var = (
                    np.asarray(betas)
                    * (1 + 1 / np.asarray(lambdas))
                    / (np.asarray(alphas) - 1)
                )
            if i == 0:
                mean_preds = _numeric_values(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
                prediction_m2 = _zeros_like_numeric(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
                sum_vars = _numeric_values(
                    var, model.is_atom_bond_targets, 'Evidential variances',
                )
                individual_vars = [var]
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        n_atoms, n_bonds = (
                            self.test_data.number_of_atoms,
                            self.test_data.number_of_bonds,
                        )
                        individual_preds = []
                        for _ in model.atom_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_atoms).sum(), 1, self.num_models))
                            )
                        for _ in model.bond_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_bonds).sum(), 1, self.num_models))
                            )
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                mean_preds, prediction_m2 = _update_running_moments(
                    mean_preds,
                    prediction_m2,
                    preds,
                    i + 1,
                    model.is_atom_bond_targets,
                    'Predictions',
                )
                sum_vars = _accumulate_numeric_values(
                    sum_vars,
                    var,
                    model.is_atom_bond_targets,
                    'Evidential variances',
                )
                individual_vars.append(var)
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.append(
                            individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                        )

        if model.is_atom_bond_targets:
            uncal_preds, uncal_vars = [], []
            for pred, m2, var in zip(mean_preds, prediction_m2, sum_vars):
                uncal_pred = pred
                uncal_var = np.maximum(
                    (var + m2) / self.num_models, 0,
                )
                uncal_preds.append(uncal_pred)
                uncal_vars.append(uncal_var)
            self.uncal_preds = reshape_values(
                uncal_preds,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            self.uncal_vars = reshape_values(
                uncal_vars,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            self.individual_vars = individual_vars
            if self.individual_ensemble_predictions:
                self.individual_preds = reshape_individual_preds(
                    individual_preds,
                    self.test_data,
                    len(model.atom_targets),
                    len(model.bond_targets),
                    self.num_models,
                )
        else:
            uncal_preds = mean_preds
            uncal_vars = np.maximum(
                (sum_vars + prediction_m2) / self.num_models, 0,
            )
            self.uncal_preds, self.uncal_vars = (
                uncal_preds.tolist(),
                uncal_vars.tolist(),
            )
            self.individual_vars = individual_vars
            if self.individual_ensemble_predictions:
                self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_vars


class EvidentialAleatoricPredictor(UncertaintyPredictor):
    """
    Uses the evidential loss function to calculate aleatoric uncertainty variance from
    ancilliary loss function outputs. As presented in https://doi.org/10.1021/acscentsci.1c00546.
    """

    @property
    def label(self):
        return "evidential_aleatoric_uncal_var"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.loss_function != "evidential":
            raise ValueError(
                "In order to use evidential uncertainty, trained models must have used evidential regression loss function."
            )
        if self.dataset_type != "regression":
            raise ValueError(
                "Evidential aleatoric uncertainty is only compatible with regression dataset types."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(self._model_scaler_pairs()):
            (
                scaler,
                features_scaler,
                atom_descriptor_scaler,
                bond_descriptor_scaler,
                atom_bond_scaler,
            ) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_descriptor_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        bond_descriptor_scaler, scale_bond_descriptors=True
                    )

            preds, lambdas, alphas, betas = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler,
                return_unc_parameters=True,
            )
            for parameter_label, parameter_values in (
                ('Evidential lambdas', lambdas),
                ('Evidential alphas', alphas),
                ('Evidential betas', betas),
            ):
                _validate_numeric_alignment(
                    preds,
                    parameter_values,
                    model.is_atom_bond_targets,
                    parameter_label,
                )
            if model.is_atom_bond_targets:
                var = _atom_bond_elementwise(
                    'Evidential parameters',
                    lambda beta, alpha: beta / (alpha - 1),
                    betas,
                    alphas,
                )
            else:
                var = np.asarray(betas) / (np.asarray(alphas) - 1)
            if i == 0:
                mean_preds = _numeric_values(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
                prediction_m2 = _zeros_like_numeric(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
                sum_vars = _numeric_values(
                    var, model.is_atom_bond_targets, 'Evidential variances',
                )
                individual_vars = [var]
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        n_atoms, n_bonds = (
                            self.test_data.number_of_atoms,
                            self.test_data.number_of_bonds,
                        )
                        individual_preds = []
                        for _ in model.atom_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_atoms).sum(), 1, self.num_models))
                            )
                        for _ in model.bond_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_bonds).sum(), 1, self.num_models))
                            )
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                mean_preds, prediction_m2 = _update_running_moments(
                    mean_preds,
                    prediction_m2,
                    preds,
                    i + 1,
                    model.is_atom_bond_targets,
                    'Predictions',
                )
                sum_vars = _accumulate_numeric_values(
                    sum_vars,
                    var,
                    model.is_atom_bond_targets,
                    'Evidential variances',
                )
                individual_vars.append(var)
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.append(
                            individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                        )

        if model.is_atom_bond_targets:
            uncal_preds, uncal_vars = [], []
            for pred, m2, var in zip(mean_preds, prediction_m2, sum_vars):
                uncal_pred = pred
                uncal_var = np.maximum(
                    (var + m2) / self.num_models, 0,
                )
                uncal_preds.append(uncal_pred)
                uncal_vars.append(uncal_var)
            self.uncal_preds = reshape_values(
                uncal_preds,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            self.uncal_vars = reshape_values(
                uncal_vars,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            self.individual_vars = individual_vars
            if self.individual_ensemble_predictions:
                self.individual_preds = reshape_individual_preds(
                    individual_preds,
                    self.test_data,
                    len(model.atom_targets),
                    len(model.bond_targets),
                    self.num_models,
                )
        else:
            uncal_preds = mean_preds
            uncal_vars = np.maximum(
                (sum_vars + prediction_m2) / self.num_models, 0,
            )
            self.uncal_preds, self.uncal_vars = (
                uncal_preds.tolist(),
                uncal_vars.tolist(),
            )
            self.individual_vars = individual_vars
            if self.individual_ensemble_predictions:
                self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_vars


class EvidentialEpistemicPredictor(UncertaintyPredictor):
    """
    Uses the evidential loss function to calculate epistemic uncertainty variance from
    ancilliary loss function outputs. As presented in https://doi.org/10.1021/acscentsci.1c00546.
    """

    @property
    def label(self):
        return "evidential_epistemic_uncal_var"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.loss_function != "evidential":
            raise ValueError(
                "In order to use evidential uncertainty, trained models must have used evidential regression loss function."
            )
        if self.dataset_type != "regression":
            raise ValueError(
                "Evidential epistemic uncertainty is only compatible with regression dataset types."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(self._model_scaler_pairs()):
            (
                scaler,
                features_scaler,
                atom_descriptor_scaler,
                bond_descriptor_scaler,
                atom_bond_scaler,
            ) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_descriptor_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        bond_descriptor_scaler, scale_bond_descriptors=True
                    )

            preds, lambdas, alphas, betas = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler,
                return_unc_parameters=True,
            )
            for parameter_label, parameter_values in (
                ('Evidential lambdas', lambdas),
                ('Evidential alphas', alphas),
                ('Evidential betas', betas),
            ):
                _validate_numeric_alignment(
                    preds,
                    parameter_values,
                    model.is_atom_bond_targets,
                    parameter_label,
                )
            if model.is_atom_bond_targets:
                var = _atom_bond_elementwise(
                    'Evidential parameters',
                    lambda beta, weight, alpha: (
                        beta / (weight * (alpha - 1))
                    ),
                    betas,
                    lambdas,
                    alphas,
                )
            else:
                var = (
                    np.asarray(betas)
                    / (np.asarray(lambdas) * (np.asarray(alphas) - 1))
                )
            if i == 0:
                mean_preds = _numeric_values(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
                prediction_m2 = _zeros_like_numeric(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
                sum_vars = _numeric_values(
                    var, model.is_atom_bond_targets, 'Evidential variances',
                )
                individual_vars = [var]
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        n_atoms, n_bonds = (
                            self.test_data.number_of_atoms,
                            self.test_data.number_of_bonds,
                        )
                        individual_preds = []
                        for _ in model.atom_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_atoms).sum(), 1, self.num_models))
                            )
                        for _ in model.bond_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_bonds).sum(), 1, self.num_models))
                            )
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                mean_preds, prediction_m2 = _update_running_moments(
                    mean_preds,
                    prediction_m2,
                    preds,
                    i + 1,
                    model.is_atom_bond_targets,
                    'Predictions',
                )
                sum_vars = _accumulate_numeric_values(
                    sum_vars,
                    var,
                    model.is_atom_bond_targets,
                    'Evidential variances',
                )
                individual_vars.append(var)
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.append(
                            individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                        )

        if model.is_atom_bond_targets:
            uncal_preds, uncal_vars = [], []
            for pred, m2, var in zip(mean_preds, prediction_m2, sum_vars):
                uncal_pred = pred
                uncal_var = np.maximum(
                    (var + m2) / self.num_models, 0,
                )
                uncal_preds.append(uncal_pred)
                uncal_vars.append(uncal_var)
            self.uncal_preds = reshape_values(
                uncal_preds,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            self.uncal_vars = reshape_values(
                uncal_vars,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            self.individual_vars = individual_vars
            if self.individual_ensemble_predictions:
                self.individual_preds = reshape_individual_preds(
                    individual_preds,
                    self.test_data,
                    len(model.atom_targets),
                    len(model.bond_targets),
                    self.num_models,
                )
        else:
            uncal_preds = mean_preds
            uncal_vars = np.maximum(
                (sum_vars + prediction_m2) / self.num_models, 0,
            )
            self.uncal_preds, self.uncal_vars = (
                uncal_preds.tolist(),
                uncal_vars.tolist(),
            )
            self.individual_vars = individual_vars
            if self.individual_ensemble_predictions:
                self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_vars


class EnsemblePredictor(UncertaintyPredictor):
    """
    Class that predicts uncertainty for predictions based on the variance in predictions among
    an ensemble's submodels.
    """

    @property
    def label(self):
        return "ensemble_uncal_var"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.num_models < 2:
            raise ValueError(
                "Ensemble method for uncertainty is only available when multiple models are provided."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(self._model_scaler_pairs()):
            (
                scaler,
                features_scaler,
                atom_descriptor_scaler,
                bond_descriptor_scaler,
                atom_bond_scaler,
            ) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_descriptor_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        bond_descriptor_scaler, scale_bond_descriptors=True
                    )
            preds = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler,
                return_unc_parameters=False,
            )
            if self.dataset_type == "spectra":
                preds = normalize_spectra(
                    spectra=preds,
                    phase_features=self.test_data.phase_features(),
                    phase_mask=self.spectra_phase_mask,
                    excluded_sub_value=float("nan"),
                )
            self._record_train_class_sizes(model, i, preds)
            if i == 0:
                mean_preds = _numeric_values(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
                prediction_m2 = _zeros_like_numeric(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        n_atoms, n_bonds = (
                            self.test_data.number_of_atoms,
                            self.test_data.number_of_bonds,
                        )
                        individual_preds = []
                        for _ in model.atom_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_atoms).sum(), 1, self.num_models))
                            )
                        for _ in model.bond_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_bonds).sum(), 1, self.num_models))
                            )
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                mean_preds, prediction_m2 = _update_running_moments(
                    mean_preds,
                    prediction_m2,
                    preds,
                    i + 1,
                    model.is_atom_bond_targets,
                    'Predictions',
                )
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.append(
                            individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                        )

        if model.is_atom_bond_targets:
            uncal_preds, uncal_vars = [], []
            for pred, m2 in zip(mean_preds, prediction_m2):
                uncal_pred = pred
                uncal_var = np.maximum(m2 / self.num_models, 0)
                uncal_preds.append(uncal_pred)
                uncal_vars.append(uncal_var)
            self.uncal_preds = reshape_values(
                uncal_preds,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            self.uncal_vars = reshape_values(
                uncal_vars,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )

            if self.individual_ensemble_predictions:
                self.individual_preds = reshape_individual_preds(
                    individual_preds,
                    self.test_data,
                    len(model.atom_targets),
                    len(model.bond_targets),
                    self.num_models,
                )
        else:
            uncal_preds = mean_preds
            uncal_vars = np.maximum(prediction_m2 / self.num_models, 0)
            self.uncal_preds, self.uncal_vars = (
                uncal_preds.tolist(),
                uncal_vars.tolist(),
            )

            if self.individual_ensemble_predictions:
                self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_vars


class DropoutPredictor(UncertaintyPredictor):
    """
    Class that creates an artificial ensemble of models by applying monte carlo dropout to the loaded
    model parameters. Predicts uncertainty for predictions based on the variance in predictions among
    an ensemble's submodels.
    """

    @property
    def label(self):
        return "dropout_uncal_var"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.num_models > 1:
            raise ValueError(
                "Dropout method for uncertainty should be used for a single model rather than an ensemble."
            )
        if (
            not isinstance(self.dropout_sampling_size, int)
            or isinstance(self.dropout_sampling_size, bool)
            or self.dropout_sampling_size < 2
        ):
            raise ValueError('dropout_sampling_size must be an integer of at least 2.')

    def calculate_predictions(self):
        (model, scaler_list), = list(self._model_scaler_pairs())
        (
            scaler,
            features_scaler,
            atom_descriptor_scaler,
            bond_descriptor_scaler,
            atom_bond_scaler,
        ) = scaler_list
        if (
            features_scaler is not None
            or atom_descriptor_scaler is not None
            or bond_descriptor_scaler is not None
        ):
            self.test_data.reset_features_and_targets()
            if features_scaler is not None:
                self.test_data.normalize_features(features_scaler)
            if atom_descriptor_scaler is not None:
                self.test_data.normalize_features(
                    atom_descriptor_scaler, scale_atom_descriptors=True
                )
            if bond_descriptor_scaler is not None:
                self.test_data.normalize_features(
                    bond_descriptor_scaler, scale_bond_descriptors=True
                )
        for i in range(self.dropout_sampling_size):
            preds = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler,
                return_unc_parameters=False,
                dropout_prob=self.uncertainty_dropout_p,
            )
            if i == 0:
                mean_preds = _numeric_values(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
                prediction_m2 = _zeros_like_numeric(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
            else:
                mean_preds, prediction_m2 = _update_running_moments(
                    mean_preds,
                    prediction_m2,
                    preds,
                    i + 1,
                    model.is_atom_bond_targets,
                    'Predictions',
                )

        if model.is_atom_bond_targets:
            uncal_preds, uncal_vars = [], []
            for pred, m2 in zip(mean_preds, prediction_m2):
                uncal_pred = pred
                uncal_var = np.maximum(
                    m2 / self.dropout_sampling_size, 0,
                )
                uncal_preds.append(uncal_pred)
                uncal_vars.append(uncal_var)
            self.uncal_preds = reshape_values(
                uncal_preds,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            self.uncal_vars = reshape_values(
                uncal_vars,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
        else:
            uncal_preds = mean_preds
            uncal_vars = np.maximum(
                prediction_m2 / self.dropout_sampling_size, 0,
            )
            self.uncal_preds, self.uncal_vars = (
                uncal_preds.tolist(),
                uncal_vars.tolist(),
            )

    def get_uncal_output(self):
        return self.uncal_vars


class ClassPredictor(UncertaintyPredictor):
    """
    Class uses the [0,1] range of results from classification or multiclass models
    as the indicator of confidence. Used for classification and multiclass dataset types.
    """

    @property
    def label(self):
        return "classification_uncal_confidence"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type not in ["classification", "multiclass"]:
            raise ValueError(
                "Classification output uncertainty method must be used with dataset types classification or multiclass."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(self._model_scaler_pairs()):
            (
                scaler,
                features_scaler,
                atom_descriptor_scaler,
                bond_descriptor_scaler,
                atom_bond_scaler,
            ) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_descriptor_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        bond_descriptor_scaler, scale_bond_descriptors=True
                    )

            preds = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler,
                return_unc_parameters=False,
            )
            self._record_train_class_sizes(model, i, preds)
            if i == 0:
                sum_preds = _numeric_values(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        n_atoms, n_bonds = (
                            self.test_data.number_of_atoms,
                            self.test_data.number_of_bonds,
                        )
                        individual_preds = []
                        for _ in model.atom_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_atoms).sum(), 1, self.num_models))
                            )
                        for _ in model.bond_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_bonds).sum(), 1, self.num_models))
                            )
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                sum_preds = _accumulate_numeric_values(
                    sum_preds, preds, model.is_atom_bond_targets, 'Predictions',
                )
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.append(
                            individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                        )

        if model.is_atom_bond_targets:
            uncal_preds = [pred / self.num_models for pred in sum_preds]
            self.uncal_preds = reshape_values(
                uncal_preds,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            self.uncal_confidence = self.uncal_preds
            if self.individual_ensemble_predictions:
                self.individual_preds = reshape_individual_preds(
                    individual_preds,
                    self.test_data,
                    len(model.atom_targets),
                    len(model.bond_targets),
                    self.num_models,
                )
        else:
            self.uncal_preds = (sum_preds / self.num_models).tolist()
            self.uncal_confidence = self.uncal_preds
            if self.individual_ensemble_predictions:
                self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_confidence


class DirichletPredictor(UncertaintyPredictor):
    """
    Dirichlet uncertainty
    """

    @property
    def label(self):
        return "dirichlet_uncal_uncertainty"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.loss_function != "dirichlet":
            raise ValueError(
                "In order to use Dirichlet evidential uncertainty, trained models must have used dirichlet loss function."
            )
        if self.dataset_type not in ["classification", "multiclass"]:
            raise ValueError(
                f"Dirichlet evidential epistemic uncertainty is only compatible with classification dataset types. \
                    Current dataset is of type {self.dataset_type}."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(self._model_scaler_pairs()):
            (
                scaler,
                features_scaler,
                atom_descriptor_scaler,
                bond_descriptor_scaler,
                atom_bond_scaler,
            ) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_descriptor_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        bond_descriptor_scaler, scale_bond_descriptors=True
                    )

            preds, alphas = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                atom_bond_scaler=atom_bond_scaler,
                return_unc_parameters=True,
            )

            if model.is_atom_bond_targets:
                alpha_values = _atom_bond_task_arrays(
                    alphas, 'Dirichlet parameters',
                )
                u = []
                for task_index, alpha in enumerate(alpha_values):
                    if alpha.ndim != 3:
                        raise ValueError(
                            'Dirichlet parameter task '
                            f'{task_index} must be 3-D; got shape {alpha.shape}.'
                        )
                    u.append(alpha.shape[2] / np.sum(alpha, axis=2))
            else:
                alpha_values = np.asarray(alphas)
                if alpha_values.ndim != 3:
                    raise ValueError(
                        'Dirichlet parameters must be a 3-D array; got shape '
                        f'{alpha_values.shape}.'
                    )
                u = alpha_values.shape[2] / np.sum(alpha_values, axis=2)

            _validate_numeric_alignment(
                preds, u, model.is_atom_bond_targets, 'Dirichlet uncertainty',
            )
            self._record_train_class_sizes(model, i, preds)

            if i == 0:
                sum_preds = _numeric_values(
                    preds, model.is_atom_bond_targets, 'Predictions',
                )
                sum_u = _numeric_values(
                    u, model.is_atom_bond_targets, 'Dirichlet uncertainty',
                )
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        n_atoms, n_bonds = (
                            self.test_data.number_of_atoms,
                            self.test_data.number_of_bonds,
                        )
                        individual_preds = []
                        for _ in model.atom_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_atoms).sum(), 1, self.num_models))
                            )
                        for _ in model.bond_targets:
                            individual_preds.append(
                                np.zeros((np.array(n_bonds).sum(), 1, self.num_models))
                            )
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                sum_preds = _accumulate_numeric_values(
                    sum_preds, preds, model.is_atom_bond_targets, 'Predictions',
                )
                sum_u = _accumulate_numeric_values(
                    sum_u,
                    u,
                    model.is_atom_bond_targets,
                    'Dirichlet uncertainty',
                )
                if self.individual_ensemble_predictions:
                    if model.is_atom_bond_targets:
                        for j, pred in enumerate(preds):
                            individual_preds[j][:, :, i] = pred
                    else:
                        individual_preds = np.append(
                            individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                        )

        if model.is_atom_bond_targets:
            uncal_preds = [pred / self.num_models for pred in sum_preds]
            uncal_u = [uncertainty / self.num_models for uncertainty in sum_u]
            self.uncal_preds = reshape_values(
                uncal_preds,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            self.uncal_uncertainty = reshape_values(
                uncal_u,
                self.test_data,
                len(model.atom_targets),
                len(model.bond_targets),
            )
            if self.individual_ensemble_predictions:
                self.individual_preds = reshape_individual_preds(
                    individual_preds,
                    self.test_data,
                    len(model.atom_targets),
                    len(model.bond_targets),
                    self.num_models,
                )
        else:
            self.uncal_preds = (sum_preds / self.num_models).tolist()
            self.uncal_uncertainty = (sum_u / self.num_models).tolist()
            if self.individual_ensemble_predictions:
                self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_uncertainty


def build_uncertainty_predictor(
    uncertainty_method: str,
    test_data: MoleculeDataset,
    test_data_loader: MoleculeDataLoader,
    models: Iterator[MoleculeModel],
    scalers: Iterator[StandardScaler],
    num_models: int,
    dataset_type: str,
    loss_function: str,
    uncertainty_dropout_p: float,
    conformal_alpha: float,
    dropout_sampling_size: int,
    individual_ensemble_predictions: bool,
    spectra_phase_mask: List[List[bool]],
) -> UncertaintyPredictor:
    """
    Function that chooses and returns the appropriate :class: `UncertaintyPredictor` subclass
    for the provided arguments.
    """

    supported_predictors = {
        None: NoUncertaintyPredictor,
        "mve": MVEPredictor,
        "ensemble": EnsemblePredictor,
        "classification": ClassPredictor,
        "evidential_total": EvidentialTotalPredictor,
        "evidential_epistemic": EvidentialEpistemicPredictor,
        "evidential_aleatoric": EvidentialAleatoricPredictor,
        "dropout": DropoutPredictor,
        "spectra_roundrobin": RoundRobinSpectraPredictor,
        "dirichlet":  DirichletPredictor,
        "conformal_quantile_regression": ConformalQuantileRegressionPredictor,
        "conformal_regression": ConformalRegressionPredictor,
    }

    predictor_class = supported_predictors.get(uncertainty_method, None)

    if predictor_class is None:
        raise NotImplementedError(
            f"Uncertainty predictor type {uncertainty_method} is not currently supported. Avalable options are: {list(supported_predictors.keys())}"
        )
    else:
        predictor = predictor_class(
            test_data=test_data,
            test_data_loader=test_data_loader,
            models=models,
            scalers=scalers,
            num_models=num_models,
            dataset_type=dataset_type,
            loss_function=loss_function,
            uncertainty_dropout_p=uncertainty_dropout_p,
            conformal_alpha=conformal_alpha,
            dropout_sampling_size=dropout_sampling_size,
            individual_ensemble_predictions=individual_ensemble_predictions,
            spectra_phase_mask=spectra_phase_mask,
        )
    return predictor
