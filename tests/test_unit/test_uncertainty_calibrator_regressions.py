from types import SimpleNamespace

import numpy as np
import pytest

import chemprop.uncertainty.uncertainty_calibrator as calibrator_module
from chemprop.uncertainty.uncertainty_calibrator import (
    ConformalMulticlassCalibrator,
    ConformalMultilabelCalibrator,
    ConformalRegressionCalibrator,
    IsotonicMulticlassCalibrator,
    MVEWeightingCalibrator,
    PlattCalibrator,
    TScalingCalibrator,
    ZScalingCalibrator,
    ZelikmanCalibrator,
    _conformal_quantile,
    build_uncertainty_calibrator,
)
from scipy.stats import t


class _PredictionResult:
    def __init__(self, predictions, uncertainty=None):
        self.predictions = predictions
        self.uncertainty = uncertainty

    def get_uncal_preds(self):
        return self.predictions

    def get_uncal_output(self):
        return self.uncertainty

    def get_uncal_vars(self):
        return self.uncertainty


def _regression_scaling_calibrator(calibrator_class, variance, target=1.0, observed=True):
    calibrator = calibrator_class.__new__(calibrator_class)
    calibrator.calibration_predictor = _PredictionResult([[0.0]], [[variance]])
    calibrator.calibration_data = SimpleNamespace(
        is_atom_bond_targets=False,
        targets=lambda: [[target]],
        mask=lambda: [[observed]],
    )
    calibrator.regression_calibrator_metric = 'stdev'
    calibrator.interval_percentile = 95
    calibrator.num_models = 3
    return calibrator


@pytest.mark.parametrize(
    'calibrator_class', [ZScalingCalibrator, TScalingCalibrator, ZelikmanCalibrator]
)
def test_regression_scaling_calibrators_reject_all_missing_task(calibrator_class):
    calibrator = _regression_scaling_calibrator(
        calibrator_class, variance=1.0, target=None, observed=False,
    )

    with pytest.raises(ValueError, match='task 0 has no observed targets'):
        calibrator.calibrate()


@pytest.mark.parametrize(
    'calibrator_class', [ZScalingCalibrator, TScalingCalibrator, ZelikmanCalibrator]
)
@pytest.mark.parametrize('variance', [0.0, -1.0, np.nan, np.inf])
def test_regression_scaling_calibrators_reject_invalid_variance(
    calibrator_class, variance,
):
    calibrator = _regression_scaling_calibrator(calibrator_class, variance)

    with pytest.raises(ValueError, match='task 0.*strictly positive variances'):
        calibrator.calibrate()


@pytest.mark.parametrize(
    'calibrator_class', [ZScalingCalibrator, TScalingCalibrator, ZelikmanCalibrator]
)
def test_regression_scaling_calibrators_reject_nonfinite_observation(
    calibrator_class,
):
    calibrator = _regression_scaling_calibrator(
        calibrator_class, variance=1.0, target=np.inf,
    )

    with pytest.raises(ValueError, match='task 0.*finite predictions and targets'):
        calibrator.calibrate()


def test_conformal_quantile_caps_small_sample_correction():
    assert _conformal_quantile([0.4], 0.1) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="observed targets"):
        _conformal_quantile([], 0.1)


def test_conformal_regression_uses_observed_count_per_task():
    calibrator = ConformalRegressionCalibrator.__new__(
        ConformalRegressionCalibrator
    )
    calibrator.conformal_alpha = 0.1
    calibrator.calibration_predictor = _PredictionResult(
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    )
    calibrator.calibration_data = SimpleNamespace(
        is_atom_bond_targets=False,
        targets=lambda: [[0.1, None], [0.4, None], [0.2, 1.5]],
        mask=lambda: [[True, True, True], [False, False, True]],
    )

    calibrator.calibrate()

    assert calibrator.qhats == pytest.approx([0.4, 0.5])


def test_conformal_regression_calibrates_ragged_atom_bond_rows():
    calibrator = ConformalRegressionCalibrator.__new__(
        ConformalRegressionCalibrator
    )
    calibrator.conformal_alpha = 0.1
    calibrator.calibration_predictor = _PredictionResult(
        [
            [np.array([0.0, 0.0]), np.array([1.0])],
            [np.array([0.0]), np.array([], dtype=float)],
        ],
        [
            [np.array([0.0, 0.0]), np.array([0.0])],
            [np.array([0.0]), np.array([], dtype=float)],
        ],
    )
    calibrator.calibration_data = SimpleNamespace(
        is_atom_bond_targets=True,
        targets=lambda: [
            [np.array([0.1, None], dtype=object), np.array([1.5])],
            [np.array([0.4]), np.array([], dtype=float)],
        ],
        mask=lambda: [[True, False, True], [True]],
    )

    calibrator.calibrate()

    assert calibrator.qhats == pytest.approx([0.4, 0.5])


def test_conformal_multiclass_uses_observed_count_per_task():
    calibrator = ConformalMulticlassCalibrator.__new__(
        ConformalMulticlassCalibrator
    )
    calibrator.conformal_alpha = 0.1
    calibrator.calibration_predictor = _PredictionResult(
        [
            [[0.8, 0.2], [0.5, 0.5]],
            [[0.3, 0.7], [0.5, 0.5]],
            [[0.6, 0.4], [0.1, 0.9]],
        ]
    )
    calibrator.calibration_data = SimpleNamespace(
        targets=lambda: [[0, None], [1, None], [0, 1]],
        mask=lambda: [[True, True, True], [False, False, True]],
    )

    calibrator.calibrate()

    assert calibrator.qhats == pytest.approx([-0.6, -0.9])


def test_conformal_multilabel_respects_missing_target_rows():
    calibrator = ConformalMultilabelCalibrator.__new__(
        ConformalMultilabelCalibrator
    )
    calibrator.conformal_alpha = 0.2
    calibrator.calibration_predictor = _PredictionResult(
        [[0.1, 0.8], [0.9, 0.3], [0.2, 0.4]]
    )
    calibrator.calibration_data = SimpleNamespace(
        targets=lambda: [[None, 1], [0, None], [1, 0]],
        mask=lambda: [[False, True, True], [True, False, True]],
    )

    calibrator.calibrate()

    assert calibrator.tin == pytest.approx(-0.4)
    assert calibrator.tout == pytest.approx(-0.2)


def test_default_classification_calibrator_selects_isotonic(monkeypatch):
    class DummyIsotonicCalibrator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        calibrator_module, "IsotonicCalibrator", DummyIsotonicCalibrator
    )

    calibrator = build_uncertainty_calibrator(
        calibration_method=None,
        uncertainty_method="classification",
        regression_calibrator_metric=None,
        interval_percentile=95,
        calibration_data=object(),
        calibration_data_loader=object(),
        models=[],
        scalers=[],
        num_models=1,
        dataset_type="classification",
        loss_function="binary_cross_entropy",
        uncertainty_dropout_p=0.1,
        conformal_alpha=0.1,
        dropout_sampling_size=10,
        spectra_phase_mask=None,
    )

    assert calibrator.kwargs["calibration_method"] == "isotonic"


def test_isotonic_multiclass_nll_handles_missing_targets():
    calibrator = IsotonicMulticlassCalibrator.__new__(
        IsotonicMulticlassCalibrator
    )
    calibrator.calibration_data = SimpleNamespace(is_atom_bond_targets=False)
    targets = [[0, None], [1, 1], [None, 0]]
    probabilities = [
        [[0.8, 0.2], [0.5, 0.5]],
        [[0.3, 0.7], [0.4, 0.6]],
        [[0.5, 0.5], [0.9, 0.1]],
    ]
    mask = [[True, True, False], [False, True, True]]

    nll = calibrator.nll(probabilities, probabilities, targets, mask)

    np.testing.assert_allclose(
        nll,
        [-np.mean(np.log([0.8, 0.7])), -np.mean(np.log([0.6, 0.9]))],
    )


def test_tscaling_nll_uses_standard_deviation_as_student_t_scale():
    calibrator = TScalingCalibrator.__new__(TScalingCalibrator)
    calibrator.calibration_data = SimpleNamespace(is_atom_bond_targets=False)
    calibrator.num_tasks = 1
    calibrator.num_models = 4
    predictions = [[1.0], [4.0]]
    targets = [[0.0], [2.0]]
    calibrated_standard_deviations = [[2.0], [3.0]]

    result = calibrator.nll(
        predictions,
        calibrated_standard_deviations,
        targets,
        [[True, True]],
    )

    expected = -np.mean(
        t.logpdf([1.0, 2.0], df=3, scale=[2.0, 3.0])
    )
    assert result == pytest.approx([expected])


def test_tscaling_nll_rejects_nonpositive_scale():
    calibrator = TScalingCalibrator.__new__(TScalingCalibrator)
    calibrator.calibration_data = SimpleNamespace(is_atom_bond_targets=False)
    calibrator.num_tasks = 1
    calibrator.num_models = 3

    with pytest.raises(ValueError, match='strictly positive scale'):
        calibrator.nll([[0.0]], [[0.0]], [[0.0]], [[True]])


def test_mve_weighting_applies_each_model_weight_on_the_task_axis():
    calibrator = MVEWeightingCalibrator.__new__(MVEWeightingCalibrator)
    calibrator.calibration_data = SimpleNamespace(is_atom_bond_targets=False)
    calibrator.num_models = 2
    calibrator.num_tasks = 2
    calibrator.var_weighting = np.array([[0.25, 0.8], [0.75, 0.2]])
    calibrator.scaling = np.array([2.0, 3.0])
    individual_variances = [
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        [[5.0, 50.0], [6.0, 60.0], [7.0, 70.0]],
    ]
    predictor = SimpleNamespace(
        get_uncal_preds=lambda: [[0.0, 0.0]] * 3,
        get_individual_vars=lambda: individual_variances,
    )

    predictions, standard_deviations = calibrator.apply_calibration(predictor)

    assert predictions == [[0.0, 0.0]] * 3
    expected_variances = np.array([[4.0, 18.0], [5.0, 28.0], [6.0, 38.0]])
    np.testing.assert_allclose(
        standard_deviations,
        np.sqrt(expected_variances) * np.array([2.0, 3.0]),
    )


def test_atom_bond_mve_weighting_uses_prediction_dataset(monkeypatch):
    class CalibrationData:
        is_atom_bond_targets = True

        def __getitem__(self, index):
            return SimpleNamespace(atom_targets=[object()], bond_targets=[object()])

    prediction_data = object()
    calibrator = MVEWeightingCalibrator.__new__(MVEWeightingCalibrator)
    calibrator.calibration_data = CalibrationData()
    calibrator.num_models = 2
    calibrator.num_tasks = 2
    calibrator.var_weighting = np.array([[0.25, 0.8], [0.75, 0.2]])
    calibrator.scaling = np.array([2.0, 3.0])
    individual_variances = [
        [np.array([[1.0], [2.0]]), np.array([[10.0]])],
        [np.array([[5.0], [6.0]]), np.array([[50.0]])],
    ]
    predictor = SimpleNamespace(
        test_data=prediction_data,
        get_uncal_preds=lambda: [[0.0]],
        get_individual_vars=lambda: individual_variances,
    )
    captured = {}

    def fake_reshape(values, data, natom_targets, nbond_targets):
        captured.update(
            values=values,
            data=data,
            natom_targets=natom_targets,
            nbond_targets=nbond_targets,
        )
        return 'reshaped'

    monkeypatch.setattr(calibrator_module, 'reshape_values', fake_reshape)

    _, result = calibrator.apply_calibration(predictor)

    assert result == 'reshaped'
    assert captured['data'] is prediction_data
    assert (captured['natom_targets'], captured['nbond_targets']) == (1, 1)
    np.testing.assert_allclose(captured['values'][0], [[4.0], [np.sqrt(5.0) * 2]])
    np.testing.assert_allclose(captured['values'][1], [[np.sqrt(18.0) * 3]])


def test_atom_bond_platt_calibration_preserves_prediction_rows():
    class CalibrationData:
        is_atom_bond_targets = True

        def __getitem__(self, index):
            return SimpleNamespace(atom_targets=[object()], bond_targets=[object()])

    class PredictionData:
        number_of_atoms = [[2], [1]]
        number_of_bonds = [[1], [0]]

        def __len__(self):
            return 2

    raw_predictions = [
        [np.array([0.2, 0.8]), np.array([0.4])],
        [np.array([0.6]), np.array([], dtype=float)],
    ]
    predictor = SimpleNamespace(
        test_data=PredictionData(),
        get_uncal_preds=lambda: raw_predictions,
    )
    calibrator = PlattCalibrator.__new__(PlattCalibrator)
    calibrator.calibration_data = CalibrationData()
    calibrator.num_tasks = 2
    calibrator.platt_a = np.array([1.0, 1.0])
    calibrator.platt_b = np.array([0.0, 0.0])

    returned_predictions, calibrated = calibrator.apply_calibration(predictor)

    assert returned_predictions is raw_predictions
    assert calibrated.shape == (2, 2)
    np.testing.assert_allclose(calibrated[0, 0], [0.2, 0.8])
    np.testing.assert_allclose(calibrated[1, 0], [0.6])
    np.testing.assert_allclose(calibrated[0, 1], [0.4])
    assert calibrated[1, 1].size == 0
