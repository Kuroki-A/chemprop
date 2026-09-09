from types import SimpleNamespace

import numpy as np
import pytest
import torch

import chemprop.uncertainty.uncertainty_predictor as predictor_module
from chemprop.train.predict import predict as model_predict
from chemprop.uncertainty.uncertainty_predictor import (
    ClassPredictor,
    ConformalQuantileRegressionPredictor,
    ConformalRegressionPredictor,
    DirichletPredictor,
    DropoutPredictor,
    EnsemblePredictor,
    EvidentialAleatoricPredictor,
    EvidentialEpistemicPredictor,
    EvidentialTotalPredictor,
    MVEPredictor,
    NoUncertaintyPredictor,
    RoundRobinSpectraPredictor,
    _update_running_moments,
)


def _predictor_kwargs(**overrides):
    values = dict(
        test_data=SimpleNamespace(),
        test_data_loader=object(),
        models=[],
        scalers=[],
        num_models=1,
        dataset_type="regression",
        loss_function="mse",
        uncertainty_dropout_p=0.1,
        conformal_alpha=0.1,
        dropout_sampling_size=2,
        individual_ensemble_predictions=False,
        spectra_phase_mask=None,
    )
    values.update(overrides)
    return values


_CLASS_SIZE_PREDICTORS = (
    (EnsemblePredictor, "mse", "plain"),
    (ClassPredictor, "mse", "plain"),
    (DirichletPredictor, "dirichlet", "dirichlet"),
)


def _classification_model(train_class_sizes):
    return SimpleNamespace(
        is_atom_bond_targets=False,
        train_class_sizes=train_class_sizes,
    )


def _install_classification_predictions(monkeypatch, result_kind):
    result = [[0.25]]
    if result_kind == "dirichlet":
        result = (result, np.asarray([[[2.0, 3.0]]]))
    monkeypatch.setattr(
        predictor_module,
        "predict",
        lambda **_kwargs: result,
    )


@pytest.mark.parametrize(
    "predictor_class,loss_function,result_kind", _CLASS_SIZE_PREDICTORS,
)
@pytest.mark.parametrize(
    "class_sizes",
    [
        (None, [[8, 2]]),
        ([[8, 2]], None),
    ],
    ids=["missing-first", "missing-later"],
)
def test_classification_ensemble_rejects_mixed_train_class_size_presence(
    monkeypatch, predictor_class, loss_function, result_kind, class_sizes,
):
    _install_classification_predictions(monkeypatch, result_kind)
    models = [_classification_model(value) for value in class_sizes]

    with pytest.raises(
        ValueError,
        match="inconsistent train_class_sizes metadata.*Do not mix legacy",
    ):
        predictor_class(
            **_predictor_kwargs(
                models=models,
                scalers=[(None,) * 5] * 2,
                num_models=2,
                dataset_type="classification",
                loss_function=loss_function,
            )
        )


@pytest.mark.parametrize(
    "predictor_class,loss_function,result_kind", _CLASS_SIZE_PREDICTORS,
)
def test_classification_ensemble_rejects_train_class_size_shape_mismatch(
    monkeypatch, predictor_class, loss_function, result_kind,
):
    _install_classification_predictions(monkeypatch, result_kind)
    models = [
        _classification_model([[8, 2]]),
        _classification_model([[4, 1], [3, 2]]),
    ]

    with pytest.raises(ValueError, match="train_class_sizes has shape.*require"):
        predictor_class(
            **_predictor_kwargs(
                models=models,
                scalers=[(None,) * 5] * 2,
                num_models=2,
                dataset_type="classification",
                loss_function=loss_function,
            )
        )


@pytest.mark.parametrize(
    "predictor_class,loss_function,result_kind", _CLASS_SIZE_PREDICTORS,
)
def test_classification_ensemble_records_every_valid_train_class_size(
    monkeypatch, predictor_class, loss_function, result_kind,
):
    _install_classification_predictions(monkeypatch, result_kind)
    models = [
        _classification_model([[8, 2]]),
        _classification_model([[7, 3]]),
    ]

    predictor = predictor_class(
        **_predictor_kwargs(
            models=models,
            scalers=[(None,) * 5] * 2,
            num_models=2,
            dataset_type="classification",
            loss_function=loss_function,
        )
    )

    assert predictor.train_class_sizes == [[[8.0, 2.0]], [[7.0, 3.0]]]


@pytest.mark.parametrize(
    "models,scalers",
    [([], []), ([SimpleNamespace()], []), ([], [(None,) * 5])],
)
def test_predictor_rejects_model_scaler_counts_that_do_not_match_num_models(
    models, scalers,
):
    with pytest.raises(ValueError, match="must both match num_models=1"):
        NoUncertaintyPredictor(
            **_predictor_kwargs(models=models, scalers=scalers),
        )


def test_dropout_predictor_accepts_preloaded_lists(monkeypatch):
    outputs = iter(([[1.0], [3.0]], [[3.0], [5.0]]))
    monkeypatch.setattr(
        predictor_module,
        "predict",
        lambda **_kwargs: next(outputs),
    )
    model = SimpleNamespace(is_atom_bond_targets=False)

    predictor = DropoutPredictor(
        **_predictor_kwargs(
            models=[model],
            scalers=[(None, None, None, None, None)],
        ),
    )

    assert predictor.get_uncal_preds() == [[2.0], [4.0]]
    assert predictor.get_uncal_vars() == [[1.0], [1.0]]


def test_roundrobin_predictor_requires_spectra_dataset():
    with pytest.raises(ValueError, match="requires the spectra dataset type"):
        RoundRobinSpectraPredictor(
            **_predictor_kwargs(num_models=2, dataset_type="regression"),
        )


@pytest.mark.parametrize("num_models", [0, -1, True])
def test_predictor_rejects_invalid_num_models(num_models):
    with pytest.raises(ValueError, match="num_models"):
        NoUncertaintyPredictor(
            **_predictor_kwargs(num_models=num_models),
        )


def test_mc_dropout_probability_is_restored_after_prediction():
    class EmptyPredictionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.25)
            self.is_atom_bond_targets = False
            self.loss_function = "mse"

    model = EmptyPredictionModel()

    assert model_predict(model, [], dropout_prob=0.8) == []
    assert model.dropout.p == pytest.approx(0.25)
    assert not model.dropout.training


@pytest.mark.parametrize("dropout_probability", [-0.1, 1, np.nan, np.inf])
def test_predict_rejects_invalid_dropout_probability(dropout_probability):
    model = torch.nn.Sequential(torch.nn.Dropout(0.25))
    with pytest.raises(ValueError, match="dropout_prob"):
        model_predict(model, [], dropout_prob=dropout_probability)


class _RaggedAtomBondData:
    """Minimal two-row dataset with different atom and bond cardinalities."""

    number_of_atoms = [[2], [1]]
    number_of_bonds = [[1], [0]]

    def __len__(self):
        return 2


def _atom_bond_model(loss_function="mse"):
    return SimpleNamespace(
        is_atom_bond_targets=True,
        atom_targets=["atom_task"],
        bond_targets=["bond_task"],
        loss_function=loss_function,
        train_class_sizes=None,
    )


def _ragged_predictions(offset=0.0):
    return [
        np.array([[1.0 + offset], [2.0 + offset], [3.0 + offset]]),
        np.array([[4.0 + offset]]),
    ]


def _ragged_uncertainty_result(kind, offset=0.0):
    predictions = _ragged_predictions(offset)
    if kind == "mve":
        variances = [np.full_like(task, 0.25) for task in predictions]
        return predictions, variances
    if kind == "evidential":
        lambdas = [np.full_like(task, 2.0) for task in predictions]
        alphas = [np.full_like(task, 3.0) for task in predictions]
        betas = [np.full_like(task, 4.0) for task in predictions]
        return predictions, lambdas, alphas, betas
    if kind == "dirichlet":
        alphas = [
            np.full((len(task), task.shape[1], 2), 2.0)
            for task in predictions
        ]
        return predictions, alphas
    return predictions


def _assert_ragged_atom_bond_contract(values):
    assert isinstance(values, np.ndarray)
    assert values.dtype == object
    assert values.shape == (2, 2)
    assert [len(values[0, 0]), len(values[1, 0])] == [2, 1]
    assert [len(values[0, 1]), len(values[1, 1])] == [1, 0]
    for cell in values.flat:
        assert isinstance(cell, np.ndarray)
        assert cell.ndim == 1


@pytest.mark.parametrize(
    (
        "predictor_class",
        "dataset_type",
        "loss_function",
        "result_kind",
        "num_models",
    ),
    [
        (NoUncertaintyPredictor, "regression", "mse", "plain", 1),
        (MVEPredictor, "regression", "mve", "mve", 1),
        (
            EvidentialTotalPredictor,
            "regression",
            "evidential",
            "evidential",
            1,
        ),
        (
            EvidentialAleatoricPredictor,
            "regression",
            "evidential",
            "evidential",
            1,
        ),
        (
            EvidentialEpistemicPredictor,
            "regression",
            "evidential",
            "evidential",
            1,
        ),
        (EnsemblePredictor, "regression", "mse", "plain", 2),
        (DropoutPredictor, "regression", "mse", "plain", 1),
        (ClassPredictor, "classification", "mse", "plain", 1),
        (DirichletPredictor, "classification", "dirichlet", "dirichlet", 1),
    ],
)
def test_supported_atom_bond_uncertainty_predictors_preserve_ragged_contract(
    monkeypatch,
    predictor_class,
    dataset_type,
    loss_function,
    result_kind,
    num_models,
):
    """Every supported method returns row-major object cells, never ragged ndarray."""
    model = _atom_bond_model(loss_function)
    monkeypatch.setattr(
        predictor_module,
        "predict",
        lambda **_kwargs: _ragged_uncertainty_result(result_kind),
    )
    scaler = (None, None, None, None, None)

    predictor = predictor_class(
        **_predictor_kwargs(
            test_data=_RaggedAtomBondData(),
            models=[model] * num_models,
            scalers=[scaler] * num_models,
            num_models=num_models,
            dataset_type=dataset_type,
            loss_function=loss_function,
        )
    )

    _assert_ragged_atom_bond_contract(predictor.get_uncal_preds())
    _assert_ragged_atom_bond_contract(predictor.get_uncal_output())


def test_mve_ensemble_does_not_mutate_first_models_variance(monkeypatch):
    first_variance = [
        np.array([[0.25], [0.5], [0.75]]),
        np.array([[1.25]]),
    ]
    first_snapshot = [task.copy() for task in first_variance]
    results = iter(
        [
            (_ragged_predictions(), first_variance),
            (
                _ragged_predictions(offset=2.0),
                [np.full((3, 1), 2.0), np.full((1, 1), 3.0)],
            ),
        ]
    )
    monkeypatch.setattr(
        predictor_module, "predict", lambda **_kwargs: next(results),
    )
    model = _atom_bond_model("mve")
    scaler = (None, None, None, None, None)

    predictor = MVEPredictor(
        **_predictor_kwargs(
            test_data=_RaggedAtomBondData(),
            models=[model, model],
            scalers=[scaler, scaler],
            num_models=2,
            loss_function="mve",
        )
    )

    for actual, expected in zip(first_variance, first_snapshot):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(predictor.get_individual_vars()[0], first_snapshot):
        np.testing.assert_array_equal(actual, expected)


def test_welford_moments_remain_accurate_for_large_prediction_offsets():
    base = 1.0e12
    mean = np.array([[base - 1.0]])
    m2 = np.zeros_like(mean)

    mean, m2 = _update_running_moments(
        mean,
        m2,
        [[base + 1.0]],
        count=2,
        is_atom_bond=False,
        label="Predictions",
    )

    assert mean.item() == base
    assert (m2 / 2).item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("predictor_class", "result_kind", "loss_function", "message"),
    [
        (MVEPredictor, "mve", "mve", "MVE variances"),
        (
            EvidentialTotalPredictor,
            "evidential",
            "evidential",
            "Evidential lambdas",
        ),
        (
            EvidentialAleatoricPredictor,
            "evidential",
            "evidential",
            "Evidential lambdas",
        ),
        (
            EvidentialEpistemicPredictor,
            "evidential",
            "evidential",
            "Evidential lambdas",
        ),
    ],
)
def test_atom_bond_uncertainty_rejects_misaligned_ancillary_shapes(
    monkeypatch, predictor_class, result_kind, loss_function, message,
):
    result = list(_ragged_uncertainty_result(result_kind))
    # The first ancillary task now has two values, while its prediction has three.
    result[1] = [np.ones((2, 1)), np.ones((1, 1))]
    monkeypatch.setattr(
        predictor_module, "predict", lambda **_kwargs: tuple(result),
    )
    model = _atom_bond_model(loss_function)

    with pytest.raises(ValueError, match=message):
        predictor_class(
            **_predictor_kwargs(
                test_data=_RaggedAtomBondData(),
                models=[model],
                scalers=[(None, None, None, None, None)],
                loss_function=loss_function,
            )
        )


@pytest.mark.parametrize(
    "predictor_class",
    [ConformalQuantileRegressionPredictor, ConformalRegressionPredictor],
)
def test_conformal_atom_bond_rejection_happens_before_model_prediction(
    monkeypatch, predictor_class,
):
    prediction_called = False

    def unexpected_prediction(**_kwargs):
        nonlocal prediction_called
        prediction_called = True
        raise AssertionError("prediction must not start")

    monkeypatch.setattr(predictor_module, "predict", unexpected_prediction)

    with pytest.raises(NotImplementedError, match="atom/bond"):
        predictor_class(
            **_predictor_kwargs(
                test_data=_RaggedAtomBondData(),
                models=[_atom_bond_model()],
                scalers=[(None, None, None, None, None)],
            )
        )

    assert prediction_called is False
