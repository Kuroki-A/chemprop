from types import SimpleNamespace

import numpy as np
import pytest
import torch

import chemprop.uncertainty.uncertainty_predictor as predictor_module
from chemprop.train.predict import predict as model_predict
from chemprop.uncertainty.uncertainty_predictor import (
    DropoutPredictor,
    NoUncertaintyPredictor,
    RoundRobinSpectraPredictor,
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
