"""Chemprop unit tests for chemprop/train/loss_functions.py"""
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from chemprop.train.loss_functions import (
    bounded_mse_loss,
    dirichlet_class_loss,
    evidential_loss,
    get_loss_func,
    mcc_class_loss,
    mcc_multiclass_loss,
    normal_mve,
    quantile_loss,
    sid_loss,
    wasserstein_loss,
)


# Fixtures
@pytest.fixture(params=["regression", "classification", "multiclass"])
def dataset_type(request):
    return request.param


@pytest.fixture(params=["mse", "bounded_mse", "mve", "evidential", "quantile_interval"])
def regression_function(request):
    return request.param


@pytest.fixture(params=["binary_cross_entropy", "mcc", "dirichlet"])
def classification_function(request):
    return request.param


@pytest.fixture(params=["cross_entropy", "mcc", "dirichlet"])
def multiclass_function(request):
    return request.param


@pytest.fixture(params=["sid", "wasserstein"])
def spectra_function(request):
    return request.param


# Tests
def test_get_regression_function(regression_function):
    """
    Tests the get_loss_func function for regression.
    """
    args = SimpleNamespace(
        loss_function=regression_function,
        dataset_type="regression",
    )
    assert get_loss_func(args)


def test_get_class_function(classification_function):
    """
    Tests the get_loss_func function for classification.
    """
    args = SimpleNamespace(
        loss_function=classification_function,
        dataset_type="classification",
    )
    assert get_loss_func(args)


def test_get_multiclass_function(multiclass_function):
    """
    Tests the get_loss_func function for multiclass.
    """
    args = SimpleNamespace(
        loss_function=multiclass_function,
        dataset_type="multiclass",
    )
    assert get_loss_func(args)


def test_get_spectra_function(spectra_function):
    """
    Tests the get_loss_func function for spectra.
    """
    args = SimpleNamespace(
        loss_function=spectra_function,
        dataset_type="spectra",
    )
    assert get_loss_func(args)


@pytest.mark.parametrize("loss_func", [sid_loss, wasserstein_loss])
def test_spectra_losses_are_finite_for_zero_predictions_and_empty_masks(loss_func):
    predictions = torch.zeros(2, 3, requires_grad=True)
    targets = torch.tensor([[0.2, 0.3, 0.5], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True], [False, False, False]])

    loss = loss_func(predictions, targets, mask)
    loss.sum().backward()

    assert torch.isfinite(loss).all()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_get_unsupported_function(dataset_type):
    """
    Tests the error triggering for unsupported loss functions in get_loss_func.
    """
    with pytest.raises(ValueError):
        args = SimpleNamespace(dataset_type=dataset_type, loss_function="dummy_loss")
        get_loss_func(args=args)


@pytest.mark.parametrize(
    "preds,targets,lt_targets,gt_targets,mse",
    [
        (
            torch.tensor([[-3, 2], [1, -1]], dtype=float),
            torch.zeros([2, 2], dtype=float),
            torch.zeros([2, 2], dtype=bool),
            torch.zeros([2, 2], dtype=bool),
            15,
        ),
        (
            torch.tensor([[-3, 2], [1, -1]], dtype=float),
            torch.zeros([2, 2], dtype=float),
            torch.zeros([2, 2], dtype=bool),
            torch.ones([2, 2], dtype=bool),
            10,
        ),
        (
            torch.tensor([[-3, 2], [1, -1]], dtype=float),
            torch.zeros([2, 2], dtype=float),
            torch.ones([2, 2], dtype=bool),
            torch.zeros([2, 2], dtype=bool),
            5,
        ),
    ],
)
def test_bounded_mse(preds, targets, lt_targets, gt_targets, mse):
    """
    Testing the bounded_mse loss function
    """
    loss = bounded_mse_loss(preds, targets, lt_targets, gt_targets)
    assert loss.sum() == mse


@pytest.mark.parametrize(
    "preds,targets,likelihood",
    [
        (
            torch.tensor([[0, 1]], dtype=float),
            torch.zeros([1, 1], dtype=float),
            [[0.3989]],
        )
    ],
)
def test_mve(preds, targets, likelihood):
    """
    Tests the normal_mve loss function
    """
    nll_calc = normal_mve(preds, targets)
    likelihood_calc = np.exp(-1 * nll_calc)
    np.testing.assert_array_almost_equal(likelihood, likelihood_calc, decimal=4)


@pytest.mark.parametrize(
    "alphas,target_labels,lam,expected_loss",
    [
        (torch.tensor([[2, 2]], dtype=float), torch.ones([1, 1], dtype=float), 0, [[0.6]]),
        (torch.tensor([[2, 2]], dtype=float), torch.ones([1, 1], dtype=float), 0.2, [[0.63862943]]),
    ],
)
def test_dirichlet(alphas, target_labels, lam, expected_loss):
    """
    Test on the dirichlet loss function for classification.
    Note these values were not hand derived, just testing for
    dimensional consistency.
    """
    loss = dirichlet_class_loss(alphas, target_labels, lam=lam)
    np.testing.assert_array_almost_equal(loss, expected_loss)


@pytest.mark.parametrize(
    "alphas,target_labels",
    [
        (
            torch.ones([1, 1], dtype=float),
            torch.ones([1, 1], dtype=float),
        ),
    ],
)
def test_dirichlet_wrong_dimensions(alphas, target_labels):
    """
    Test on the dirichlet loss function for classification
    for dimension errors.
    """
    with pytest.raises(RuntimeError):
        dirichlet_class_loss(alphas, target_labels)


@pytest.mark.parametrize(
    "alphas,targets,lam,expected_loss",
    [
        (torch.tensor([[2, 2, 2, 2]], dtype=float), torch.ones([1, 1], dtype=float), 0, [[1.56893861]]),
        (torch.tensor([[2, 2, 2, 2]], dtype=float), torch.ones([1, 1], dtype=float), 0.2, [[2.768938541]]),
    ],
)
def test_evidential(alphas, targets, lam, expected_loss):
    """
    Test on the evidential loss function for classification.
    Note these values were not hand derived, just testing for
    dimensional consistency.
    """
    loss = evidential_loss(alphas, targets, lam=lam)
    np.testing.assert_array_almost_equal(loss, expected_loss)


@pytest.mark.parametrize(
    "alphas,targets",
    [
        (
            torch.ones([2, 2], dtype=float),
            torch.ones([2, 2], dtype=float),
        ),
    ],
)
def test_evidential_wrong_dimensions(alphas, targets):
    """
    Test on the dirichlet loss function for classification
    for dimension errors.
    """
    with pytest.raises(RuntimeError):
        evidential_loss(alphas, targets)


@pytest.mark.parametrize(
    "preds,targets,quantiles,expected_loss",
    [
        (
            torch.tensor([0, 0.5, 1], dtype=float),
            torch.tensor([0, 0, 0], dtype=float),
            torch.tensor([0.25, 0.5, 0.75], dtype=float),
            [0, 0.25, 0.25],
        ),
        (
            torch.tensor([0, 0.5, 1], dtype=float),
            torch.tensor([0.5, 0.5, 0.5], dtype=float),
            torch.tensor([0, 1, 1], dtype=float),
            [0, 0, 0],
        ),
        (
            torch.tensor([0, 0.5, 1], dtype=float),
            torch.tensor([0.5, 0.5, 0.5], dtype=float),
            torch.tensor([1, 0, 0], dtype=float),
            [0.5, 0, 0.5],
        ),
        (
            torch.tensor([0, 0.5, 1], dtype=float),
            torch.tensor([0, 0.5, 1], dtype=float),
            torch.tensor([0, 1, 1], dtype=float),
            [0, 0, 0],
        ),
    ],
)
def test_quantile(preds, targets, quantiles, expected_loss):
    """
    Test on the evidential loss function for regression.
    """
    loss = quantile_loss(preds, targets, quantiles)
    np.testing.assert_array_almost_equal(loss, expected_loss, decimal=4)

@pytest.mark.parametrize(
    "predictions,targets,data_weights,mask,expected_loss",
    [
        (
            torch.tensor([[0.2, 0.7, 0.1], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]], dtype=float),
            torch.tensor([0, 0, 0], dtype=int),
            torch.tensor([[1], [1], [1]], dtype=float),
            torch.tensor([True, True, True], dtype=bool),
            1 - 0.0,
        ),
        (
            torch.tensor([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1]], dtype=float),
            torch.tensor([1, 2, 0], dtype=int),
            torch.tensor([[1], [1], [1]], dtype=float),
            torch.tensor([True, True, True], dtype=bool),
            1 - 0.0,
        ),
        (
            torch.tensor([[0.2, 0.7, 0.1], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]], dtype=float),
            torch.tensor([2, 0, 2], dtype=int),
            torch.tensor([[1], [1], [1]], dtype=float),
            torch.tensor([True, True, True], dtype=bool),
            1 - 0.6123724356957946,
        ),
        (
            torch.tensor([[0.2, 0.7, 0.1], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]], dtype=float),
            torch.tensor([2, 0, 2], dtype=int),
            torch.tensor([[0.5], [1], [1.5]], dtype=float),
            torch.tensor([True, True, True], dtype=bool),
            1 - 0.7462025072446364,
        ),
        (
            torch.tensor([[0.2, 0.7, 0.1], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]], dtype=float),
            torch.tensor([2, 0, 2], dtype=int),
            torch.tensor([[1], [1], [1]], dtype=float),
            torch.tensor([False, True, True], dtype=bool),
            1 - 1.0,
        ),
    ],
)
def test_multiclass_mcc(predictions, targets, data_weights, mask, expected_loss):
    """
    Test the multiclass MCC loss function on hard probabilities by comparing
    to sklearn's results.
    """
    hard_predictions = torch.nn.functional.one_hot(
        predictions.argmax(dim=1), num_classes=predictions.shape[1]
    ).to(dtype=predictions.dtype)
    loss = mcc_multiclass_loss(hard_predictions, targets, data_weights, mask)
    np.testing.assert_almost_equal(loss.item(), expected_loss)


def test_multiclass_mcc_has_nonzero_gradient():
    """The MCC training objective must be differentiable w.r.t. model logits."""
    logits = torch.tensor(
        [[1.2, -0.4, 0.1], [-0.3, 0.9, 0.2], [0.2, -0.5, 0.7], [0.6, 0.1, -0.2]],
        requires_grad=True,
    )
    predictions = torch.softmax(logits, dim=1)
    targets = torch.tensor([0, 2, 1, 0])
    data_weights = torch.ones(4, 1)
    mask = torch.ones(4, dtype=torch.bool)

    loss = mcc_multiclass_loss(predictions, targets, data_weights, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_mcc_losses_are_finite_for_degenerate_batches():
    """Missing/single-class batches must not inject NaNs into other tasks."""
    binary_predictions = torch.tensor([[0.2], [0.8]], requires_grad=True)
    binary_loss = mcc_class_loss(
        binary_predictions,
        torch.zeros(2, 1),
        torch.ones(2, 1),
        torch.ones(2, 1, dtype=torch.bool),
    )
    assert torch.equal(binary_loss, torch.ones_like(binary_loss))
    assert torch.isfinite(binary_loss).all()

    multiclass_logits = torch.randn(2, 3, requires_grad=True)
    multiclass_loss = mcc_multiclass_loss(
        torch.softmax(multiclass_logits, dim=1),
        torch.zeros(2, dtype=torch.long),
        torch.ones(2, 1),
        torch.zeros(2, dtype=torch.bool),
    )
    assert multiclass_loss.item() == pytest.approx(1.0)
    assert torch.isfinite(multiclass_loss)
