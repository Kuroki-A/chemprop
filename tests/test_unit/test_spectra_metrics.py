import numpy as np
import pytest

from chemprop.train.metrics import bounded_mse, bounded_rmse, sid_metric, wasserstein_metric


@pytest.mark.parametrize("metric_func", [sid_metric, wasserstein_metric])
def test_spectra_metrics_average_all_batches(metric_func):
    targets = [[0.9, 0.1] for _ in range(51)]
    predictions = [[0.1, 0.9] for _ in range(50)] + [[0.9, 0.1]]

    expected = metric_func(predictions, targets, batch_size=100)
    actual = metric_func(predictions, targets, batch_size=50)

    assert actual == pytest.approx(expected)
    assert actual > 0


def test_bounded_squared_metrics_support_current_sklearn():
    targets = [1.0, 2.0]
    predictions = [3.0, 4.0]
    greater_than = [True, False]
    less_than = [False, False]

    assert bounded_mse(targets, predictions, greater_than, less_than) == pytest.approx(2.0)
    assert bounded_rmse(targets, predictions, greater_than, less_than) == pytest.approx(2 ** 0.5)


@pytest.mark.parametrize("metric_func", [sid_metric, wasserstein_metric])
@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_spectra_metrics_reject_invalid_batch_size(metric_func, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        metric_func([[0.5, 0.5]], [[0.5, 0.5]], batch_size=batch_size)


@pytest.mark.parametrize("metric_func", [sid_metric, wasserstein_metric])
@pytest.mark.parametrize("threshold", [0, -1, np.nan, np.inf])
def test_spectra_metrics_reject_invalid_threshold(metric_func, threshold):
    with pytest.raises(ValueError, match="threshold"):
        metric_func([[0.5, 0.5]], [[0.5, 0.5]], threshold=threshold)


@pytest.mark.parametrize("metric_func", [sid_metric, wasserstein_metric])
def test_spectra_metrics_reject_empty_or_nonfinite_inputs(metric_func):
    with pytest.raises(ValueError, match="at least one"):
        metric_func([], [])
    with pytest.raises(ValueError, match="same rectangular shape"):
        metric_func([[0.5, 0.5]], [[0.5]])
    with pytest.raises(ValueError, match="finite"):
        metric_func([[np.inf, 0.5]], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="positive"):
        metric_func([[0.0, 0.0]], [[0.5, 0.5]], threshold=None)


def test_sid_rejects_zero_target_instead_of_returning_nan():
    with pytest.raises(ValueError, match="strictly positive.*target"):
        sid_metric([[0.5, 0.5]], [[1.0, 0.0]])
