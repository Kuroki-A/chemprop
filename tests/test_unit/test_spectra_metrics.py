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
