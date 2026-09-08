import pytest
import numpy as np

from chemprop.train.evaluate import evaluate_predictions
from chemprop.train.metrics import compute_hard_predictions, recall_metric,precision_metric,balanced_accuracy_metric,f1_metric,mcc_metric,prc_auc,accuracy
test_cases = [
    ([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2]),
    ([1, 1, 0, 0], [0.7, 0.8, 0.3, 0.2]),
    ([0, 0, 1, 1], [0.2, 0.4, 0.6, 0.8]),
    ([1, 0, 0, 1], [0.9, 0.1, 0.3, 0.7]),
    ([0, 1, 0, 1], [0.3, 0.6, 0.4, 0.5])
]

# Expected values
expected_auc = [1.0, 1.0, 1.0, 1.0, 1.0]
expected_prc_auc = [1.0, 1.0, 1.0, 1.0, 1.0]
expected_recall = [1.0, 1.0, 1.0, 1.0, 0.5]
expected_precision = [1.0, 1.0, 1.0, 1.0, 1.0]
expected_balanced_accuracy = [1.0, 1.0, 1.0, 1.0, 0.75]
expected_mcc = [1.0, 1.0, 1.0, 1.0, 0.5773502691896258]
expected_f1 = [1.0, 1.0, 1.0, 1.0, 0.6666666666666666]
expected_accuracy = [1.0, 1.0, 1.0, 1.0, 0.75]
@pytest.mark.parametrize("case, expected", list(zip(test_cases, expected_prc_auc)))
def test_prc_auc(case, expected):
    targets, preds = case
    assert abs(prc_auc(targets, preds) - expected) < 1e-3

@pytest.mark.parametrize("case, expected", list(zip(test_cases, expected_accuracy)))
def test_accuracy(case, expected):
    targets, preds = case
    assert abs(accuracy(targets, preds) - expected) < 1e-3

@pytest.mark.parametrize("case, expected", list(zip(test_cases, expected_recall)))
def test_recall(case, expected):
    targets, preds = case
    assert abs(recall_metric(targets, preds) - expected) < 1e-3

@pytest.mark.parametrize("case, expected", list(zip(test_cases, expected_precision)))
def test_precision(case, expected):
    targets, preds = case
    assert abs(precision_metric(targets, preds) - expected) < 1e-3

@pytest.mark.parametrize("case, expected", list(zip(test_cases, expected_balanced_accuracy)))
def test_balanced_accuracy(case, expected):
    targets, preds = case
    assert abs(balanced_accuracy_metric(targets, preds) - expected) < 1e-3

@pytest.mark.parametrize("case, expected", list(zip(test_cases, expected_f1)))
def test_f1(case, expected):
    targets, preds = case
    assert abs(f1_metric(targets, preds) - expected) < 1e-3

@pytest.mark.parametrize("case, expected", list(zip(test_cases, expected_mcc)))
def test_mcc(case, expected):
    targets, preds = case
    assert abs(mcc_metric(targets, preds) - expected) < 1e-3


def test_compute_hard_predictions_accepts_numpy_arrays():
    assert compute_hard_predictions(np.array([0.2, 0.8])) == [0, 1]
    assert compute_hard_predictions(np.array([[0.1, 0.9], [0.8, 0.2]])) == [1, 0]


@pytest.mark.parametrize(
    "metric_func, expected",
    [
        (accuracy, 0.75),
        (recall_metric, 0.5),
        (precision_metric, 1.0),
        (balanced_accuracy_metric, 0.75),
        (f1_metric, 2 / 3),
        (mcc_metric, 1 / np.sqrt(3)),
    ],
)
def test_binary_metrics_honor_threshold(metric_func, expected):
    targets = [0, 1, 1, 0]
    preds = [0.4, 0.6, 0.8, 0.2]

    assert metric_func(targets, preds, threshold=0.7) == pytest.approx(expected)


def test_single_class_only_invalidates_auc_metrics():
    results = evaluate_predictions(
        preds=[[0.1], [0.2], [0.4]],
        targets=[[0], [0], [0]],
        num_tasks=1,
        metrics=[
            "auc",
            "prc-auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "balanced_accuracy",
            "cross_entropy",
            "binary_cross_entropy",
        ],
        dataset_type="classification",
    )

    assert np.isnan(results["auc"][0])
    assert np.isnan(results["prc-auc"][0])
    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "balanced_accuracy",
        "cross_entropy",
        "binary_cross_entropy",
    ):
        assert np.isfinite(results[metric][0])


def test_constant_predictions_do_not_invalidate_defined_metrics():
    results = evaluate_predictions(
        preds=[[0.0], [0.0], [0.0], [0.0]],
        targets=[[0], [1], [0], [1]],
        num_tasks=1,
        metrics=["auc", "prc-auc", "accuracy", "binary_cross_entropy"],
        dataset_type="classification",
    )

    assert all(np.isfinite(values[0]) for values in results.values())


def test_tasks_without_targets_keep_their_metric_position():
    results = evaluate_predictions(
        preds=[[1.0, 10.0], [3.0, 20.0]],
        targets=[[1.0, None], [2.0, None]],
        num_tasks=2,
        metrics=["rmse", "mae"],
        dataset_type="regression",
    )

    assert len(results["rmse"]) == len(results["mae"]) == 2
    assert results["rmse"][0] == pytest.approx(np.sqrt(0.5))
    assert results["mae"][0] == pytest.approx(0.5)
    assert np.isnan(results["rmse"][1])
    assert np.isnan(results["mae"][1])


def test_atom_targets_without_labels_keep_their_metric_position():
    results = evaluate_predictions(
        preds=[
            np.asarray([[1.0], [3.0]]),
            np.asarray([[10.0], [20.0]]),
        ],
        targets=[
            [np.asarray([1.0]), np.asarray([None], dtype=object)],
            [np.asarray([2.0]), np.asarray([None], dtype=object)],
        ],
        num_tasks=2,
        metrics=["rmse", "mae"],
        dataset_type="regression",
        is_atom_bond_targets=True,
    )

    assert len(results["rmse"]) == len(results["mae"]) == 2
    assert results["rmse"][0] == pytest.approx(np.sqrt(0.5))
    assert results["mae"][0] == pytest.approx(0.5)
    assert np.isnan(results["rmse"][1])
    assert np.isnan(results["mae"][1])
