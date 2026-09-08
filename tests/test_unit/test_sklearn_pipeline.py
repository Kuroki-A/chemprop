from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

from chemprop.args import SklearnPredictArgs, SklearnTrainArgs
from chemprop.data import MoleculeDatapoint, MoleculeDataset, get_data
from chemprop.sklearn_predict import predict_sklearn
from chemprop.sklearn_train import (
    SklearnModelBundle,
    _build_sklearn_model,
    _fit_single_task_models,
    impute_sklearn,
    load_sklearn_checkpoint,
    predict,
    run_sklearn,
)


DATA_PATH = Path(__file__).parents[1] / 'data' / 'regression.csv'


def _molecule_dataset(rows):
    return MoleculeDataset(
        [
            MoleculeDatapoint(smiles=[smiles], targets=list(targets))
            for smiles, targets in rows
        ]
    )


@pytest.mark.parametrize(
    ('option', 'value', 'message'),
    [
        ('--radius', '-1', 'radius'),
        ('--num_bits', '0', 'num_bits'),
        ('--num_trees', '0', 'num_trees'),
    ],
)
def test_sklearn_train_rejects_invalid_estimator_sizes(option, value, message):
    with pytest.raises(ValueError, match=message):
        SklearnTrainArgs().parse_args([
            '--data_path', str(DATA_PATH),
            '--dataset_type', 'regression',
            '--model_type', 'random_forest',
            option, value,
        ])


def test_sklearn_predict_validates_common_sizes(tmp_path):
    test_path = tmp_path / 'test.csv'
    test_path.write_text('smiles\nCC\n', encoding='utf-8')

    with pytest.raises(ValueError, match='number_of_molecules'):
        SklearnPredictArgs().parse_args([
            '--test_path', str(test_path),
            '--preds_path', str(tmp_path / 'preds.csv'),
            '--checkpoint_path', str(tmp_path / 'model.pkl'),
            '--number_of_molecules', '0',
        ])


def test_classification_estimators_return_positive_class_probabilities():
    features = np.asarray(
        [[-3.0], [-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0], [3.0]]
    )
    targets = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    svm_args = SimpleNamespace(
        dataset_type="classification",
        model_type="svm",
        seed=19,
        num_trees=4,
        class_weight="balanced",
    )
    svm = _build_sklearn_model(svm_args)
    assert isinstance(svm, SVC)
    assert svm.probability is True
    assert svm.class_weight == "balanced"
    assert svm.random_state == 19
    svm.fit(features, targets)

    expected = svm.predict_proba(features)[:, np.flatnonzero(svm.classes_ == 1)[0]]
    actual = np.asarray(
        predict(svm, "svm", "classification", features), dtype=float
    )[:, 0]
    np.testing.assert_allclose(actual, expected)
    assert np.all((0 <= actual) & (actual <= 1))

    bundle = SklearnModelBundle(
        models=[svm],
        train_args={"task_names": ["activity"]},
        single_task=False,
    )
    np.testing.assert_allclose(
        predict(bundle, "svm", "classification", features),
        expected.reshape(-1, 1),
    )

    multi_targets = np.column_stack([targets, np.zeros(len(targets), dtype=int)])
    forest = RandomForestClassifier(n_estimators=5, random_state=7).fit(
        features, multi_targets
    )
    forest_predictions = np.asarray(
        predict(forest, "random_forest", "classification", features)
    )
    assert forest_predictions.shape == (len(features), 2)
    assert np.all(forest_predictions[:, 1] == 0)


def test_svm_single_task_imputation_uses_probability_predictions():
    features = np.asarray(
        [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0], [0.25]]
    )
    targets = [[0], [0], [0], [1], [1], [1], [None]]
    data = MoleculeDataset(
        [
            MoleculeDatapoint(
                smiles=["C"], features=row_features, targets=row_targets
            )
            for row_features, row_targets in zip(features, targets)
        ]
    )
    args = SimpleNamespace(
        dataset_type="classification",
        model_type="svm",
        impute_mode="single_task",
        task_names=["activity"],
        seed=11,
        num_trees=4,
        class_weight=None,
    )

    imputed = impute_sklearn(_build_sklearn_model(args), data, args)

    assert imputed[-1][0] in {0, 1}
    assert all(row[0] is not None for row in imputed)


def test_single_task_svm_supports_multitask_data_and_uses_row_weights(
    monkeypatch,
):
    features = np.asarray(
        [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]]
    )
    data = MoleculeDataset(
        [
            MoleculeDatapoint(
                smiles=["C"],
                features=row_features,
                targets=[index / 2, None if index == 1 else index / 3],
                data_weight=float(index + 1),
            )
            for index, row_features in enumerate(features)
        ]
    )
    args = SimpleNamespace(
        dataset_type="regression",
        model_type="svm",
        task_names=["first", "second"],
        seed=3,
        num_trees=4,
        class_weight=None,
    )

    observed_weights = []
    original_fit = __import__("sklearn.svm", fromlist=["SVR"]).SVR.fit

    def capture_fit(estimator, fit_features, fit_targets, **kwargs):
        observed_weights.append(np.asarray(kwargs["sample_weight"]))
        return original_fit(estimator, fit_features, fit_targets, **kwargs)

    monkeypatch.setattr("sklearn.svm.SVR.fit", capture_fit)
    models = _fit_single_task_models(_build_sklearn_model(args), data, args)

    assert len(models) == 2
    np.testing.assert_array_equal(observed_weights[0], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(observed_weights[1], [1, 3, 4, 5, 6])
    assert np.asarray(
        predict(
            SklearnModelBundle(
                models=models,
                train_args={"task_names": args.task_names},
                single_task=True,
            ),
            "svm",
            "regression",
            features,
        )
    ).shape == (len(features), 2)


def test_legacy_uncalibrated_svm_checkpoint_is_rejected():
    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    model = SVC().fit(features, [0, 0, 1, 1])

    with pytest.raises(ValueError, match="must be retrained"):
        predict(model, "svm", "classification", features)


def test_single_task_checkpoint_round_trip_and_fit_once(
    tmp_path: Path, monkeypatch
):
    data_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "smiles": ["C", "CC", "CCC"],
            "first": [1.0, 2.0, 3.0],
            "second": [2.0, 4.0, 6.0],
        }
    ).to_csv(data_path, index=False)

    train_data = _molecule_dataset(
        [
            ("C", (1.0, 2.0)),
            ("CC", (2.0, None)),
            ("CCC", (3.0, 6.0)),
            ("CCCC", (4.0, 8.0)),
            ("CCO", (5.0, None)),
            ("CCN", (6.0, 12.0)),
        ]
    )
    # The second validation task intentionally has no labels. Its score must
    # remain in position as nan and must not trigger an estimator refit.
    val_data = _molecule_dataset(
        [("CO", (2.5, None)), ("CN", (3.5, None))]
    )
    test_data = _molecule_dataset(
        [("CCCl", (4.5, 9.0)), ("CCBr", (5.5, 11.0))]
    )
    all_data = _molecule_dataset(
        [("C", (1.0, 2.0)), ("CC", (2.0, 4.0)), ("CCC", (3.0, 6.0))]
    )

    sklearn_train_module = __import__("chemprop.sklearn_train", fromlist=["dummy"])
    monkeypatch.setattr(
        sklearn_train_module,
        "split_data",
        lambda **_: (train_data, val_data, test_data),
    )

    fit_count = 0
    original_fit = RandomForestRegressor.fit

    def counting_fit(estimator, *fit_args, **fit_kwargs):
        nonlocal fit_count
        fit_count += 1
        return original_fit(estimator, *fit_args, **fit_kwargs)

    monkeypatch.setattr(RandomForestRegressor, "fit", counting_fit)

    save_dir = tmp_path / "model"
    save_dir.mkdir()
    train_args = SklearnTrainArgs().parse_args(
        [
            "--data_path",
            str(data_path),
            "--dataset_type",
            "regression",
            "--model_type",
            "random_forest",
            "--single_task",
            "--save_dir",
            str(save_dir),
            "--num_trees",
            "4",
            "--num_bits",
            "64",
            "--no_cuda",
            "--quiet",
        ]
    )

    valid_scores, test_scores = run_sklearn(
        train_args, all_data, fold_num=0
    )

    # One fit per task, with no additional validation/test fits.
    assert fit_count == 2
    assert np.isfinite(valid_scores["rmse"][0])
    assert np.isnan(valid_scores["rmse"][1])
    assert np.all(np.isfinite(test_scores["rmse"]))

    checkpoint_path = save_dir / "model.pkl"
    bundle = load_sklearn_checkpoint(str(checkpoint_path))
    assert bundle.single_task
    assert len(bundle.models) == 2
    assert bundle.train_args["task_names"] == ["first", "second"]
    assert not list(save_dir.glob("*.tmp"))

    predict_path = tmp_path / "predict.csv"
    pd.DataFrame({"smiles": ["COC", "CCF"]}).to_csv(
        predict_path, index=False
    )
    preds_path = tmp_path / "preds.csv"
    predict_args = SklearnPredictArgs().parse_args(
        [
            "--test_path",
            str(predict_path),
            "--preds_path",
            str(preds_path),
            "--checkpoint_path",
            str(checkpoint_path),
        ]
    )
    predict_sklearn(predict_args)

    prediction_table = pd.read_csv(preds_path)
    assert prediction_table.columns.tolist() == ["smiles", "first", "second"]
    assert prediction_table.shape == (2, 3)
    assert np.all(np.isfinite(prediction_table[["first", "second"]]))


def test_run_sklearn_uses_separate_splits_and_checkpoint_task_order(
    tmp_path: Path, monkeypatch
):
    main_path = tmp_path / "main.csv"
    val_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "smiles": ["C", "CC", "CCC", "CCCC", "CCO", "CCN"],
            "first": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "second": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    ).to_csv(main_path, index=False)
    # Deliberately reverse target columns in both separate files. Passing the
    # main checkpoint task order to get_data must restore [first, second].
    pd.DataFrame(
        {
            "smiles": ["CO", "CN"],
            "second": [20.0, 40.0],
            "first": [10.0, 30.0],
        }
    ).to_csv(val_path, index=False)
    pd.DataFrame(
        {
            "smiles": ["CCCl", "CCBr"],
            "second": [60.0, 80.0],
            "first": [50.0, 70.0],
        }
    ).to_csv(test_path, index=False)

    save_dir = tmp_path / "separate_model"
    save_dir.mkdir()
    args = SklearnTrainArgs().parse_args(
        [
            "--data_path",
            str(main_path),
            "--separate_val_path",
            str(val_path),
            "--separate_test_path",
            str(test_path),
            "--dataset_type",
            "regression",
            "--model_type",
            "random_forest",
            "--save_dir",
            str(save_dir),
            "--num_trees",
            "4",
            "--num_bits",
            "64",
            "--no_cuda",
            "--quiet",
        ]
    )
    main_data = get_data(path=str(main_path), args=args)

    sklearn_train_module = __import__("chemprop.sklearn_train", fromlist=["dummy"])

    def unexpected_split(**_):
        raise AssertionError("Both separate splits must bypass split_data")

    monkeypatch.setattr(
        sklearn_train_module,
        "split_data",
        unexpected_split,
    )
    observed_targets = []
    original_evaluate = sklearn_train_module._evaluate_multi_task_model

    def capture_evaluation(model, dataset, metrics, train_args, logger=None):
        observed_targets.append(dataset.targets())
        return original_evaluate(model, dataset, metrics, train_args, logger)

    monkeypatch.setattr(
        sklearn_train_module, "_evaluate_multi_task_model", capture_evaluation
    )

    valid_scores, test_scores = run_sklearn(args, main_data, fold_num=0)

    assert observed_targets == [
        [[10.0, 20.0], [30.0, 40.0]],
        [[50.0, 60.0], [70.0, 80.0]],
    ]
    assert np.all(np.isfinite(valid_scores["rmse"]))
    assert np.all(np.isfinite(test_scores["rmse"]))
    assert load_sklearn_checkpoint(str(save_dir / "model.pkl")).train_args[
        "task_names"
    ] == ["first", "second"]
