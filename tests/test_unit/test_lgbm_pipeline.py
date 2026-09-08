import csv
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import average_precision_score


pytest.importorskip("lightgbm")

from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.train.make_predictions import (
    _validate_ensemble_train_args,
    load_data,
    load_model_lgbm,
    make_predictions_lgbm,
)
from chemprop.features import get_features_generators_metadata
from chemprop.train.run_training_lgbm import (
    _lightgbm_feval,
    _lightgbm_metric,
    build_frozen_lgbm_encoder,
    encode_lgbm_features,
    evaluate_lgbm_predictions,
    predict_task_boosters,
    train_task_boosters,
)
from chemprop.train.metrics import prc_auc
from chemprop.utils import (
    LightGBMCheckpointError,
    load_checkpoint_lgbm,
    save_checkpoint_lgbm,
)


SMILES = [
    "C",
    "CC",
    "CCC",
    "CCCC",
    "CO",
    "CCO",
    "CCCO",
    "CN",
    "CCN",
    "CCCN",
    "C=C",
    "CC=C",
    "C#N",
    "CC#N",
    "c1ccccc1",
    "Cc1ccccc1",
    "O=C=O",
    "CC(=O)O",
    "C1CCCCC1",
    "CC1CCCCC1",
    "N#N",
    "O",
    "N",
    "S",
]


def _training_args(tmp_path: Path, dataset_type: str, task_names) -> TrainArgs:
    data_path = tmp_path / f"{dataset_type}.csv"
    data_path.write_text("smiles," + ",".join(task_names) + "\n")
    args = TrainArgs().parse_args(
        [
            "--data_path",
            str(data_path),
            "--dataset_type",
            dataset_type,
            "--model_type",
            "lgbm",
            "--hidden_size",
            "12",
            "--ffn_hidden_size",
            "12",
            "--depth",
            "2",
            "--batch_size",
            "8",
            "--num_workers",
            "0",
            "--no_cuda",
            "--quiet",
        ]
    )
    args.task_names = list(task_names)
    args.features_size = None
    args.lgbm_num_boost_round = 30
    args.lgbm_early_stopping_rounds = 5
    args.lgbm_num_threads = 1
    args.lgbm_min_data_in_leaf = 2
    return args


def _molecule_dataset(smiles, targets) -> MoleculeDataset:
    return MoleculeDataset(
        [
            MoleculeDatapoint(smiles=[smile], targets=list(row_targets))
            for smile, row_targets in zip(smiles, targets)
        ]
    )


def test_lgbm_regression_bundle_fresh_process_round_trip_and_empty_input(
    tmp_path: Path,
):
    task_names = ["target_a", "target_b"]
    args = _training_args(tmp_path, "regression", task_names)
    targets = [
        [index / 7, None if index % 4 == 0 else np.sin(index)]
        for index in range(len(SMILES))
    ]
    train_data = _molecule_dataset(SMILES[:16], targets[:16])
    val_data = _molecule_dataset(SMILES[16:20], targets[16:20])
    test_data = _molecule_dataset(SMILES[20:], targets[20:])

    encoder = build_frozen_lgbm_encoder(args)
    train_features = encode_lgbm_features(encoder, train_data, 8, 0)
    val_features = encode_lgbm_features(encoder, val_data, 8, 0)
    test_features = encode_lgbm_features(encoder, test_data, 8, 0)
    train_targets = np.asarray(train_data.targets(), dtype=float)
    val_targets = np.asarray(val_data.targets(), dtype=float)

    checkpoint_paths = []
    member_predictions = []
    for model_index in range(2):
        boosters = train_task_boosters(
            args,
            train_features,
            train_targets,
            val_features,
            val_targets,
            seed=17 + model_index,
        )
        member_predictions.append(predict_task_boosters(boosters, test_features))
        checkpoint_path = tmp_path / f"model_{model_index}.pkl"
        saved_path = save_checkpoint_lgbm(
            str(checkpoint_path),
            encoder,
            boosters,
            args=args,
            model_index=model_index,
            seed=17 + model_index,
        )
        assert os.path.isabs(saved_path)
        checkpoint_paths.append(saved_path)

    reloaded = load_checkpoint_lgbm(checkpoint_paths[0])
    assert not reloaded.encoder.training
    assert all(not parameter.requires_grad for parameter in reloaded.encoder.parameters())
    reloaded_features = encode_lgbm_features(reloaded.encoder, test_data, 8, 0)
    np.testing.assert_array_equal(reloaded_features, test_features)
    np.testing.assert_allclose(
        predict_task_boosters(reloaded.task_boosters, reloaded_features),
        member_predictions[0],
        rtol=0,
        atol=0,
    )

    with open(checkpoint_paths[0], "rb") as checkpoint_file:
        corrupted_bundle = pickle.load(checkpoint_file)
    corrupted_bundle["metadata"]["task_names"] = ["target_b", "target_a"]
    corrupted_path = tmp_path / "corrupted_task_names.pkl"
    with corrupted_path.open("wb") as checkpoint_file:
        pickle.dump(corrupted_bundle, checkpoint_file)
    with pytest.raises(LightGBMCheckpointError, match="metadata"):
        load_checkpoint_lgbm(str(corrupted_path))

    test_path = tmp_path / "test.csv"
    with test_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["smiles"])
        writer.writerows([[smile] for smile in SMILES[20:]])
    preds_path = tmp_path / "fresh_process_predictions.csv"
    command = [
        sys.executable,
        "-c",
        "from chemprop.train.make_predictions import chemprop_predict; chemprop_predict()",
        "--test_path",
        str(test_path),
        "--preds_path",
        str(preds_path),
        "--checkpoint_paths",
        *checkpoint_paths,
        "--individual_ensemble_predictions",
        "--num_workers",
        "0",
        "--no_cuda",
    ]
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    with preds_path.open(newline="") as file:
        saved_rows = list(csv.DictReader(file))
    expected_members = np.stack(member_predictions, axis=2)
    expected_ensemble = np.mean(expected_members, axis=2)
    assert not np.array_equal(expected_members[:, :, 0], expected_members[:, :, 1])
    np.testing.assert_allclose(
        [[float(row[name]) for name in task_names] for row in saved_rows],
        expected_ensemble,
        rtol=0,
        atol=1e-12,
    )
    for model_index in range(2):
        np.testing.assert_allclose(
            [
                [float(row[f"{name}_model_{model_index}"]) for name in task_names]
                for row in saved_rows
            ],
            expected_members[:, :, model_index],
            rtol=0,
            atol=1e-12,
        )

    predict_args = PredictArgs().parse_args(
        [
            "--test_path",
            str(test_path),
            "--preds_path",
            str(tmp_path / "empty.csv"),
            "--checkpoint_paths",
            *checkpoint_paths,
            "--num_workers",
            "0",
            "--no_cuda",
        ]
    )
    assert predict_args.model_type == "lgbm"
    model_objects = load_model_lgbm(predict_args, generator=False)
    assert make_predictions_lgbm(
        predict_args,
        smiles=[],
        model_objects=model_objects,
    ) == []
    assert (tmp_path / "empty.csv").read_text().splitlines()[0] == (
        "smiles,target_a,target_b"
    )
    predict_args.preds_path = str(tmp_path / "invalid.csv")
    invalid_predictions = make_predictions_lgbm(
        predict_args,
        smiles=[["not_a_smiles"]],
        model_objects=model_objects,
        return_index_dict=True,
    )
    assert invalid_predictions == {
        0: ["Invalid SMILES", "Invalid SMILES"]
    }


def test_lgbm_classification_multitask_missing_targets_and_seed_reproducibility():
    rng = np.random.default_rng(4)
    train_features = rng.normal(size=(48, 10))
    val_features = rng.normal(size=(16, 10))
    train_targets = np.column_stack(
        [
            np.arange(48) % 2,
            np.where(np.arange(48) % 5 == 0, np.nan, (np.arange(48) // 2) % 2),
        ]
    )
    val_targets = np.column_stack([np.arange(16) % 2, (np.arange(16) // 2) % 2])
    args = SimpleNamespace(
        dataset_type="classification",
        metric="auc",
        num_tasks=2,
        task_names=["active_a", "active_b"],
        class_balance=True,
        quiet=True,
        num_workers=0,
        lgbm_num_boost_round=25,
        lgbm_early_stopping_rounds=5,
        lgbm_num_threads=1,
        lgbm_min_data_in_leaf=2,
    )

    first = train_task_boosters(
        args,
        train_features,
        train_targets,
        val_features,
        val_targets,
        seed=9,
    )
    second = train_task_boosters(
        args,
        train_features,
        train_targets,
        val_features,
        val_targets,
        seed=9,
    )
    assert len(first) == len(second) == 2
    first_predictions = predict_task_boosters(first, val_features)
    second_predictions = predict_task_boosters(second, val_features)
    np.testing.assert_array_equal(first_predictions, second_predictions)
    assert np.all((0 <= first_predictions) & (first_predictions <= 1))

    args.metric = "prc-auc"
    prc_boosters = train_task_boosters(
        args,
        train_features,
        train_targets,
        val_features,
        val_targets,
        seed=9,
    )
    for task_index, booster in enumerate(prc_boosters):
        predictions = booster.predict(val_features)
        assert booster.best_score["validation"]["prc-auc"] == pytest.approx(
            prc_auc(val_targets[:, task_index], predictions)
        )


@pytest.mark.parametrize(
    "dataset_type, metric, expected",
    [
        ("classification", "auc", "auc"),
        ("classification", "binary_cross_entropy", "binary_logloss"),
        ("classification", "prc-auc", "None"),
        ("regression", "rmse", "rmse"),
        ("regression", "mae", "l1"),
        ("regression", "mse", "l2"),
    ],
)
def test_lightgbm_primary_metric_mapping_is_exact(
    dataset_type, metric, expected
):
    assert _lightgbm_metric(
        SimpleNamespace(dataset_type=dataset_type, metric=metric)
    ) == expected


def test_lightgbm_prc_auc_callback_matches_chemprop_not_average_precision():
    targets = np.asarray([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    predictions = np.asarray(
        [0.25, 0.41, 0.41, 0.41, 0.49, 0.49, 0.49, 0.22, 0.22, 0.54, 0.54, 0.54]
    )
    dataset = __import__("lightgbm").Dataset(
        np.arange(len(targets)).reshape(-1, 1), label=targets
    )
    callback = _lightgbm_feval(
        SimpleNamespace(dataset_type="classification", metric="prc-auc")
    )

    name, score, higher_is_better = callback(predictions, dataset)

    assert name == "prc-auc"
    assert higher_is_better is True
    assert score == pytest.approx(prc_auc(targets, predictions))
    assert score != pytest.approx(average_precision_score(targets, predictions))


def test_lgbm_multiple_metrics_preserve_task_positions_and_reject_legacy(tmp_path: Path):
    args = SimpleNamespace(
        metrics=["rmse", "mae"],
        num_tasks=2,
        dataset_type="regression",
    )
    results = evaluate_lgbm_predictions(
        predictions=np.asarray([[1.5, 3.0], [2.5, 4.0]]),
        targets=[[1.0, None], [3.0, None]],
        args=args,
    )
    assert set(results) == {"rmse", "mae"}
    assert all(len(values) == 2 for values in results.values())
    assert np.isnan(results["rmse"][1])

    legacy_path = tmp_path / "legacy.pkl"
    with legacy_path.open("wb") as file:
        pickle.dump({"legacy": "raw Booster had no encoder state"}, file)
    with pytest.raises(LightGBMCheckpointError, match="legacy raw-Booster"):
        load_checkpoint_lgbm(str(legacy_path))


def test_ensemble_rejects_same_width_but_different_feature_schema():
    first = SimpleNamespace(features_generator_metadata={"columns": ["A", "B"]})
    second = SimpleNamespace(features_generator_metadata={"columns": ["B", "A"]})
    with pytest.raises(ValueError, match="features_generator_metadata"):
        _validate_ensemble_train_args(
            [first, second],
            ("features_generator_metadata",),
            "FFN",
        )


def test_empty_prediction_still_validates_generator_identity():
    expected_metadata = get_features_generators_metadata(
        ["morgan"], total_dimension=2048
    )
    predict_args = SimpleNamespace(
        features_generator=["fcfp"],
        selected_features_path=None,
        batch_size=1,
        num_workers=0,
    )
    train_args = SimpleNamespace(
        features_size=2048,
        features_generator_metadata=expected_metadata,
        features_source_metadata=None,
    )

    with pytest.raises(ValueError, match="feature schema"):
        load_data(predict_args, smiles=[], train_args=train_args)

    predict_args.features_generator = ["morgan"]
    train_args.features_source_metadata = {
        "schema_version": 1,
        "external_features": [],
        "phase_features": None,
        "generated_dimension": 2048,
        "total_dimension": 2048,
    }
    full_data, test_data, _, index_map = load_data(
        predict_args, smiles=[], train_args=train_args
    )
    assert len(full_data) == len(test_data) == 0
    assert index_map == {}
