import csv
import importlib
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
from chemprop.data import MoleculeDatapoint, MoleculeDataset, StandardScaler
from chemprop.train.make_predictions import (
    _validate_ensemble_train_args,
    load_data,
    load_model_lgbm,
    make_predictions_lgbm,
    predict_lgbm,
)
from chemprop.features import get_features_generators_metadata
from chemprop.train.run_training_lgbm import (
    _load_split_data,
    _lightgbm_feval,
    _lightgbm_metric,
    build_frozen_lgbm_encoder,
    encode_lgbm_features,
    evaluate_lgbm_predictions,
    predict_task_boosters,
    run_training_lgbm,
    train_task_boosters,
)
from chemprop.train.metrics import prc_auc
from chemprop.utils import (
    LightGBMCheckpointError,
    LightGBMModelBundle,
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
            "--features_generator",
            "morgan",
            "--features_only",
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
    args.features_size = 2048
    args.lgbm_num_boost_round = 30
    args.lgbm_early_stopping_rounds = 5
    args.lgbm_num_threads = 1
    args.lgbm_min_data_in_leaf = 2
    return args


def _molecule_dataset(smiles, targets) -> MoleculeDataset:
    return MoleculeDataset(
        [
            MoleculeDatapoint(
                smiles=[smile],
                targets=list(row_targets),
                features_generator=["morgan"],
            )
            for smile, row_targets in zip(smiles, targets)
        ]
    )


def test_lgbm_external_splits_do_not_inherit_training_row_weights(
    tmp_path: Path, monkeypatch,
):
    args = _training_args(tmp_path, "regression", ["target"])
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    validation_path.write_text(
        "smiles,target\nCC,1\nCCC,2\n", encoding="utf-8"
    )
    test_path.write_text(
        "smiles,target\nCO,3\nCCO,4\n", encoding="utf-8"
    )
    weights_path = tmp_path / "training_weights.csv"
    weights_path.write_text(
        "weight\n1\n2\n3\n", encoding="utf-8"
    )
    args.task_names = ["target"]
    args.separate_val_path = str(validation_path)
    args.separate_test_path = str(test_path)
    args.data_weights_path = str(weights_path)
    main_data = MoleculeDataset([])
    monkeypatch.setattr(
        "chemprop.train.run_training_lgbm.validate_features_source_metadata",
        lambda *_args, **_kwargs: None,
    )

    train_data, validation_data, test_data = _load_split_data(
        args, main_data, logger=None,
    )

    assert train_data is main_data
    assert validation_data.data_weights() == [1.0, 1.0]
    assert test_data.data_weights() == [1.0, 1.0]


def test_lgbm_features_only_encoding_bypasses_graph_construction(
    tmp_path: Path, monkeypatch
):
    args = _training_args(tmp_path, "regression", ["target"])
    args.features_size = 3
    data = MoleculeDataset(
        [
            MoleculeDatapoint(
                smiles=["CC"], targets=[1.0], features=np.asarray([1, 2, 3])
            ),
            MoleculeDatapoint(
                smiles=["CCC"], targets=[2.0], features=np.asarray([4, 5, 6])
            ),
        ]
    )
    encoder = build_frozen_lgbm_encoder(args)

    def fail_if_loader_is_built(*_args, **_kwargs):
        raise AssertionError("features-only encoding must not build molecular graphs")

    monkeypatch.setattr(
        "chemprop.train.run_training_lgbm.MoleculeDataLoader",
        fail_if_loader_is_built,
    )

    encoded = encode_lgbm_features(encoder, data, batch_size=1, num_workers=0)

    np.testing.assert_array_equal(
        encoded, np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    )
    assert encoded.dtype == np.float32


def test_lgbm_prediction_rejects_random_mpn_bundle(monkeypatch):
    unsafe_bundle = LightGBMModelBundle(
        encoder=None,
        task_boosters=[],
        train_args=SimpleNamespace(features_only=False),
        scalers=(None, None, None, None, None),
        task_names=["target"],
        dataset_type="regression",
        model_index=0,
        seed=0,
        checkpoint_path="unsafe.pkl",
    )
    prediction_module = importlib.import_module("chemprop.train.make_predictions")
    monkeypatch.setattr(
        prediction_module,
        "load_checkpoint_lgbm",
        lambda _path, device=None: unsafe_bundle,
    )
    predict_args = SimpleNamespace(checkpoint_paths=["unsafe.pkl"], device=None)

    with pytest.raises(ValueError, match="untrained random MPN.*Retrain"):
        load_model_lgbm(predict_args)
    with pytest.raises(ValueError, match="untrained random MPN.*Retrain"):
        predict_lgbm(
            SimpleNamespace(batch_size=1, num_workers=0),
            unsafe_bundle,
            scaler=None,
            test_data=MoleculeDataset([]),
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

    raw_val_targets = np.asarray(val_data.targets(), dtype=float)
    features_scaler = train_data.normalize_features(replace_nan_token=0)
    val_data.normalize_features(features_scaler)
    test_data.normalize_features(features_scaler)
    scaler = train_data.normalize_targets()
    val_data.set_targets(scaler.transform(raw_val_targets).tolist())

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
        member_predictions.append(
            np.asarray(
                scaler.inverse_transform(
                    predict_task_boosters(boosters, test_features)
                ),
                dtype=float,
            )
        )
        checkpoint_path = tmp_path / f"model_{model_index}.pkl"
        saved_path = save_checkpoint_lgbm(
            str(checkpoint_path),
            encoder,
            boosters,
            scaler=scaler,
            features_scaler=features_scaler,
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
        np.asarray(
            reloaded.scalers[0].inverse_transform(
                predict_task_boosters(reloaded.task_boosters, reloaded_features)
            ),
            dtype=float,
        ),
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

    original_bundle_bytes = Path(checkpoint_paths[0]).read_bytes()
    corruptions = [
        (
            "boolean_version",
            lambda value: value.__setitem__("version", True),
            "Unsupported LightGBM bundle version",
        ),
        (
            "missing_target_scaler",
            lambda value: value["scalers"].__setitem__("data", None),
            "target data scaler.*missing",
        ),
        (
            "missing_feature_scaler",
            lambda value: value["scalers"].__setitem__("features", None),
            "molecular feature scaler.*missing",
        ),
        (
            "feature_scaler_width",
            lambda value: value["scalers"]["features"].__setitem__(
                "means", value["scalers"]["features"]["means"][:-1]
            ),
            "molecular feature scaler.*parameter widths",
        ),
        (
            "feature_scaler_zero_std",
            lambda value: value["scalers"]["features"]["stds"].__setitem__(0, 0),
            "molecular feature scaler.*positive",
        ),
        (
            "unexpected_atom_scaler",
            lambda value: value["scalers"].__setitem__(
                "atom_descriptor", {"means": np.zeros(1), "stds": np.ones(1)}
            ),
            "unexpected atom descriptor scaler",
        ),
        (
            "booster_count",
            lambda value: value.__setitem__("task_boosters", []),
            "one LightGBM Booster per task",
        ),
        (
            "booster_objective",
            lambda value: value["task_boosters"][0].params.__setitem__(
                "objective", "binary"
            ),
            "objective.*expected 'regression'",
        ),
        (
            "encoded_width",
            lambda value: setattr(
                value["args"], "features_size", value["args"].features_size + 1
            ),
            "Booster feature width.*expected encoded width",
        ),
        (
            "metadata_width",
            lambda value: value["metadata"].__setitem__(
                "encoded_feature_width", value["metadata"]["encoded_feature_width"] + 1
            ),
            "metadata encoded feature width",
        ),
        (
            "metadata_missing_seed",
            lambda value: value["metadata"].pop("seed"),
            "invalid or incomplete metadata",
        ),
        (
            "metadata_bad_index",
            lambda value: value["metadata"].__setitem__("model_index", True),
            "invalid model index or seed",
        ),
        (
            "encoder_state",
            lambda value: value.__setitem__(
                "encoder_state_dict", {"unexpected": np.asarray([1.0])}
            ),
            "invalid MPN state",
        ),
    ]
    for corruption_name, corrupt, message in corruptions:
        bundle = pickle.loads(original_bundle_bytes)
        corrupt(bundle)
        corruption_path = tmp_path / f"corrupted_{corruption_name}.pkl"
        with corruption_path.open("wb") as checkpoint_file:
            pickle.dump(bundle, checkpoint_file)
        with pytest.raises(LightGBMCheckpointError, match=message):
            load_checkpoint_lgbm(str(corruption_path))

    # ``encoded_feature_width`` was added as redundant metadata without
    # changing version 1, so already-created version-1 bundles remain usable;
    # their args, scaler widths, and Booster widths still cross-validate it.
    earlier_v1_bundle = pickle.loads(original_bundle_bytes)
    earlier_v1_bundle["metadata"].pop("encoded_feature_width")
    earlier_v1_path = tmp_path / "earlier_v1_bundle.pkl"
    with earlier_v1_path.open("wb") as checkpoint_file:
        pickle.dump(earlier_v1_bundle, checkpoint_file)
    assert load_checkpoint_lgbm(str(earlier_v1_path)).task_names == task_names

    with pytest.raises(ValueError, match="target data scaler.*missing"):
        save_checkpoint_lgbm(
            str(tmp_path / "missing_regression_scaler.pkl"),
            encoder,
            boosters,
            features_scaler=features_scaler,
            args=args,
        )

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
        "--features_generator",
        "morgan",
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
        encoding="utf-8",
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
            "--features_generator",
            "morgan",
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

    invalid_prediction_features = val_features.copy()
    invalid_prediction_features[0, 0] = np.inf
    with pytest.raises(ValueError, match="prediction features contain"):
        predict_task_boosters(first, invalid_prediction_features)
    with pytest.raises(ValueError, match="feature width does not match"):
        predict_task_boosters(first, val_features[:, :-1])

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


def test_lgbm_classification_bundle_scaler_contract(tmp_path: Path):
    args = _training_args(tmp_path, "classification", ["active"])
    args.features_size = 3
    args.features_generator = None
    args.features_path = [str(tmp_path / "external_features.npz")]
    args.lgbm_num_boost_round = 4
    args.lgbm_early_stopping_rounds = 0

    train_features = np.arange(36, dtype=float).reshape(12, 3)
    val_features = np.arange(18, dtype=float).reshape(6, 3)
    features_scaler = StandardScaler(replace_nan_token=0).fit(train_features)
    scaled_train_features = features_scaler.transform(train_features)
    scaled_val_features = features_scaler.transform(val_features)
    train_targets = (np.arange(12) % 2).reshape(-1, 1)
    val_targets = (np.arange(6) % 2).reshape(-1, 1)
    boosters = train_task_boosters(
        args,
        scaled_train_features,
        train_targets,
        scaled_val_features,
        val_targets,
        seed=3,
    )
    encoder = build_frozen_lgbm_encoder(args)
    checkpoint_path = tmp_path / "classification.pkl"
    save_checkpoint_lgbm(
        str(checkpoint_path),
        encoder,
        boosters,
        features_scaler=features_scaler,
        args=args,
    )
    loaded = load_checkpoint_lgbm(str(checkpoint_path))
    assert loaded.scalers[0] is None
    assert loaded.scalers[1] is not None

    invalid_target_scaler = StandardScaler().fit(train_targets)
    with pytest.raises(ValueError, match="target data scaler.*disable"):
        save_checkpoint_lgbm(
            str(tmp_path / "classification_with_target_scaler.pkl"),
            encoder,
            boosters,
            scaler=invalid_target_scaler,
            features_scaler=features_scaler,
            args=args,
        )
    with pytest.raises(ValueError, match="molecular feature scaler.*missing"):
        save_checkpoint_lgbm(
            str(tmp_path / "classification_without_feature_scaler.pkl"),
            encoder,
            boosters,
            args=args,
        )

    # Unscaled inputs are also a valid, distinct contract (for example with
    # rdkit_2d_normalized); in that case a feature scaler must be absent.
    args.features_scaling = False
    unscaled_boosters = train_task_boosters(
        args,
        train_features,
        train_targets,
        val_features,
        val_targets,
        seed=4,
    )
    unscaled_encoder = build_frozen_lgbm_encoder(args)
    unscaled_path = tmp_path / "classification_unscaled.pkl"
    save_checkpoint_lgbm(
        str(unscaled_path),
        unscaled_encoder,
        unscaled_boosters,
        args=args,
    )
    assert load_checkpoint_lgbm(str(unscaled_path)).scalers[1] is None
    with pytest.raises(ValueError, match="molecular feature scaler.*disable"):
        save_checkpoint_lgbm(
            str(tmp_path / "unscaled_with_feature_scaler.pkl"),
            unscaled_encoder,
            unscaled_boosters,
            features_scaler=features_scaler,
            args=args,
        )


def test_lgbm_training_writes_a_self_consistent_regression_bundle(tmp_path: Path):
    args = _training_args(tmp_path, "regression", ["target"])
    args.save_dir = str(tmp_path / "training")
    args.split_sizes = [0.7, 0.15, 0.15]
    args.lgbm_num_boost_round = 4
    args.lgbm_early_stopping_rounds = 0
    data = _molecule_dataset(
        SMILES,
        [[np.sin(index / 3)] for index in range(len(SMILES))],
    )

    validation_scores, test_scores = run_training_lgbm(
        args=args,
        data=data,
        fold_num=0,
    )

    assert set(validation_scores) == {"rmse"}
    assert set(test_scores) == {"rmse"}
    bundle = load_checkpoint_lgbm(
        str(Path(args.save_dir) / "model_0" / "model.pkl")
    )
    assert bundle.scalers[0] is not None
    assert bundle.scalers[0].means.shape == (1,)
    assert bundle.scalers[1] is not None
    assert bundle.scalers[1].means.shape == (args.features_size,)
    assert bundle.task_boosters[0].num_feature() == args.features_size


@pytest.mark.parametrize("metric", ["auc", "prc-auc"])
def test_lgbm_rank_metric_does_not_early_stop_on_single_class_validation(metric):
    rng = np.random.default_rng(12)
    train_features = rng.normal(size=(80, 8))
    val_features = rng.normal(size=(20, 8))
    train_targets = (np.arange(80) % 2).reshape(-1, 1)
    val_targets = np.zeros((20, 1))
    args = SimpleNamespace(
        dataset_type="classification",
        metric=metric,
        num_tasks=1,
        task_names=["active"],
        class_balance=False,
        quiet=True,
        num_workers=0,
        lgbm_num_boost_round=12,
        lgbm_early_stopping_rounds=3,
        lgbm_num_threads=1,
        lgbm_min_data_in_leaf=2,
    )

    booster = train_task_boosters(
        args,
        train_features,
        train_targets,
        val_features,
        val_targets,
        seed=5,
    )[0]

    assert booster.best_iteration == 0
    assert "validation" not in booster.best_score


def test_lgbm_undefined_primary_validation_metric_fails_fast(
    tmp_path: Path, monkeypatch,
):
    training_module = importlib.import_module("chemprop.train.run_training_lgbm")
    args = _training_args(tmp_path, "classification", ["active"])
    args.save_dir = str(tmp_path / "undefined_validation")
    args.metric = "auc"
    args.skip_test_evaluation = True
    args.lgbm_num_boost_round = 4
    args.lgbm_early_stopping_rounds = 0
    train_data = _molecule_dataset(
        SMILES[:16], [[index % 2] for index in range(16)],
    )
    validation_data = _molecule_dataset(
        SMILES[16:20], [[0] for _ in range(4)],
    )
    monkeypatch.setattr(
        training_module,
        "_load_split_data",
        lambda *_args, **_kwargs: (
            train_data,
            validation_data,
            MoleculeDataset([]),
        ),
    )

    with pytest.raises(ValueError, match="cannot select a trained checkpoint"):
        run_training_lgbm(
            args=args,
            data=MoleculeDataset([]),
            fold_num=0,
        )

    assert (
        Path(args.save_dir) / "model_0" / "model.pkl"
    ).is_file()


def test_lgbm_rejects_invalid_targets_features_and_data_weights():
    args = SimpleNamespace(
        dataset_type="classification",
        metric="binary_cross_entropy",
        num_tasks=1,
        task_names=["active"],
        class_balance=False,
        quiet=True,
        num_workers=0,
        lgbm_num_boost_round=5,
        lgbm_early_stopping_rounds=0,
        lgbm_num_threads=1,
        lgbm_min_data_in_leaf=1,
    )
    train_features = np.arange(16, dtype=float).reshape(8, 2)
    val_features = np.arange(8, dtype=float).reshape(4, 2)
    train_targets = (np.arange(8) % 2).reshape(-1, 1)
    val_targets = (np.arange(4) % 2).reshape(-1, 1)

    invalid_val_targets = val_targets.astype(float)
    invalid_val_targets[0, 0] = 2
    with pytest.raises(ValueError, match="validation labels other than 0 and 1"):
        train_task_boosters(
            args,
            train_features,
            train_targets,
            val_features,
            invalid_val_targets,
            seed=1,
        )

    invalid_features = train_features.copy()
    invalid_features[0, 0] = np.inf
    with pytest.raises(ValueError, match="encoded features contain"):
        train_task_boosters(
            args,
            invalid_features,
            train_targets,
            val_features,
            val_targets,
            seed=1,
        )

    with pytest.raises(ValueError, match="non-negative"):
        train_task_boosters(
            args,
            train_features,
            train_targets,
            val_features,
            val_targets,
            seed=1,
            train_weights=[1, 1, 1, 1, 1, 1, 1, -1],
        )

    regression_args = SimpleNamespace(
        **{**vars(args), "dataset_type": "regression", "metric": "rmse"}
    )
    infinite_targets = train_targets.astype(float)
    infinite_targets[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite value"):
        train_task_boosters(
            regression_args,
            train_features,
            infinite_targets,
            val_features,
            val_targets.astype(float),
            seed=1,
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

    corrupt_pickle_path = tmp_path / "corrupt.pkl"
    corrupt_pickle_path.write_bytes(b"\x80\xff")
    with pytest.raises(LightGBMCheckpointError, match="Could not read"):
        load_checkpoint_lgbm(str(corrupt_pickle_path))


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
