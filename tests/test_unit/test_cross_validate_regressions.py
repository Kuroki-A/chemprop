import importlib
import json
import stat
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from chemprop.args import TrainArgs
from chemprop.constants import TEST_SCORES_FILE_NAME
from chemprop.data import MoleculeDatapoint, MoleculeDataset


def test_dataset_cache_is_content_addressed_and_round_trips(tmp_path: Path, monkeypatch):
    cross_validate_module = importlib.import_module("chemprop.train.cross_validate")
    data_path = tmp_path / "data.csv"
    data_path.write_text("smiles,target\nCC,2\nCCC,3\n")
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("CHEMPROP_CACHE_DIR", str(cache_dir))
    args = TrainArgs().parse_args(
        [
            "--data_path",
            str(data_path),
            "--dataset_type",
            "regression",
            "--no_cuda",
        ]
    )

    manifest = cross_validate_module._dataset_cache_manifest(args)
    assert {
        "get_data",
        "MoleculeDataset",
        "make_mol",
        "featurization",
        "load_features",
        "feature_generators",
    } <= set(manifest["source_digests"])
    cache_path = cross_validate_module._dataset_cache_path(args, manifest)
    dataset = MoleculeDataset([MoleculeDatapoint(smiles=["CC"], targets=[2.0])])
    cross_validate_module._save_dataset_cache(cache_path, manifest, dataset)
    loaded = cross_validate_module._load_dataset_cache(cache_path, manifest)

    assert loaded.smiles() == [["CC"]]
    assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
    assert not list(cache_dir.glob("*.tmp"))
    mismatched_manifest = {**manifest, "schema_version": manifest["schema_version"] + 1}
    assert cross_validate_module._load_dataset_cache(cache_path, mismatched_manifest) is None

    data_path.write_text("smiles,target\nCC,2\nCCC,4\n")
    changed_manifest = cross_validate_module._dataset_cache_manifest(args)
    changed_path = cross_validate_module._dataset_cache_path(args, changed_manifest)

    assert changed_manifest != manifest
    assert changed_path != cache_path

    selected_features_path = tmp_path / "selected.csv"
    selected_features_path.write_text("rdkit_2d\nMolWt\n")
    args.selected_features_path = str(selected_features_path)
    selected_manifest = cross_validate_module._dataset_cache_manifest(args)
    selected_features_path.write_text("rdkit_2d\nMolLogP\n")
    changed_selected_manifest = cross_validate_module._dataset_cache_manifest(args)
    assert changed_selected_manifest != selected_manifest

    args.adding_h = True
    config_manifest = cross_validate_module._dataset_cache_manifest(args)
    assert config_manifest != changed_selected_manifest

    features_path = tmp_path / "features.npz"
    np.savez_compressed(features_path, features=np.ones((2, 3)))
    args.features_path = [str(features_path)]
    without_sidecar = cross_validate_module._dataset_cache_manifest(args)
    sidecar_path = Path(f"{features_path}.manifest.json")
    sidecar_path.write_text('{"schema_version": 1}')
    with_sidecar = cross_validate_module._dataset_cache_manifest(args)
    sidecar_path.write_text('{"schema_version": 2}')
    changed_sidecar = cross_validate_module._dataset_cache_manifest(args)
    assert without_sidecar != with_sidecar != changed_sidecar

    args.features_generator = ["mordred", "map4", "secfp"]
    generator_manifest = cross_validate_module._dataset_cache_manifest(args)
    generator_versions = generator_manifest["features_generator_metadata"]["versions"]
    assert {"mordredcommunity", "mhfp"} <= set(generator_versions)
    assert "map4" not in generator_versions


def test_dataset_cache_rejects_insecure_paths_before_unpickling(
    tmp_path: Path, monkeypatch,
):
    cross_validate_module = importlib.import_module("chemprop.train.cross_validate")
    dataset = MoleculeDataset([MoleculeDatapoint(smiles=["CC"], targets=[2.0])])
    manifest = {"schema_version": 1}

    insecure_dir = tmp_path / "insecure"
    insecure_dir.mkdir()
    insecure_dir.chmod(0o777)
    with pytest.raises(ValueError, match="private.*0700"):
        cross_validate_module._save_dataset_cache(
            str(insecure_dir / "cache.pt"), manifest, dataset,
        )

    private_dir = tmp_path / "private"
    private_dir.mkdir(mode=0o700)
    target = tmp_path / "attacker.pt"
    target.write_bytes(b"not a cache")
    cache_symlink = private_dir / "cache.pt"
    cache_symlink.symlink_to(target)
    load_called = False

    def fail_if_loaded(*args, **kwargs):
        nonlocal load_called
        load_called = True
        raise AssertionError("torch.load must not inspect an unsafe cache")

    monkeypatch.setattr(cross_validate_module.torch, "load", fail_if_loaded)
    with pytest.raises(ValueError, match="symbolic link"):
        cross_validate_module._load_dataset_cache(str(cache_symlink), manifest)
    assert load_called is False


def test_resume_manifest_tracks_separate_feature_sidecars(tmp_path: Path):
    cross_validate_module = importlib.import_module("chemprop.train.cross_validate")
    features_path = tmp_path / "separate_features.npz"
    np.savez_compressed(features_path, features=np.ones((2, 3)))
    args = TrainArgs().parse_args(
        [
            "--data_path",
            "tests/data/regression.csv",
            "--dataset_type",
            "regression",
            "--no_cuda",
        ]
    )
    args.separate_val_features_path = [str(features_path)]

    without_sidecar = cross_validate_module._resume_input_manifest(args)
    sidecar_path = Path(f"{features_path}.manifest.json")
    sidecar_path.write_text('{"schema_version": 1}')
    with_sidecar = cross_validate_module._resume_input_manifest(args)
    sidecar_path.write_text('{"schema_version": 2}')
    changed_sidecar = cross_validate_module._resume_input_manifest(args)

    assert without_sidecar != with_sidecar != changed_sidecar


def test_cross_validate_persists_validation_scores_and_resumes(tmp_path: Path, monkeypatch):
    cross_validate_module = importlib.import_module("chemprop.train.cross_validate")
    save_dir = tmp_path / "run"
    args = TrainArgs().parse_args(
        [
            "--data_path",
            "tests/data/regression.csv",
            "--dataset_type",
            "regression",
            "--save_dir",
            str(save_dir),
            "--extra_metrics",
            "mae",
            "--no_cuda",
            "--quiet",
        ]
    )
    data = MoleculeDataset(
        [
            MoleculeDatapoint(smiles=["CC"], targets=[2.0]),
            MoleculeDatapoint(smiles=["CCC"], targets=[3.0]),
        ]
    )
    monkeypatch.setattr(cross_validate_module, "get_task_names", lambda **_: ["target"])
    monkeypatch.setattr(cross_validate_module, "get_data", lambda **_: data)

    call_count = 0

    def fake_train(args, data, fold_num, logger):
        nonlocal call_count
        call_count += 1
        return {"rmse": [1.5], "mae": [1.25]}, {
            "rmse": [2.5],
            "mae": [2.25],
        }

    mean_score, std_score = cross_validate_module.cross_validate(args, fake_train)

    assert mean_score == pytest.approx(2.5)
    assert std_score == pytest.approx(0.0)
    assert call_count == 1
    assert (save_dir / "fold_0" / "valid_scores.json").is_file()
    assert (save_dir / "fold_0" / "test_scores.json").is_file()
    score_table = pd.read_csv(save_dir / TEST_SCORES_FILE_NAME)
    assert score_table.loc[0, "Mean rmse"] == 2.5
    assert score_table.loc[0, "Mean mae"] == 2.25

    args.save_dir = str(save_dir)
    args.resume_experiment = True
    resumed_mean, resumed_std = cross_validate_module.cross_validate(args, fake_train)

    assert resumed_mean == pytest.approx(2.5)
    assert resumed_std == pytest.approx(0.0)
    assert call_count == 1


def test_validation_only_cross_validate_never_persists_test_results(tmp_path: Path, monkeypatch):
    cross_validate_module = importlib.import_module("chemprop.train.cross_validate")
    save_dir = tmp_path / "validation_only"
    args = TrainArgs().parse_args(
        [
            "--data_path",
            "tests/data/regression.csv",
            "--dataset_type",
            "regression",
            "--save_dir",
            str(save_dir),
            "--data_type",
            "validation",
            "--skip_test_evaluation",
            "--save_preds",
            "--extra_metrics",
            "mae",
            "--no_cuda",
            "--quiet",
        ]
    )
    data = MoleculeDataset(
        [
            MoleculeDatapoint(smiles=["CC"], targets=[2.0]),
            MoleculeDatapoint(smiles=["CCC"], targets=[3.0]),
        ]
    )
    monkeypatch.setattr(cross_validate_module, "get_task_names", lambda **_: ["target"])
    monkeypatch.setattr(cross_validate_module, "get_data", lambda **_: data)

    call_count = 0

    def fake_train(args, data, fold_num, logger):
        nonlocal call_count
        call_count += 1
        # Cross-validation must discard held-out results even from a legacy
        # callback which has not implemented validation-only evaluation.
        return {"rmse": [1.5], "mae": [1.25]}, {
            "rmse": [999.0],
            "mae": [999.0],
        }

    mean_score, std_score = cross_validate_module.cross_validate(args, fake_train)

    assert mean_score == pytest.approx(1.5)
    assert std_score == pytest.approx(0.0)
    assert call_count == 1
    assert (save_dir / "fold_0" / "valid_scores.json").is_file()
    assert not (save_dir / "fold_0" / "test_scores.json").exists()
    assert not (save_dir / TEST_SCORES_FILE_NAME).exists()
    assert not (save_dir / "test_preds.csv").exists()

    args.save_dir = str(save_dir)
    args.resume_experiment = True
    resumed_mean, resumed_std = cross_validate_module.cross_validate(args, fake_train)

    assert resumed_mean == pytest.approx(1.5)
    assert resumed_std == pytest.approx(0.0)
    assert call_count == 1


def test_resume_retrains_for_corrupt_or_changed_artifacts(tmp_path: Path, monkeypatch):
    cross_validate_module = importlib.import_module("chemprop.train.cross_validate")
    data_path = tmp_path / "data.csv"
    data_path.write_text("smiles,target\nCC,2\nCCC,3\n")
    save_dir = tmp_path / "resume_contract"
    args = TrainArgs().parse_args(
        [
            "--data_path",
            str(data_path),
            "--dataset_type",
            "regression",
            "--save_dir",
            str(save_dir),
            "--no_cuda",
            "--quiet",
        ]
    )
    data = MoleculeDataset(
        [
            MoleculeDatapoint(smiles=["CC"], targets=[2.0]),
            MoleculeDatapoint(smiles=["CCC"], targets=[3.0]),
        ]
    )
    data._features_source_metadata = {"schema_version": 1, "sources": []}
    monkeypatch.setattr(cross_validate_module, "get_task_names", lambda **_: ["target"])
    monkeypatch.setattr(cross_validate_module, "get_data", lambda **_: data)

    call_count = 0

    def fake_train(args, data, fold_num, logger):
        nonlocal call_count
        call_count += 1
        return {"rmse": [1.0]}, {"rmse": [2.0]}

    def run(train_func=fake_train):
        args.save_dir = str(save_dir)
        return cross_validate_module.cross_validate(args, train_func)

    run()
    fold_dir = save_dir / "fold_0"
    manifest_path = fold_dir / "resume_manifest.json"
    record = json.loads(manifest_path.read_text())
    assert set(record["manifest"]) == {
        "schema_version",
        "fold_num",
        "dataset",
        "secondary_inputs",
        "config",
        "code",
    }
    assert record["manifest"]["config"]["features_source_metadata"] == {
        "schema_version": 1,
        "sources": [],
    }
    assert call_count == 1

    # A complete, identical transaction resumes without calling the trainer.
    args.resume_experiment = True
    run()
    assert call_count == 1

    # A syntactically corrupt score file cannot be trusted.
    (fold_dir / "valid_scores.json").write_text("{")
    run()
    assert call_count == 2

    # Training configuration changes invalidate otherwise valid scores.
    args.hidden_size += 1
    run()
    assert call_count == 3

    # The primary data file is content-fingerprinted, not identified only by path.
    data_path.write_text("smiles,target\nCC,2\nCCC,4\n")
    run()
    assert call_count == 4

    alternate_call_count = 0

    def alternate_train(args, data, fold_num, logger):
        nonlocal alternate_call_count
        alternate_call_count += 1
        return {"rmse": [3.0]}, {"rmse": [4.0]}

    # The callback identity/signature is part of the code contract.
    run(alternate_train)
    assert alternate_call_count == 1

    # A corrupt completion record also forces a clean retrain.
    manifest_path.write_text("not-json")
    run(alternate_train)
    assert alternate_call_count == 2
    assert not list(save_dir.rglob("*.tmp"))


@pytest.mark.parametrize(
    ("task_names", "ensemble_size"),
    [(["target_a", "target_b"], 1), (["target"], 2)],
)
def test_ffn_validation_scores_use_metrics_and_task_axis(
    tmp_path: Path, monkeypatch, task_names, ensemble_size
):
    """FFN validation scores match LightGBM/sklearn callback semantics."""
    cross_validate_module = importlib.import_module("chemprop.train.cross_validate")
    run_training_module = importlib.import_module("chemprop.train.run_training")
    data_path = tmp_path / "ffn_scores.csv"
    smiles = [
        "C",
        "CC",
        "CCC",
        "CCCC",
        "CCO",
        "CCN",
        "CO",
        "CN",
        "C=C",
        "CCCl",
        "CCBr",
        "c1ccccc1",
        "CC(=O)O",
        "CC(C)C",
        "C1CCCCC1",
        "COC",
        "CC#N",
        "CCS",
        "O=C=O",
        "N#N",
    ]
    rows = ["smiles," + ",".join(task_names)]
    datapoints = []
    for row_index, smile in enumerate(smiles):
        targets = [float(row_index + task_index) for task_index in range(len(task_names))]
        rows.append(smile + "," + ",".join(str(target) for target in targets))
        datapoints.append(MoleculeDatapoint(smiles=[smile], targets=targets))
    data_path.write_text("\n".join(rows) + "\n")
    data = MoleculeDataset(datapoints)

    save_dir = tmp_path / f"ensemble_{ensemble_size}"
    args = TrainArgs().parse_args(
        [
            "--data_path",
            str(data_path),
            "--dataset_type",
            "regression",
            "--save_dir",
            str(save_dir),
            "--target_columns",
            *task_names,
            "--extra_metrics",
            "mae",
            "--ensemble_size",
            str(ensemble_size),
            "--epochs",
            "0",
            "--hidden_size",
            "8",
            "--depth",
            "2",
            "--ffn_num_layers",
            "1",
            "--batch_size",
            "8",
            "--num_workers",
            "0",
            "--split_sizes",
            "0.6",
            "0.2",
            "0.2",
            "--no_cuda",
            "--quiet",
        ]
    )
    monkeypatch.setattr(
        cross_validate_module, "get_task_names", lambda **_: list(task_names)
    )
    monkeypatch.setattr(cross_validate_module, "get_data", lambda **_: data)

    cross_validate_module.cross_validate(args, run_training_module.run_training)

    fold_dir = save_dir / "fold_0"
    valid_scores = json.loads((fold_dir / "valid_scores.json").read_text())
    test_scores = json.loads((fold_dir / "test_scores.json").read_text())
    assert set(valid_scores) == {"rmse", "mae"}
    assert set(test_scores) == {"rmse", "mae"}
    assert all(len(scores) == len(task_names) for scores in valid_scores.values())
    assert all(len(scores) == len(task_names) for scores in test_scores.values())


def test_score_payload_rejects_missing_metrics_and_member_axis(tmp_path: Path):
    cross_validate_module = importlib.import_module("chemprop.train.cross_validate")
    data_path = tmp_path / "two_tasks.csv"
    data_path.write_text("smiles,a,b\nCC,1,2\n")
    args = TrainArgs().parse_args(
        [
            "--data_path",
            str(data_path),
            "--dataset_type",
            "regression",
            "--target_columns",
            "a",
            "b",
            "--extra_metrics",
            "mae",
            "--ensemble_size",
            "1",
            "--no_cuda",
        ]
    )
    args.task_names = ["a", "b"]

    with pytest.raises(ValueError, match="metric names"):
        cross_validate_module._validate_score_payload({"rmse": [1.0, 2.0]}, args)
    with pytest.raises(ValueError, match=r"expected \(2,\)"):
        cross_validate_module._validate_score_payload(
            {"rmse": [1.0], "mae": [2.0]}, args
        )
