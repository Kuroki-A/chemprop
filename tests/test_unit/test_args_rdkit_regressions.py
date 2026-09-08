from pathlib import Path

import pytest

from chemprop.args import HyperoptArgs, InterpretArgs, TrainArgs, get_checkpoint_paths
from chemprop.rdkit import make_mol


def test_checkpoint_directory_is_sorted(tmp_path: Path):
    (tmp_path / "z.pt").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.pt").touch()

    paths = get_checkpoint_paths(checkpoint_dir=str(tmp_path))

    assert paths == sorted(paths)


def test_interpret_args_without_model_type_reaches_checkpoint_validation():
    with pytest.raises(ValueError, match="Found no checkpoints"):
        InterpretArgs().parse_args(["--data_path", "tests/data/regression.csv"])


def test_hyperopt_always_selects_on_validation(tmp_path: Path):
    args = HyperoptArgs().parse_args(
        [
            "--data_path",
            "tests/data/regression.csv",
            "--dataset_type",
            "regression",
            "--config_save_path",
            str(tmp_path / "config.json"),
            "--save_dir",
            str(tmp_path / "results"),
            "--data_type",
            "test",
            "--no_cuda",
        ]
    )

    assert args.data_type == "validation"
    assert args.skip_test_evaluation is True


def test_unimplemented_aggregation_is_rejected():
    with pytest.raises(SystemExit):
        TrainArgs().parse_args(
            [
                "--data_path",
                "tests/data/regression.csv",
                "--dataset_type",
                "regression",
                "--aggregation",
                "max",
                "--no_cuda",
            ]
        )


def test_invalid_smiles_with_added_hydrogens_returns_none():
    assert make_mol("not_a_smiles", keep_h=False, add_h=True, keep_atom_map=False) is None


def test_lgbm_cli_hyperparameters_are_validated():
    with pytest.raises(ValueError, match="lgbm_num_boost_round"):
        TrainArgs().parse_args(
            [
                "--data_path",
                "tests/data/regression.csv",
                "--dataset_type",
                "regression",
                "--model_type",
                "lgbm",
                "--lgbm_num_boost_round",
                "0",
                "--no_cuda",
            ]
        )


def test_lgbm_rejects_silently_ignored_loss_function():
    with pytest.raises(ValueError, match="supports only --loss_function mse"):
        TrainArgs().parse_args(
            [
                "--data_path",
                "tests/data/regression.csv",
                "--dataset_type",
                "regression",
                "--model_type",
                "lgbm",
                "--loss_function",
                "mve",
                "--no_cuda",
            ]
        )


def test_lgbm_rejects_silently_ignored_target_weights():
    with pytest.raises(NotImplementedError, match="target_weights"):
        TrainArgs().parse_args(
            [
                "--data_path",
                "tests/data/regression.csv",
                "--dataset_type",
                "regression",
                "--model_type",
                "lgbm",
                "--target_weights",
                "2",
                "--no_cuda",
            ]
        )


@pytest.mark.parametrize(
    "dataset_type, metric",
    [
        ("classification", "accuracy"),
        ("classification", "f1"),
        ("classification", "mcc"),
        ("classification", "recall"),
        ("classification", "precision"),
        ("classification", "balanced_accuracy"),
        ("regression", "r2"),
    ],
)
def test_lgbm_rejects_unsupported_primary_early_stopping_metrics(
    dataset_type, metric
):
    with pytest.raises(NotImplementedError, match="early stopping"):
        TrainArgs().parse_args(
            [
                "--data_path",
                f"tests/data/{dataset_type}.csv",
                "--dataset_type",
                dataset_type,
                "--model_type",
                "lgbm",
                "--metric",
                metric,
                "--no_cuda",
            ]
        )


def test_lgbm_allows_non_primary_post_training_metrics():
    args = TrainArgs().parse_args(
        [
            "--data_path",
            "tests/data/classification.csv",
            "--dataset_type",
            "classification",
            "--model_type",
            "lgbm",
            "--metric",
            "prc-auc",
            "--extra_metrics",
            "accuracy",
            "f1",
            "--no_cuda",
        ]
    )

    assert args.metrics == ["prc-auc", "accuracy", "f1"]


@pytest.mark.parametrize("weights", [[-1, -2], [0, 0], [1, -1], [float("nan"), 1]])
def test_target_weights_validate_before_normalizing(weights):
    with pytest.raises(ValueError, match="target weight"):
        TrainArgs().parse_args(
            [
                "--data_path",
                "tests/data/regression.csv",
                "--dataset_type",
                "regression",
                "--target_weights",
                *map(str, weights),
                "--no_cuda",
            ]
        )


def test_quantile_loss_metric_is_minimized():
    args = TrainArgs().parse_args(
        [
            "--data_path",
            "tests/data/regression.csv",
            "--dataset_type",
            "regression",
            "--loss_function",
            "quantile_interval",
            "--no_cuda",
        ]
    )

    assert args.metric == "quantile"
    assert args.minimize_score is True


def test_ensemble_size_must_be_positive():
    with pytest.raises(ValueError, match="ensemble_size"):
        TrainArgs().parse_args(
            [
                "--data_path",
                "tests/data/regression.csv",
                "--dataset_type",
                "regression",
                "--ensemble_size",
                "0",
                "--no_cuda",
            ]
        )


def test_hyperopt_rejects_unsupported_lightgbm_backend(tmp_path):
    data_path = tmp_path / "data.csv"
    data_path.write_text("smiles,target\nCC,1\nCCC,2\n")
    with pytest.raises(NotImplementedError, match="supports only --model_type FFN"):
        HyperoptArgs().parse_args(
            [
                "--data_path", str(data_path),
                "--dataset_type", "regression",
                "--model_type", "lgbm",
                "--config_save_path", str(tmp_path / "best.json"),
                "--no_cuda",
            ]
        )
