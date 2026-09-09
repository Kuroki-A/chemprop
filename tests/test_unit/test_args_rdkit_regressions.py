from pathlib import Path

import pytest

from chemprop.args import (
    HyperoptArgs,
    InterpretArgs,
    PredictArgs,
    SklearnTrainArgs,
    TrainArgs,
    get_checkpoint_paths,
)
from chemprop.rdkit import make_mol


def test_checkpoint_directory_is_sorted(tmp_path: Path):
    (tmp_path / "z.pt").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.pt").touch()

    paths = get_checkpoint_paths(checkpoint_dir=str(tmp_path))

    assert paths == sorted(paths)


def test_checkpoint_paths_reject_wrong_extension():
    with pytest.raises(ValueError, match=r'extension "\.pt".*model\.pkl'):
        get_checkpoint_paths(checkpoint_paths=['model.pt', 'model.pkl'])


def test_checkpoint_extension_matching_is_case_insensitive(tmp_path: Path):
    checkpoint = tmp_path / 'MODEL.PT'
    checkpoint.touch()

    assert get_checkpoint_paths(checkpoint_path=str(checkpoint)) == [str(checkpoint)]
    assert get_checkpoint_paths(checkpoint_dir=str(tmp_path)) == [str(checkpoint)]


def test_predict_args_reject_mixed_checkpoint_directory(tmp_path: Path):
    (tmp_path / 'ffn.pt').touch()
    (tmp_path / 'lgbm.pkl').touch()

    with pytest.raises(ValueError, match=r'mix FFN \.pt.*LightGBM \.pkl'):
        PredictArgs().parse_args(
            [
                '--test_path',
                'tests/data/regression.csv',
                '--preds_path',
                str(tmp_path / 'preds.csv'),
                '--checkpoint_dir',
                str(tmp_path),
                '--no_cuda',
            ]
        )


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


def test_train_args_extra_metrics_are_not_shared_between_instances():
    first = TrainArgs()
    second = TrainArgs()

    first.extra_metrics.append("mae")

    assert first.extra_metrics == ["mae"]
    assert second.extra_metrics == []


def test_hyperopt_search_keywords_are_not_shared_between_instances():
    first = HyperoptArgs()
    second = HyperoptArgs()

    first.search_parameter_keywords.append("dropout")

    assert first.search_parameter_keywords == ["basic", "dropout"]
    assert second.search_parameter_keywords == ["basic"]


def test_implicit_training_save_directories_are_owned_by_each_instance():
    common = [
        "--data_path",
        "tests/data/regression.csv",
        "--dataset_type",
        "regression",
        "--no_cuda",
    ]
    first = TrainArgs().parse_args(common)
    first_directory = Path(first.save_dir)
    second = TrainArgs().parse_args(common)

    assert first._temp_save_dir is not second._temp_save_dir
    assert first_directory.is_dir()
    assert Path(second.save_dir).is_dir()


@pytest.mark.parametrize(
    "option,value,message",
    [
        ("--batch_size", "0", "batch_size"),
        ("--num_workers", "-1", "num_workers"),
        ("--max_data_size", "0", "max_data_size"),
        ("--number_of_molecules", "0", "number_of_molecules"),
        ("--epochs", "-1", "epochs"),
    ],
)
def test_invalid_common_training_sizes_are_rejected(option, value, message):
    with pytest.raises(ValueError, match=message):
        TrainArgs().parse_args(
            [
                "--data_path",
                "tests/data/regression.csv",
                "--dataset_type",
                "regression",
                option,
                value,
                "--no_cuda",
            ]
        )


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
                "--features_generator",
                "morgan",
                "--features_only",
                "--lgbm_num_boost_round",
                "0",
                "--no_cuda",
            ]
        )


@pytest.mark.parametrize(
    "option,value",
    [
        ("--lgbm_learning_rate", "nan"),
        ("--lgbm_learning_rate", "inf"),
        ("--lgbm_feature_fraction", "nan"),
        ("--lgbm_feature_fraction", "inf"),
        ("--lgbm_bagging_fraction", "nan"),
        ("--lgbm_bagging_fraction", "inf"),
    ],
)
def test_lgbm_cli_float_hyperparameters_must_be_finite(option, value):
    with pytest.raises(ValueError, match=option.removeprefix("--")):
        TrainArgs().parse_args(
            [
                "--data_path",
                "tests/data/regression.csv",
                "--dataset_type",
                "regression",
                "--model_type",
                "lgbm",
                "--features_generator",
                "morgan",
                "--features_only",
                option,
                value,
                "--no_cuda",
            ]
        )


def _predict_cli_args(tmp_path):
    return [
        "--test_path",
        "tests/data/regression.csv",
        "--preds_path",
        str(tmp_path / "predictions.csv"),
        "--checkpoint_path",
        str(tmp_path / "model.pt"),
        "--no_cuda",
    ]


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--calibration_method", "zscaling"],
        ["--calibration_path", "calibration.csv"],
    ],
)
def test_prediction_requires_calibration_method_and_path_together(
    tmp_path, extra_args,
):
    with pytest.raises(ValueError, match="must be provided together"):
        PredictArgs().parse_args(_predict_cli_args(tmp_path) + extra_args)


def test_prediction_rejects_silently_unused_auxiliary_output_args(tmp_path):
    with pytest.raises(ValueError, match="require --calibration_path"):
        PredictArgs().parse_args(
            _predict_cli_args(tmp_path)
            + ["--calibration_features_path", "features.npz"]
        )
    with pytest.raises(ValueError, match="requires --evaluation_methods"):
        PredictArgs().parse_args(
            _predict_cli_args(tmp_path)
            + ["--evaluation_scores_path", str(tmp_path / "scores.csv")]
        )


@pytest.mark.parametrize(
    'constraint_args',
    [
        ['--constraints_path', 'prediction_constraints.csv'],
        ['--calibration_constraints_path', 'calibration_constraints.csv'],
    ],
)
def test_prediction_calibration_requires_both_constraint_files(
    tmp_path, constraint_args,
):
    with pytest.raises(ValueError, match='constraints_path.*either both'):
        PredictArgs().parse_args(
            _predict_cli_args(tmp_path)
            + [
                '--calibration_method',
                'zscaling',
                '--calibration_path',
                'calibration.csv',
            ]
            + constraint_args
        )


def test_prediction_calibration_accepts_matching_constraint_files(tmp_path):
    args = PredictArgs().parse_args(
        _predict_cli_args(tmp_path)
        + [
            '--calibration_method',
            'zscaling',
            '--calibration_path',
            'calibration.csv',
            '--constraints_path',
            'prediction_constraints.csv',
            '--calibration_constraints_path',
            'calibration_constraints.csv',
        ]
    )

    assert args.constraints_path == 'prediction_constraints.csv'
    assert args.calibration_constraints_path == 'calibration_constraints.csv'


@pytest.mark.parametrize(
    "option,value,message",
    [
        ("--uncertainty_dropout_p", "0", "dropout probability"),
        ("--uncertainty_dropout_p", "1", "dropout probability"),
        ("--uncertainty_dropout_p", "nan", "dropout probability"),
        ("--calibration_interval_percentile", "nan", "calibration interval"),
        ("--conformal_alpha", "0", "conformal_alpha"),
        ("--conformal_alpha", "1", "conformal_alpha"),
        ("--conformal_alpha", "nan", "conformal_alpha"),
    ],
)
def test_prediction_uncertainty_floats_must_be_finite_and_open_interval(
    tmp_path, option, value, message,
):
    with pytest.raises(ValueError, match=message):
        PredictArgs().parse_args(
            _predict_cli_args(tmp_path) + [option, value]
        )


def test_dropout_rejects_individual_checkpoint_output_flag(tmp_path):
    with pytest.raises(ValueError, match="not supported.*dropout"):
        PredictArgs().parse_args(
            _predict_cli_args(tmp_path)
            + [
                "--uncertainty_method",
                "dropout",
                "--individual_ensemble_predictions",
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
                "--features_generator",
                "morgan",
                "--features_only",
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
                "--features_generator",
                "morgan",
                "--features_only",
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
                "--features_generator",
                "morgan",
                "--features_only",
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
            "--features_generator",
            "morgan",
            "--features_only",
            "--metric",
            "prc-auc",
            "--extra_metrics",
            "accuracy",
            "f1",
            "--no_cuda",
        ]
    )

    assert args.metrics == ["prc-auc", "accuracy", "f1"]


def test_lgbm_requires_deterministic_features_only_representation():
    with pytest.raises(ValueError, match="requires --features_only") as error:
        TrainArgs().parse_args(
            [
                "--data_path",
                "tests/data/regression.csv",
                "--dataset_type",
                "regression",
                "--model_type",
                "lgbm",
                "--no_cuda",
            ]
        )

    assert "--features_generator morgan --features_only" in str(error.value)


def test_sklearn_rejects_target_weights_instead_of_silently_ignoring_them():
    with pytest.raises(NotImplementedError, match="target_weights"):
        SklearnTrainArgs().parse_args(
            [
                "--data_path",
                "tests/data/regression.csv",
                "--dataset_type",
                "regression",
                "--model_type",
                "random_forest",
                "--target_weights",
                "2",
                "--no_cuda",
            ]
        )


def test_sklearn_prediction_requires_a_checkpoint(tmp_path):
    from chemprop.args import SklearnPredictArgs

    with pytest.raises(ValueError, match="no sklearn checkpoints"):
        SklearnPredictArgs().parse_args(
            [
                "--test_path",
                "tests/data/regression.csv",
                "--preds_path",
                str(tmp_path / "preds.csv"),
            ]
        )


@pytest.mark.parametrize(
    "dataset_type, impute_mode",
    [("classification", "mean"), ("regression", "frequent")],
)
def test_sklearn_rejects_incompatible_imputation_modes(
    dataset_type, impute_mode
):
    with pytest.raises(ValueError, match="impute_mode"):
        SklearnTrainArgs().parse_args(
            [
                "--data_path",
                f"tests/data/{dataset_type}.csv",
                "--dataset_type",
                dataset_type,
                "--model_type",
                "random_forest",
                "--impute_mode",
                impute_mode,
                "--no_cuda",
            ]
        )


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


@pytest.mark.parametrize("floor", ["0", "-1", "nan", "inf"])
def test_spectra_target_floor_must_be_finite_and_positive(floor):
    with pytest.raises(ValueError, match="spectra_target_floor"):
        TrainArgs().parse_args(
            [
                "--data_path",
                "tests/data/regression.csv",
                "--dataset_type",
                "spectra",
                "--spectra_target_floor",
                floor,
                "--no_cuda",
            ]
        )


def test_spectra_phase_mask_requires_phase_features():
    with pytest.raises(ValueError, match="requires phase_features_path"):
        TrainArgs().parse_args(
            [
                "--data_path",
                "tests/data/regression.csv",
                "--dataset_type",
                "spectra",
                "--spectra_phase_mask_path",
                "phase_mask.npz",
                "--no_cuda",
            ]
        )


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
                "--features_generator", "morgan",
                "--features_only",
                "--config_save_path", str(tmp_path / "best.json"),
                "--no_cuda",
            ]
        )
