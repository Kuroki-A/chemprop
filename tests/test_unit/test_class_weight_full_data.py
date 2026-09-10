import importlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chemprop.args import HyperoptArgs, PredictArgs, SklearnTrainArgs, TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
from chemprop.data.utils import _target_contains_observed_value, get_data
from chemprop.models import MoleculeModel
from chemprop.train.cross_validate import _training_config_manifest, cross_validate
from chemprop.train.make_predictions import make_predictions
from chemprop.train.run_training import (
    _full_data_metric_mean,
    _resolve_balanced_class_weights,
    _validate_full_data_runtime_args,
    _validate_training_split,
)
from chemprop.train.train import train, _validate_balanced_class_weight_training
from chemprop.utils import load_args, load_checkpoint, load_scalers, save_checkpoint


SMILES = ["C", "CC", "CCC", "CCCC", "CCO", "CCN", "CO", "CN", "C=C", "C#N"]


def _write_dataset(path: Path, targets) -> None:
    rows = ["smiles,target"]
    rows.extend(
        f"{smiles},{'' if target is None else target}"
        for smiles, target in zip(SMILES, targets)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_smiles(path: Path, smiles=SMILES) -> None:
    path.write_text(
        "smiles\n" + "\n".join(smiles) + "\n", encoding="utf-8"
    )


def _dataset(targets, features=None) -> MoleculeDataset:
    if features is None:
        features = [None] * len(targets)
    return MoleculeDataset(
        [
            MoleculeDatapoint(
                smiles=[SMILES[index % len(SMILES)]],
                targets=list(target) if isinstance(target, (list, tuple)) else [target],
                features=None if feature is None else np.asarray(feature, dtype=float),
            )
            for index, (target, feature) in enumerate(zip(targets, features))
        ]
    )


def _class_weight_args(num_tasks=1):
    return SimpleNamespace(
        class_weight="balanced",
        class_balance=False,
        data_weights_path=None,
        dataset_type="classification",
        is_atom_bond_targets=False,
        loss_function="binary_cross_entropy",
        model_type="FFN",
        num_tasks=num_tasks,
    )


def test_balanced_class_weight_formula_uses_observed_training_rows_only():
    train_data = _dataset([0] * 8 + [1] * 2 + [None])
    args = _class_weight_args()

    metadata = _resolve_balanced_class_weights(args, train_data)

    assert args.resolved_class_counts == [8, 2]
    assert args.resolved_class_weights == pytest.approx([0.625, 2.5])
    assert args.class_weight_observed_count == 10
    assert metadata["source"] == "post-split training data only"
    row_weights = [args.resolved_class_weights[int(target)] for target in [0] * 8 + [1] * 2]
    assert np.mean(row_weights) == pytest.approx(1.0)

    # Validation/test ratios are deliberately not inputs to the resolver and
    # therefore cannot affect the training-only result.
    _dataset([1] * 10)
    _dataset([0] * 10)
    repeated = _resolve_balanced_class_weights(args, train_data)
    assert repeated == metadata


@pytest.mark.parametrize(
    "args,targets,message",
    [
        (_class_weight_args(num_tasks=2), [[0, 1], [1, 0]], "single-task"),
        (_class_weight_args(), [0, 0, None], "both binary classes"),
        (_class_weight_args(), [0, 1, 2], "binary values 0 or 1"),
    ],
)
def test_balanced_class_weight_rejects_unsupported_training_targets(
    args, targets, message,
):
    with pytest.raises(ValueError, match=message):
        _resolve_balanced_class_weights(args, _dataset(targets))


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"model_type": "lgbm"}, "FFN backend"),
        ({"loss_function": "mcc"}, "binary_cross_entropy"),
        ({"class_balance": True}, "class_balance"),
        ({"data_weights_path": "weights.csv"}, "data_weights_path"),
    ],
)
def test_balanced_class_weight_programmatic_api_enforces_public_contract(
    updates, message,
):
    args = _class_weight_args()
    for name, value in updates.items():
        setattr(args, name, value)

    with pytest.raises(ValueError, match=message):
        _resolve_balanced_class_weights(args, _dataset([0, 1]))


def test_resolved_class_weight_fields_do_not_change_cv_resume_manifest(
    tmp_path: Path,
):
    data_path = tmp_path / "classification.csv"
    _write_dataset(data_path, [0] * 8 + [1] * 2)
    args = TrainArgs().parse_args(
        [
            "--data_path", str(data_path),
            "--dataset_type", "classification",
            "--class_weight", "balanced",
            "--no_cuda",
        ]
    )
    args.task_names = ["target"]
    before = _training_config_manifest(args)

    _resolve_balanced_class_weights(args, _dataset([0] * 8 + [1] * 2))

    assert _training_config_manifest(args) == before


def test_nested_atom_bond_targets_without_observations_are_missing():
    assert not _target_contains_observed_value(np.asarray([None, None]))
    assert not _target_contains_observed_value([np.asarray([None]), None])
    assert _target_contains_observed_value(np.asarray([None, 1.5]))


def test_full_data_metric_mean_respects_ignore_nan_metrics():
    scores = [0.8, np.nan]
    assert _full_data_metric_mean(scores, "auc", False) is None
    assert _full_data_metric_mean(scores, "auc", True) == pytest.approx(0.8)


def test_class_weight_loader_keeps_every_training_row():
    data = _dataset([0] * 8 + [1] * 2 + [None])
    loader = MoleculeDataLoader(
        data, batch_size=4, num_workers=0, class_balance=False, shuffle=False,
    )

    assert loader.iter_size == len(data)
    assert list(loader._sampler) == list(range(len(data)))


class _Batch:
    def __init__(self, targets, data_weights=None):
        self._targets = [[float(target)] for target in targets]
        self._masks = [tuple(True for _ in targets)]
        self._data_weights = list(data_weights or [1.0] * len(targets))
        self.number_of_atoms = [1] * len(targets)
        self.number_of_bonds = [0] * len(targets)

    def __len__(self):
        return len(self._targets)

    def batch_graph(self):
        return None

    def features(self):
        return None

    def targets(self):
        return self._targets

    def mask(self):
        return self._masks

    def atom_descriptors(self):
        return None

    def atom_features(self):
        return None

    def bond_descriptors(self):
        return None

    def bond_features(self):
        return None

    def constraints(self):
        return None

    def data_weights(self):
        return self._data_weights


class _LogitModel(torch.nn.Module):
    is_atom_bond_targets = False

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.as_tensor(logits, dtype=torch.float))

    def forward(self, *args):
        return self.logits.reshape(-1, 1)


def _loss_args(**overrides):
    values = dict(
        device=torch.device("cpu"),
        dataset_type="classification",
        loss_function="binary_cross_entropy",
        target_weights=None,
        class_weight=None,
        atom_targets=[],
        bond_targets=[],
        atom_constraints=[],
        bond_constraints=[],
        adding_bond_types=False,
        grad_clip=None,
        batch_size=10,
        log_frequency=100,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_train_applies_expected_balanced_weight_to_each_bce_loss():
    targets = torch.tensor([0.0] * 8 + [1.0] * 2)
    logits = torch.linspace(-1.0, 1.0, len(targets))
    expected_logits = logits.clone().requires_grad_(True)
    row_weights = torch.tensor([0.625] * 8 + [2.5] * 2)
    expected_loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            expected_logits, targets, reduction="none"
        )
        * row_weights
    ).mean()
    expected_loss.backward()

    model = _LogitModel(logits)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    train(
        model=model,
        data_loader=[_Batch(targets.tolist())],
        loss_func=torch.nn.BCEWithLogitsLoss(reduction="none"),
        optimizer=optimizer,
        scheduler=object(),
        args=_loss_args(
            class_weight="balanced", resolved_class_weights=[0.625, 2.5]
        ),
    )

    torch.testing.assert_close(model.logits.grad, expected_logits.grad)


def test_data_weights_are_multiplied_directly_inside_train_without_renormalizing():
    targets = torch.tensor([0.0, 1.0])
    logits = torch.tensor([-0.4, 0.8])
    supplied_weights = torch.tensor([2.0, 0.5])
    expected_logits = logits.clone().requires_grad_(True)
    expected_loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            expected_logits, targets, reduction="none"
        )
        * supplied_weights
    ).mean()
    expected_loss.backward()

    model = _LogitModel(logits)
    train(
        model=model,
        data_loader=[_Batch(targets.tolist(), supplied_weights.tolist())],
        loss_func=torch.nn.BCEWithLogitsLoss(reduction="none"),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        scheduler=object(),
        args=_loss_args(batch_size=2),
    )

    torch.testing.assert_close(model.logits.grad, expected_logits.grad)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"class_weight": "unsupported"}, "Unsupported FFN class weight mode"),
        (
            {"class_weight": "balanced", "model_type": "lgbm"},
            "supports only the FFN backend",
        ),
        (
            {"class_weight": "balanced", "class_balance": True},
            "cannot be combined with --class_balance",
        ),
        (
            {"class_weight": "balanced", "data_weights_path": "weights.csv"},
            "cannot be combined with --data_weights_path",
        ),
        (
            {"class_weight": "balanced", "is_atom_bond_targets": True},
            "molecule-level targets only",
        ),
        (
            {"class_weight": "balanced", "num_tasks": 2},
            "single-task binary classification",
        ),
    ],
)
def test_low_level_train_class_weight_contract(overrides, message):
    args = _loss_args(**overrides)
    with pytest.raises(ValueError, match=message):
        _validate_balanced_class_weight_training(_LogitModel([0.0]), args)


def test_low_level_train_rejects_nonbinary_observed_target():
    model = _LogitModel([0.0, 0.0])
    with pytest.raises(ValueError, match="binary values 0 or 1"):
        train(
            model=model,
            data_loader=[_Batch([0.0, 2.0])],
            loss_func=torch.nn.BCEWithLogitsLoss(reduction="none"),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            scheduler=object(),
            args=_loss_args(
                class_weight="balanced",
                resolved_class_weights=[1.0, 1.0],
            ),
        )


@pytest.mark.parametrize(
    "extra_args,message",
    [
        (["--class_balance"], "cannot be combined with --class_balance"),
        (["--data_weights_path", "weights.csv"], "cannot be combined with --data_weights_path"),
    ],
)
def test_ffn_class_weight_rejects_other_row_balancing_modes(
    tmp_path: Path, extra_args, message,
):
    data_path = tmp_path / "classification.csv"
    _write_dataset(data_path, [0] * 8 + [1] * 2)
    with pytest.raises(ValueError, match=message):
        TrainArgs().parse_args(
            [
                "--data_path", str(data_path),
                "--dataset_type", "classification",
                "--class_weight", "balanced",
                "--no_cuda",
                *extra_args,
            ]
        )


@pytest.mark.parametrize(
    "dataset_type,extra_args,message",
    [
        ("regression", [], "only for classification"),
        ("multiclass", [], "only for classification"),
        ("classification", ["--loss_function", "mcc"], "requires --loss_function binary_cross_entropy"),
        ("classification", ["--is_atom_bond_targets"], "molecule-level"),
    ],
)
def test_ffn_class_weight_rejects_unsupported_problem_types(
    tmp_path: Path, dataset_type, extra_args, message,
):
    data_path = tmp_path / "data.csv"
    _write_dataset(data_path, [0] * 8 + [1] * 2)
    with pytest.raises(ValueError, match=message):
        TrainArgs().parse_args(
            [
                "--data_path", str(data_path),
                "--dataset_type", dataset_type,
                "--class_weight", "balanced",
                "--no_cuda",
                *extra_args,
            ]
        )


def test_class_weight_meaning_remains_explicit_for_sklearn_and_lightgbm(
    tmp_path: Path,
):
    data_path = tmp_path / "classification.csv"
    _write_dataset(data_path, [0] * 8 + [1] * 2)

    sklearn_args = SklearnTrainArgs().parse_args(
        [
            "--data_path", str(data_path),
            "--dataset_type", "classification",
            "--model_type", "random_forest",
            "--class_weight", "balanced",
            "--no_cuda",
        ]
    )
    assert sklearn_args.class_weight == "balanced"

    with pytest.raises(ValueError, match="FFN-only"):
        TrainArgs().parse_args(
            [
                "--data_path", str(data_path),
                "--dataset_type", "classification",
                "--model_type", "lgbm",
                "--class_weight", "balanced",
                "--no_cuda",
            ]
        )


@pytest.mark.parametrize(
    "extra_args,message",
    [
        (["--epochs", "0"], "epochs.*greater than 0"),
        (["--split_sizes", "0.8", "0.1", "0.1"], "cannot be combined with --split_sizes"),
        (["--split_type", "scaffold_balanced"], "non-default --split_type"),
        (["--num_folds", "2"], "cross-validation"),
        (["--separate_val_path", "validation.csv"], "separate_val_path"),
        (["--max_data_size", "5"], "max_data_size"),
        (["--resume_experiment"], "resume_experiment"),
        (["--split_key_molecule", "1"], "split_key_molecule"),
        (["--data_type", "validation"], "data_type validation"),
        (["--folds_file", "folds.pkl"], "split arguments"),
        (["--crossval_index_file", "index.pkl"], "split arguments"),
    ],
)
def test_full_data_argument_conflicts_are_rejected(
    tmp_path: Path, extra_args, message,
):
    data_path = tmp_path / "regression.csv"
    _write_dataset(data_path, list(range(10)))
    with pytest.raises(ValueError, match=message):
        TrainArgs().parse_args(
            [
                "--data_path", str(data_path),
                "--dataset_type", "regression",
                "--train_on_full_data",
                "--no_cuda",
                *extra_args,
            ]
        )


def test_full_data_rejects_test_only_lightgbm_and_hyperopt(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    _write_dataset(data_path, list(range(10)))

    with pytest.raises(ValueError, match="cannot be combined with --test"):
        TrainArgs().parse_args(
            [
                "--data_path", str(data_path),
                "--dataset_type", "regression",
                "--train_on_full_data",
                "--test",
                "--checkpoint_path", str(tmp_path / "model.pt"),
                "--no_cuda",
            ]
        )


def test_full_data_warns_that_early_stopping_is_disabled(tmp_path: Path):
    data_path = tmp_path / "regression.csv"
    _write_dataset(data_path, list(range(10)))

    with pytest.warns(
        UserWarning,
        match="validation-based checkpoint selection and early stopping",
    ):
        args = TrainArgs().parse_args(
            [
                "--data_path", str(data_path),
                "--dataset_type", "regression",
                "--train_on_full_data",
                "--early_stopping", "1",
                "--epochs", "2",
                "--no_cuda",
            ]
        )

    assert args.early_stopping == 1
    assert args.train_on_full_data is True


def test_full_data_runtime_guard_rejects_programmatic_split_override(
    tmp_path: Path,
):
    data_path = tmp_path / "regression.csv"
    _write_dataset(data_path, list(range(10)))
    args = _full_data_args(data_path, tmp_path / "models")
    _validate_full_data_runtime_args(args)

    args.split_sizes = [0.8, 0.1, 0.1]
    with pytest.raises(ValueError, match="no-split marker"):
        _validate_full_data_runtime_args(args)
    with pytest.raises(ValueError, match="only --model_type FFN"):
        TrainArgs().parse_args(
            [
                "--data_path", str(data_path),
                "--dataset_type", "regression",
                "--train_on_full_data",
                "--model_type", "lgbm",
                "--features_generator", "morgan",
                "--features_only",
                "--no_cuda",
            ]
        )
    with pytest.raises(ValueError, match="chemprop_hyperopt"):
        HyperoptArgs().parse_args(
            [
                "--data_path", str(data_path),
                "--dataset_type", "regression",
                "--train_on_full_data",
                "--config_save_path", str(tmp_path / "config.json"),
                "--no_cuda",
            ]
        )


def _full_data_args(data_path: Path, save_dir: Path, *extra_args) -> TrainArgs:
    args = TrainArgs().parse_args(
        [
            "--data_path", str(data_path),
            "--dataset_type", "regression",
            "--train_on_full_data",
            "--epochs", "2",
            "--ensemble_size", "2",
            "--save_dir", str(save_dir),
            "--hidden_size", "8",
            "--ffn_hidden_size", "8",
            "--depth", "1",
            "--ffn_num_layers", "1",
            "--batch_size", "4",
            "--num_workers", "0",
            "--no_cuda",
            "--quiet",
            *extra_args,
        ]
    )
    args.task_names = ["target"]
    return args


def test_full_data_runs_every_epoch_skips_validation_and_saves_final_members(
    tmp_path: Path, monkeypatch,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    data_path = tmp_path / "regression.csv"
    _write_dataset(data_path, [1, 2, 3, 4, 5, 6, None, None, None, None])
    save_dir = tmp_path / "models"
    args = _full_data_args(data_path, save_dir)
    data = get_data(path=str(data_path), args=args, skip_none_targets=True)
    args.features_size = data.features_size()

    monkeypatch.setattr(
        run_training_module,
        "split_data",
        lambda **_: (_ for _ in ()).throw(AssertionError("split_data called")),
    )
    monkeypatch.setattr(
        run_training_module,
        "evaluate",
        lambda **_: (_ for _ in ()).throw(AssertionError("validation called")),
    )

    epoch_counts = {}
    observed_train_sizes = []

    def deterministic_epoch(model, data_loader, n_iter=0, **kwargs):
        observed_train_sizes.append(data_loader.iter_size)
        model_key = id(model)
        epoch_counts[model_key] = epoch_counts.get(model_key, 0) + 1
        with torch.no_grad():
            next(model.parameters()).fill_(float(epoch_counts[model_key]))
        return n_iter + data_loader.iter_size

    scheduler_steps = []
    original_scheduler_builder = run_training_module.build_lr_scheduler

    def capture_scheduler(optimizer, scheduler_args, **kwargs):
        scheduler_steps.append(kwargs.get("steps_per_epoch"))
        return original_scheduler_builder(optimizer, scheduler_args, **kwargs)

    monkeypatch.setattr(run_training_module, "train", deterministic_epoch)
    monkeypatch.setattr(run_training_module, "build_lr_scheduler", capture_scheduler)

    loader_datasets = []
    original_loader = run_training_module.MoleculeDataLoader

    def capture_loader(*args, **kwargs):
        dataset = kwargs.get("dataset", args[0] if args else None)
        loader_datasets.append(dataset)
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(run_training_module, "MoleculeDataLoader", capture_loader)

    valid_scores, test_scores = run_training_module.run_training(
        args, data, fold_num=0,
    )

    assert valid_scores == {}
    assert test_scores == {}
    assert sorted(epoch_counts.values()) == [2, 2]
    assert observed_train_sizes == [6, 6, 6, 6]
    assert scheduler_steps == [2, 2]
    assert loader_datasets == [data]

    for model_index in range(2):
        checkpoint_path = save_dir / f"model_{model_index}" / "model.pt"
        checkpoint_args = load_args(str(checkpoint_path))
        assert checkpoint_args.train_on_full_data is True
        assert checkpoint_args.train_data_size == 6
        assert checkpoint_args.full_data_requested_epochs == 2
        assert checkpoint_args.full_data_completed_epochs == 2
        assert checkpoint_args.full_data_seed == args.seed
        assert checkpoint_args.full_data_pytorch_seed == args.pytorch_seed
        assert checkpoint_args.full_data_validation_disabled is True
        assert checkpoint_args.full_data_checkpoint_policy == "last_epoch"
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        first_tensor = next(iter(state["state_dict"].values()))
        assert torch.all(first_tensor == 2)
        assert state["data_scaler"]["means"].tolist() == pytest.approx([3.5])
        load_checkpoint(str(checkpoint_path), device=torch.device("cpu"))


def test_normal_mode_still_rejects_an_empty_validation_split():
    args = SimpleNamespace(
        num_tasks=1,
        task_names=["target"],
        class_balance=False,
    )
    with pytest.raises(ValueError, match="validation data split is empty"):
        _validate_training_split(
            args,
            _dataset([1.0]),
            MoleculeDataset([]),
        )


@pytest.mark.parametrize(
    "missing_target",
    [None, np.asarray([None, None], dtype=object)],
)
def test_normal_mode_rejects_validation_without_observed_labels(
    missing_target,
):
    args = SimpleNamespace(
        num_tasks=1,
        task_names=["target"],
        class_balance=False,
    )
    validation_data = _dataset([missing_target])

    with pytest.raises(ValueError, match="validation data contain no observed labels"):
        _validate_training_split(args, _dataset([1.0]), validation_data)

    # Full-data mode deliberately does not depend on validation data.
    _validate_training_split(
        args,
        _dataset([1.0]),
        validation_data,
        require_validation=False,
    )


def test_run_training_resolves_weights_from_the_actual_training_split_only(
    tmp_path: Path, monkeypatch,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    data_path = tmp_path / "classification.csv"
    _write_dataset(data_path, [0, 1] * 5)
    complete_data = _dataset([0, 1] * 5)
    train_data = _dataset([0] * 8 + [1] * 2)
    observed_metadata = []

    original_resolver = run_training_module._resolve_balanced_class_weights

    def capture_resolver(args, received_train_data):
        assert received_train_data is train_data
        metadata = original_resolver(args, received_train_data)
        observed_metadata.append(metadata)
        return metadata

    monkeypatch.setattr(
        run_training_module, "_resolve_balanced_class_weights", capture_resolver,
    )
    monkeypatch.setattr(
        run_training_module,
        "train",
        lambda n_iter=0, **kwargs: n_iter + kwargs["data_loader"].iter_size,
    )
    metric_calls = []

    def unweighted_validation(**kwargs):
        metric_calls.append(kwargs)
        return {"binary_cross_entropy": [0.5]}

    def unweighted_predictions(**kwargs):
        metric_calls.append(kwargs)
        return {"binary_cross_entropy": [0.5]}

    monkeypatch.setattr(run_training_module, "evaluate", unweighted_validation)
    monkeypatch.setattr(
        run_training_module,
        "predict",
        lambda data_loader, **kwargs: [[0.5] for _ in range(data_loader.iter_size)],
    )
    monkeypatch.setattr(
        run_training_module,
        "evaluate_predictions",
        unweighted_predictions,
    )

    for run_index, held_out_targets in enumerate(([1] * 10, [0] * 10)):
        val_data = _dataset(held_out_targets)
        test_data = _dataset(list(reversed(held_out_targets)))
        monkeypatch.setattr(
            run_training_module,
            "split_data",
            lambda **kwargs: (train_data, val_data, test_data),
        )
        args = TrainArgs().parse_args(
            [
                "--data_path", str(data_path),
                "--dataset_type", "classification",
                "--class_weight", "balanced",
                "--metric", "binary_cross_entropy",
                "--epochs", "1",
                "--save_dir", str(tmp_path / f"normal_{run_index}"),
                "--hidden_size", "8",
                "--ffn_hidden_size", "8",
                "--depth", "1",
                "--ffn_num_layers", "1",
                "--batch_size", "5",
                "--num_workers", "0",
                "--no_cuda",
                "--quiet",
            ]
        )
        args.task_names = ["target"]
        args.features_size = None
        run_training_module.run_training(args, complete_data, fold_num=0)

        checkpoint_args = load_args(
            str(tmp_path / f"normal_{run_index}" / "model_0" / "model.pt")
        )
        assert checkpoint_args.resolved_class_counts == [8, 2]
        assert checkpoint_args.resolved_class_weights == pytest.approx([0.625, 2.5])

    assert observed_metadata[0] == observed_metadata[1]
    assert metric_calls
    assert all(
        "data_weights" not in call
        and "class_weight" not in call
        and "resolved_class_weights" not in call
        for call in metric_calls
    )


def _real_full_classification_args(
    data_path: Path, save_dir: Path, features_only: bool,
) -> TrainArgs:
    raw_args = [
        "--data_path", str(data_path),
        "--dataset_type", "classification",
        "--class_weight", "balanced",
        "--train_on_full_data",
        "--metric", "binary_cross_entropy",
        "--epochs", "1",
        "--ensemble_size", "1",
        "--save_dir", str(save_dir),
        "--hidden_size", "8",
        "--ffn_hidden_size", "8",
        "--depth", "1",
        "--ffn_num_layers", "1",
        "--batch_size", "5",
        "--num_workers", "0",
        "--seed", "7",
        "--pytorch_seed", "11",
        "--no_cuda",
        "--quiet",
    ]
    if features_only:
        raw_args.extend(["--features_generator", "morgan", "--features_only"])
    return TrainArgs().parse_args(raw_args)


@pytest.mark.parametrize("features_only", [False, True])
def test_real_full_data_class_weight_smoke_and_predict_round_trip(
    tmp_path: Path, monkeypatch, features_only: bool,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    data_path = tmp_path / "classification.csv"
    _write_dataset(data_path, [0] * 8 + [1] * 2)
    save_dir = tmp_path / ("features_only" if features_only else "dmpnn")
    args = _real_full_classification_args(data_path, save_dir, features_only)

    schedulers = []
    original_scheduler_builder = run_training_module.build_lr_scheduler

    def capture_scheduler(*builder_args, **builder_kwargs):
        scheduler = original_scheduler_builder(*builder_args, **builder_kwargs)
        schedulers.append(scheduler)
        return scheduler

    monkeypatch.setattr(
        run_training_module, "build_lr_scheduler", capture_scheduler,
    )

    assert cross_validate(args, run_training_module.run_training) == (None, None)

    checkpoint_path = save_dir / "fold_0" / "model_0" / "model.pt"
    checkpoint_args = load_args(str(checkpoint_path))
    assert checkpoint_args.resolved_class_counts == [8, 2]
    assert checkpoint_args.resolved_class_weights == pytest.approx([0.625, 2.5])
    assert checkpoint_args.class_weight_observed_count == 10
    assert checkpoint_args.full_data_completed_epochs == 1
    assert len(schedulers) == 1
    # PyTorch initializes _LRScheduler by priming the first learning rate.
    # Each optimizer update then consumes exactly one of total_steps rates;
    # the trailing scheduler step prepares (but never consumes) one extra rate.
    assert schedulers[0].current_step - 1 == schedulers[0].total_steps[0]
    assert not (save_dir / "fold_0" / "valid_scores.json").exists()
    assert not (save_dir / "fold_0" / "resume_manifest.json").exists()

    prediction_input = tmp_path / f"predict_{features_only}.csv"
    _write_smiles(prediction_input, SMILES[:3])
    predictions_path = tmp_path / f"predictions_{features_only}.csv"
    predict_raw_args = [
        "--test_path", str(prediction_input),
        "--preds_path", str(predictions_path),
        "--checkpoint_path", str(checkpoint_path),
        "--num_workers", "0",
        "--no_cuda",
    ]
    if features_only:
        predict_raw_args.extend(["--features_generator", "morgan"])
    predictions = make_predictions(PredictArgs().parse_args(predict_raw_args))

    assert len(predictions) == 3
    assert np.all(np.isfinite(np.asarray(predictions, dtype=float)))
    assert predictions_path.is_file()


def test_full_data_seed_reproducibility(tmp_path: Path):
    data_path = tmp_path / "classification.csv"
    _write_dataset(data_path, [0] * 8 + [1] * 2)
    checkpoint_states = []

    for run_index in range(2):
        save_dir = tmp_path / f"repeat_{run_index}"
        args = _real_full_classification_args(
            data_path, save_dir, features_only=True,
        )
        assert cross_validate(args, importlib.import_module(
            "chemprop.train.run_training"
        ).run_training) == (None, None)
        state = torch.load(
            save_dir / "fold_0" / "model_0" / "model.pt",
            map_location="cpu",
            weights_only=False,
        )["state_dict"]
        checkpoint_states.append(state)

    assert checkpoint_states[0].keys() == checkpoint_states[1].keys()
    for name in checkpoint_states[0]:
        assert torch.equal(checkpoint_states[0][name], checkpoint_states[1][name])


def test_full_data_external_test_feature_loading_cannot_change_training_seed(
    tmp_path: Path, monkeypatch,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    data_path = tmp_path / "classification.csv"
    test_path = tmp_path / "external.csv"
    _write_dataset(data_path, [0] * 8 + [1] * 2)
    _write_dataset(test_path, [0, 1])
    original_get_data = run_training_module.get_data
    original_predict = run_training_module.predict
    checkpoint_states = []

    def rng_consuming_get_data(*args, **kwargs):
        torch.rand(37)
        return original_get_data(*args, **kwargs)

    monkeypatch.setattr(
        run_training_module, "get_data", rng_consuming_get_data,
    )

    def rng_consuming_predict(*args, **kwargs):
        torch.rand(41)
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(
        run_training_module, "predict", rng_consuming_predict,
    )
    for run_index, use_external_test in enumerate((False, True)):
        save_dir = tmp_path / f"external_seed_{run_index}"
        args = _real_full_classification_args(
            data_path, save_dir, features_only=True,
        )
        if use_external_test:
            args.separate_test_path = str(test_path)
        args.ensemble_size = 2
        assert cross_validate(args, run_training_module.run_training) == (None, None)
        checkpoint_states.append([
            torch.load(
                save_dir / "fold_0" / f"model_{model_index}" / "model.pt",
                map_location="cpu",
                weights_only=False,
            )["state_dict"]
            for model_index in range(2)
        ])

    for model_index in range(2):
        for name in checkpoint_states[0][model_index]:
            assert torch.equal(
                checkpoint_states[0][model_index][name],
                checkpoint_states[1][model_index][name],
            )


def test_full_data_scalers_and_data_weights_exclude_separate_test(
    tmp_path: Path, monkeypatch,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_dataset(train_path, [1, 2, 3, 4, 5, 6])
    _write_dataset(test_path, [1000, 2000])
    train_features = np.asarray(
        [[0, 10], [2, 12], [4, 14], [6, 16], [8, 18], [10, 20]],
        dtype=float,
    )
    test_features = np.asarray([[10000, -10000], [20000, -20000]], dtype=float)
    train_features_path = tmp_path / "train_features.npz"
    test_features_path = tmp_path / "test_features.npz"
    np.savez_compressed(train_features_path, features=train_features)
    np.savez_compressed(test_features_path, features=test_features)
    weights_path = tmp_path / "weights.csv"
    weights_path.write_text(
        "weight\n1\n2\n3\n4\n5\n6\n", encoding="utf-8"
    )
    save_dir = tmp_path / "scaled"
    args = TrainArgs().parse_args(
        [
            "--data_path", str(train_path),
            "--dataset_type", "regression",
            "--train_on_full_data",
            "--epochs", "1",
            "--save_dir", str(save_dir),
            "--features_path", str(train_features_path),
            "--features_only",
            "--separate_test_path", str(test_path),
            "--separate_test_features_path", str(test_features_path),
            "--data_weights_path", str(weights_path),
            "--hidden_size", "8",
            "--ffn_hidden_size", "8",
            "--ffn_num_layers", "1",
            "--batch_size", "3",
            "--num_workers", "0",
            "--no_cuda",
            "--quiet",
        ]
    )
    monkeypatch.setattr(
        run_training_module,
        "train",
        lambda n_iter=0, **kwargs: n_iter + kwargs["data_loader"].iter_size,
    )

    assert cross_validate(args, run_training_module.run_training) == (None, None)

    checkpoint_path = save_dir / "fold_0" / "model_0" / "model.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert state["features_scaler"]["means"].tolist() == pytest.approx(
        train_features.mean(axis=0).tolist()
    )
    assert state["data_scaler"]["means"].tolist() == pytest.approx([3.5])
    assert (save_dir / "fold_0" / "test_scores.json").is_file()

    # Prediction must not depend on the training-only, row-aligned weights
    # file stored in checkpoint args. This also covers a different prediction
    # row count and deployment after that training-side file is unavailable.
    weights_path.unlink()
    prediction_path = tmp_path / "prediction.csv"
    _write_smiles(prediction_path, SMILES[:3])
    prediction_features_path = tmp_path / "prediction_features.npz"
    np.savez_compressed(
        prediction_features_path,
        features=np.asarray([[1, 2], [3, 4], [5, 6]], dtype=float),
    )
    predictions_path = tmp_path / "predictions.csv"
    predictions = make_predictions(PredictArgs().parse_args([
        "--test_path", str(prediction_path),
        "--preds_path", str(predictions_path),
        "--checkpoint_path", str(checkpoint_path),
        "--features_path", str(prediction_features_path),
        "--num_workers", "0",
        "--no_cuda",
    ]))

    assert len(predictions) == 3
    assert np.all(np.isfinite(np.asarray(predictions, dtype=float)))


def test_full_data_atom_bond_target_scaler_fits_all_rows_and_round_trips(
    tmp_path: Path, monkeypatch,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    data_path = tmp_path / "atom_bond.csv"
    data_path.write_text(
        "smiles,atom_value,bond_value\n"
        'CC,"[1.0, 3.0]","[10.0]"\n'
        'CCC,"[5.0, 7.0, 9.0]","[20.0, 30.0]"\n',
        encoding="utf-8",
    )
    save_dir = tmp_path / "atom_bond_full"
    args = TrainArgs().parse_args(
        [
            "--data_path", str(data_path),
            "--dataset_type", "regression",
            "--is_atom_bond_targets",
            "--train_on_full_data",
            "--epochs", "1",
            "--save_dir", str(save_dir),
            "--hidden_size", "8",
            "--ffn_hidden_size", "8",
            "--depth", "1",
            "--ffn_num_layers", "1",
            "--batch_size", "2",
            "--num_workers", "0",
            "--no_cuda",
            "--quiet",
        ]
    )
    monkeypatch.setattr(
        run_training_module,
        "train",
        lambda n_iter=0, **kwargs: n_iter + kwargs["data_loader"].iter_size,
    )

    assert cross_validate(args, run_training_module.run_training) == (None, None)

    checkpoint_path = save_dir / "fold_0" / "model_0" / "model.pt"
    scaler, _, _, _, atom_bond_scaler = load_scalers(str(checkpoint_path))
    assert scaler is None
    assert atom_bond_scaler.n_atom_targets == 1
    assert atom_bond_scaler.n_bond_targets == 1
    assert atom_bond_scaler.means[:, 0].tolist() == pytest.approx([5.0, 20.0])
    checkpoint_args = load_args(str(checkpoint_path))
    assert checkpoint_args.atom_targets == ["atom_value"]
    assert checkpoint_args.bond_targets == ["bond_value"]
    assert checkpoint_args.train_data_size == 2


def test_test_only_class_weight_checkpoint_never_recomputes_weights(
    tmp_path: Path, monkeypatch,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    data_path = tmp_path / "classification.csv"
    _write_dataset(data_path, [0] * 8 + [1] * 2)
    source_args = TrainArgs().parse_args(
        [
            "--data_path", str(data_path),
            "--dataset_type", "classification",
            "--class_weight", "balanced",
            "--no_cuda",
        ]
    )
    source_args.task_names = ["target"]
    source_args.features_size = None
    source_args.resolved_class_counts = [8, 2]
    source_args.resolved_class_weights = [0.625, 2.5]
    source_args.class_weight_observed_count = 10
    checkpoint_path = tmp_path / "weighted.pt"
    save_checkpoint(
        str(checkpoint_path), MoleculeModel(source_args), args=source_args,
    )
    test_args = TrainArgs().parse_args(
        [
            "--data_path", str(data_path),
            "--dataset_type", "classification",
            "--class_weight", "balanced",
            "--checkpoint_path", str(checkpoint_path),
            "--test",
            "--save_dir", str(tmp_path / "test_only"),
            "--num_workers", "0",
            "--no_cuda",
            "--quiet",
        ]
    )
    test_args.task_names = ["target"]
    test_args.features_size = None
    empty_train = MoleculeDataset([])
    validation = _dataset([0, 1])
    held_out = _dataset([0, 1])
    monkeypatch.setattr(
        run_training_module,
        "split_data",
        lambda **kwargs: (empty_train, validation, held_out),
    )
    monkeypatch.setattr(
        run_training_module,
        "_resolve_balanced_class_weights",
        lambda *_args, **_kwargs: pytest.fail(
            "test-only execution must not resolve class weights"
        ),
    )

    valid_scores, test_scores = run_training_module.run_training(
        test_args, MoleculeDataset([]), fold_num=0,
    )

    assert valid_scores.keys() == test_scores.keys() == {"auc"}


def test_full_data_class_balance_is_supported_with_actual_loader_length(
    tmp_path: Path, monkeypatch,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    data_path = tmp_path / "classification.csv"
    _write_dataset(data_path, [0] * 8 + [1] * 2)
    args = TrainArgs().parse_args(
        [
            "--data_path", str(data_path),
            "--dataset_type", "classification",
            "--train_on_full_data",
            "--class_balance",
            "--metric", "binary_cross_entropy",
            "--epochs", "1",
            "--save_dir", str(tmp_path / "balanced"),
            "--hidden_size", "8",
            "--ffn_hidden_size", "8",
            "--depth", "1",
            "--ffn_num_layers", "1",
            "--batch_size", "3",
            "--num_workers", "0",
            "--no_cuda",
            "--quiet",
        ]
    )
    args.task_names = ["target"]
    data = get_data(path=str(data_path), args=args, skip_none_targets=True)
    args.features_size = data.features_size()
    iter_sizes = []
    scheduler_steps = []
    original_scheduler_builder = run_training_module.build_lr_scheduler

    def fake_epoch(data_loader, n_iter=0, **kwargs):
        iter_sizes.append(data_loader.iter_size)
        return n_iter + data_loader.iter_size

    def capture_scheduler(*builder_args, **builder_kwargs):
        scheduler_steps.append(builder_kwargs["steps_per_epoch"])
        return original_scheduler_builder(*builder_args, **builder_kwargs)

    monkeypatch.setattr(run_training_module, "train", fake_epoch)
    monkeypatch.setattr(
        run_training_module, "build_lr_scheduler", capture_scheduler,
    )

    valid_scores, test_scores = run_training_module.run_training(
        args, data, fold_num=0,
    )

    assert valid_scores == test_scores == {}
    assert args.train_data_size == 10
    assert iter_sizes == [4]
    assert scheduler_steps == [2]


def test_normal_class_balance_scheduler_also_uses_effective_loader_length(
    tmp_path: Path, monkeypatch,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    data_path = tmp_path / "classification.csv"
    _write_dataset(data_path, [0] * 8 + [1] * 2)
    args = TrainArgs().parse_args(
        [
            "--data_path", str(data_path),
            "--dataset_type", "classification",
            "--class_balance",
            "--metric", "binary_cross_entropy",
            "--epochs", "1",
            "--save_dir", str(tmp_path / "normal_balanced"),
            "--hidden_size", "8",
            "--ffn_hidden_size", "8",
            "--depth", "1",
            "--ffn_num_layers", "1",
            "--batch_size", "3",
            "--num_workers", "0",
            "--no_cuda",
            "--quiet",
        ]
    )
    args.task_names = ["target"]
    data = get_data(path=str(data_path), args=args, skip_none_targets=True)
    args.features_size = data.features_size()
    train_data = _dataset([0] * 8 + [1] * 2)
    val_data = _dataset([0, 1])
    test_data = _dataset([0, 1])
    scheduler_steps = []
    original_scheduler_builder = run_training_module.build_lr_scheduler

    monkeypatch.setattr(
        run_training_module,
        "split_data",
        lambda **kwargs: (train_data, val_data, test_data),
    )
    monkeypatch.setattr(
        run_training_module,
        "train",
        lambda n_iter=0, **kwargs: n_iter + kwargs["data_loader"].iter_size,
    )
    monkeypatch.setattr(
        run_training_module,
        "evaluate",
        lambda **kwargs: {"binary_cross_entropy": [0.5]},
    )
    monkeypatch.setattr(
        run_training_module,
        "predict",
        lambda data_loader, **kwargs: [[0.5] for _ in range(data_loader.iter_size)],
    )
    monkeypatch.setattr(
        run_training_module,
        "evaluate_predictions",
        lambda **kwargs: {"binary_cross_entropy": [0.5]},
    )

    def capture_scheduler(*builder_args, **builder_kwargs):
        scheduler_steps.append(builder_kwargs["steps_per_epoch"])
        return original_scheduler_builder(*builder_args, **builder_kwargs)

    monkeypatch.setattr(
        run_training_module, "build_lr_scheduler", capture_scheduler,
    )

    run_training_module.run_training(args, data, fold_num=0)

    assert train_data is not data
    assert args.train_data_size == 10
    assert scheduler_steps == [2]


def test_normal_training_nonfinite_validation_metric_fails_fast(
    tmp_path: Path,
    monkeypatch,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    data_path = tmp_path / "classification.csv"
    _write_dataset(data_path, [0] * 8 + [1] * 2)
    args = TrainArgs().parse_args(
        [
            "--data_path", str(data_path),
            "--dataset_type", "classification",
            "--metric", "auc",
            "--epochs", "3",
            "--early_stopping", "1",
            "--save_dir", str(tmp_path / "normal_nonfinite"),
            "--hidden_size", "8",
            "--ffn_hidden_size", "8",
            "--depth", "1",
            "--ffn_num_layers", "1",
            "--batch_size", "4",
            "--num_workers", "0",
            "--no_cuda",
            "--quiet",
        ]
    )
    args.task_names = ["target"]
    data = get_data(path=str(data_path), args=args, skip_none_targets=True)
    args.features_size = data.features_size()
    train_data = _dataset([0, 1, 0, 1, 0, 1])
    val_data = _dataset([0, 1])
    test_data = _dataset([1, 1])
    monkeypatch.setattr(
        run_training_module,
        "split_data",
        lambda **kwargs: (train_data, val_data, test_data),
    )

    train_calls = 0

    def fake_train_epoch(model, n_iter=0, data_loader=None, **kwargs):
        nonlocal train_calls
        train_calls += 1
        return n_iter + data_loader.iter_size

    monkeypatch.setattr(run_training_module, "train", fake_train_epoch)
    monkeypatch.setattr(
        run_training_module,
        "evaluate",
        lambda **kwargs: {"auc": [np.nan]},
    )

    with pytest.raises(ValueError, match="cannot select a trained checkpoint"):
        run_training_module.run_training(args, data, fold_num=0)

    assert train_calls == 1


def test_normal_zero_epoch_logging_reports_retained_initialization(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    data_path = tmp_path / "classification.csv"
    _write_dataset(data_path, [0, 1] * 5)
    args = TrainArgs().parse_args(
        [
            "--data_path", str(data_path),
            "--dataset_type", "classification",
            "--metric", "binary_cross_entropy",
            "--epochs", "0",
            "--save_dir", str(tmp_path / "zero_epoch"),
            "--hidden_size", "8",
            "--ffn_hidden_size", "8",
            "--depth", "1",
            "--ffn_num_layers", "1",
            "--batch_size", "4",
            "--num_workers", "0",
            "--no_cuda",
            "--quiet",
        ]
    )
    args.task_names = ["target"]
    data = get_data(path=str(data_path), args=args, skip_none_targets=True)
    args.features_size = data.features_size()
    monkeypatch.setattr(
        run_training_module,
        "split_data",
        lambda **kwargs: (
            _dataset([0, 1, 0, 1, 0, 1]),
            _dataset([0, 1]),
            _dataset([0, 1]),
        ),
    )

    run_training_module.run_training(args, data, fold_num=0)

    log_text = capsys.readouterr().out
    assert "zero epochs requested" in log_text
    assert "initialization checkpoint was retained" in log_text
    assert "using the trained epoch" not in log_text
    assert (tmp_path / "zero_epoch" / "model_0" / "model.pt").is_file()


def test_cross_validation_nonfinite_validation_summary_is_not_logged_as_nan(
    tmp_path: Path,
):
    data_path = tmp_path / "regression.csv"
    _write_dataset(data_path, list(range(10)))
    save_dir = tmp_path / "nonfinite_validation_summary"
    args = TrainArgs().parse_args(
        [
            "--data_path", str(data_path),
            "--dataset_type", "regression",
            "--epochs", "1",
            "--save_dir", str(save_dir),
            "--num_workers", "0",
            "--no_cuda",
            "--quiet",
        ]
    )

    result = cross_validate(
        args,
        lambda *_args, **_kwargs: (
            {"rmse": [np.nan]},
            {"rmse": [0.5]},
        ),
    )

    assert result == pytest.approx((0.5, 0.0))
    log_text = (save_dir / "quiet.log").read_text(encoding="utf-8")
    assert "Overall valid rmse: not evaluated" in log_text
    assert "Overall valid rmse = nan" not in log_text.lower()


def test_full_data_undefined_external_metric_is_not_logged_as_a_nan_result(
    tmp_path: Path, monkeypatch,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "one_class_test.csv"
    _write_dataset(train_path, [0] * 8 + [1] * 2)
    _write_dataset(test_path, [1, 1])
    save_dir = tmp_path / "undefined_metric"
    args = TrainArgs().parse_args(
        [
            "--data_path", str(train_path),
            "--dataset_type", "classification",
            "--train_on_full_data",
            "--separate_test_path", str(test_path),
            "--metric", "auc",
            "--epochs", "1",
            "--save_dir", str(save_dir),
            "--hidden_size", "8",
            "--ffn_hidden_size", "8",
            "--depth", "1",
            "--ffn_num_layers", "1",
            "--batch_size", "5",
            "--num_workers", "0",
            "--no_cuda",
            "--quiet",
        ]
    )
    monkeypatch.setattr(
        run_training_module,
        "train",
        lambda n_iter=0, **kwargs: n_iter + kwargs["data_loader"].iter_size,
    )

    assert cross_validate(args, run_training_module.run_training) == (None, None)

    log_text = (save_dir / "quiet.log").read_text(encoding="utf-8")
    assert "test auc: not evaluated" in log_text
    assert "auc = nan" not in log_text.lower()
    with (save_dir / "fold_0" / "test_scores.json").open() as score_file:
        assert json.load(score_file) == {"auc": [None]}


def test_full_data_unlabeled_external_test_can_still_save_predictions(
    tmp_path: Path, monkeypatch,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "unlabeled_test.csv"
    _write_dataset(train_path, [0] * 8 + [1] * 2)
    _write_dataset(test_path, [None, None])
    save_dir = tmp_path / "unlabeled_predictions"
    args = _real_full_classification_args(train_path, save_dir, False)
    args.separate_test_path = str(test_path)
    args.save_preds = True
    monkeypatch.setattr(
        run_training_module,
        "train",
        lambda n_iter=0, **kwargs: n_iter + kwargs["data_loader"].iter_size,
    )

    assert cross_validate(args, run_training_module.run_training) == (None, None)

    prediction_path = save_dir / "fold_0" / "test_preds.csv"
    prediction_lines = prediction_path.read_text(encoding="utf-8").splitlines()
    assert prediction_lines[0] == "smiles,target"
    assert len(prediction_lines) == 3
    assert all(
        np.isfinite(float(line.rsplit(",", maxsplit=1)[1]))
        for line in prediction_lines[1:]
    )
    log_text = (save_dir / "quiet.log").read_text(encoding="utf-8")
    assert "test metrics: not evaluated" in log_text.lower()


def test_full_data_rejects_nonempty_fold_output_without_deleting_it(tmp_path: Path):
    data_path = tmp_path / "classification.csv"
    _write_dataset(data_path, [0] * 8 + [1] * 2)
    save_dir = tmp_path / "reused"
    stale_checkpoint = save_dir / "fold_0" / "model_4" / "model.pt"
    stale_checkpoint.parent.mkdir(parents=True)
    stale_checkpoint.write_bytes(b"stale")
    args = _real_full_classification_args(
        data_path, save_dir, features_only=True,
    )

    with pytest.raises(ValueError, match="requires a fresh fold_0"):
        cross_validate(args, lambda *_args, **_kwargs: ({}, {}))

    assert stale_checkpoint.read_bytes() == b"stale"


def test_full_data_split_indices_keep_empty_validation_position(
    tmp_path: Path, monkeypatch,
):
    run_training_module = importlib.import_module("chemprop.train.run_training")
    data_path = tmp_path / "classification.csv"
    _write_dataset(data_path, [0] * 8 + [1] * 2)
    save_dir = tmp_path / "saved_splits"
    args = _real_full_classification_args(
        data_path, save_dir, features_only=True,
    )
    args.save_smiles_splits = True
    monkeypatch.setattr(
        run_training_module,
        "train",
        lambda n_iter=0, **kwargs: n_iter + kwargs["data_loader"].iter_size,
    )

    assert cross_validate(args, run_training_module.run_training) == (None, None)

    with (save_dir / "fold_0" / "split_indices.pckl").open("rb") as split_file:
        split_indices = pickle.load(split_file)
    assert split_indices == [list(range(10)), [], []]
    assert not (save_dir / "fold_0" / "val_smiles.csv").exists()


def test_full_data_rejects_zero_retained_training_data_weight(tmp_path: Path):
    data_path = tmp_path / "regression.csv"
    _write_dataset(data_path, [None, 1.0, 2.0])
    weights_path = tmp_path / "weights.csv"
    weights_path.write_text("weight\n1\n0\n0\n", encoding="utf-8")
    args = TrainArgs().parse_args(
        [
            "--data_path", str(data_path),
            "--dataset_type", "regression",
            "--data_weights_path", str(weights_path),
            "--train_on_full_data",
            "--epochs", "1",
            "--save_dir", str(tmp_path / "zero_weight"),
            "--num_workers", "0",
            "--no_cuda",
            "--quiet",
        ]
    )

    with pytest.raises(ValueError, match="zero total positive weight"):
        cross_validate(args, importlib.import_module(
            "chemprop.train.run_training"
        ).run_training)
