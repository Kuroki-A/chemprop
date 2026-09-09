"""Regression tests for exact checkpoint restore and frozen transfer semantics."""

import csv
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from chemprop.args import TrainArgs
from chemprop.data import (
    MoleculeDataLoader,
    MoleculeDatapoint,
    MoleculeDataset,
    StandardScaler,
    get_data,
)
from chemprop.models import MoleculeModel
from chemprop.train.run_training import (
    _load_test_checkpoint_scalers,
    _validate_checkpoint_scaler_contract,
    run_training,
)
from chemprop.utils import (
    load_checkpoint,
    load_checkpoint_for_training,
    load_args,
    load_frzn_model,
    save_checkpoint,
)


DATA_PATH = Path(__file__).parents[1] / "data" / "regression.csv"


def _args(*extra: str, task_names=("logSolubility",)) -> TrainArgs:
    args = TrainArgs().parse_args(
        [
            "--data_path",
            str(DATA_PATH),
            "--dataset_type",
            "regression",
            "--no_cuda",
            *extra,
        ]
    )
    args.task_names = list(task_names)
    return args


def _fill_parameters(model: MoleculeModel) -> None:
    """Makes copied and untouched parameters unambiguous in assertions."""
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index + 1) / 100)


def _clone_state(model: MoleculeModel):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_prefix_copied(
    source_state, target_state, source_prefix: str, target_prefix: str
) -> None:
    source_by_suffix = {
        name[len(source_prefix) :]: value
        for name, value in source_state.items()
        if name.startswith(source_prefix)
    }
    target_by_suffix = {
        name[len(target_prefix) :]: value
        for name, value in target_state.items()
        if name.startswith(target_prefix)
    }
    assert source_by_suffix.keys() == target_by_suffix.keys()
    for suffix, source_value in source_by_suffix.items():
        torch.testing.assert_close(target_by_suffix[suffix], source_value)


def _assert_state_unchanged(before, after, prefix: str) -> None:
    names = [name for name in before if name.startswith(prefix)]
    assert names
    for name in names:
        torch.testing.assert_close(after[name], before[name])


def _assert_frozen(model: MoleculeModel, prefix: str) -> None:
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(prefix)
    ]
    assert parameters
    assert all(not parameter.requires_grad for parameter in parameters)


def test_load_checkpoint_reconstructs_saved_nondefault_architecture(tmp_path: Path):
    args = _args(
        "--hidden_size",
        "17",
        "--depth",
        "5",
        "--ffn_hidden_size",
        "13",
        "--ffn_num_layers",
        "3",
    )
    source = MoleculeModel(args)
    _fill_parameters(source)
    checkpoint_path = tmp_path / "nondefault.pt"
    save_checkpoint(checkpoint_path, source, args=args)

    loaded = load_checkpoint(checkpoint_path, device=torch.device("cpu"))

    assert loaded.encoder.encoder[0].hidden_size == 17
    assert loaded.encoder.encoder[0].depth == 5
    assert [layer.out_features for layer in loaded.readout if isinstance(layer, torch.nn.Linear)] == [13, 13, 1]
    for name, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_load_checkpoint_reconstructs_saved_features_only_architecture(tmp_path: Path):
    args = _args(
        "--features_generator",
        "morgan",
        "--features_only",
        "--ffn_hidden_size",
        "13",
        "--ffn_num_layers",
        "3",
    )
    args.features_size = 23
    source = MoleculeModel(args)
    _fill_parameters(source)
    checkpoint_path = tmp_path / "features_only.pt"
    save_checkpoint(checkpoint_path, source, args=args)

    loaded = load_checkpoint(checkpoint_path, device=torch.device("cpu"))

    assert loaded.encoder.features_only is True
    linears = [layer for layer in loaded.readout if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (23, 13),
        (13, 13),
        (13, 1),
    ]
    for name, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_test_mode_rejects_frozen_checkpoint_transfer(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()

    with pytest.raises(ValueError, match="cannot be combined with --checkpoint_frzn"):
        _args(
            "--checkpoint_path",
            str(checkpoint_path),
            "--checkpoint_frzn",
            str(checkpoint_path),
            "--test",
        )


def test_run_training_test_mode_preserves_saved_architecture(tmp_path: Path):
    source_args = _args(
        "--hidden_size",
        "17",
        "--depth",
        "5",
        "--ffn_hidden_size",
        "13",
        "--ffn_num_layers",
        "3",
    )
    source = MoleculeModel(source_args)
    checkpoint_path = tmp_path / "source.pt"
    save_checkpoint(
        checkpoint_path,
        source,
        scaler=StandardScaler(
            means=np.array([0.0]), stds=np.array([1.0])
        ),
        args=source_args,
    )

    output_dir = tmp_path / "test_output"
    test_args = _args(
        "--checkpoint_path",
        str(checkpoint_path),
        "--test",
        "--save_dir",
        str(output_dir),
        "--num_workers",
        "0",
    )
    data = get_data(path=str(DATA_PATH), args=test_args, skip_none_targets=True)
    test_args.features_size = data.features_size()

    valid_scores, test_scores = run_training(test_args, data, fold_num=0)

    assert valid_scores.keys() == test_scores.keys() == {"rmse"}
    rewritten = load_checkpoint(
        output_dir / "model_0" / "model.pt", device=torch.device("cpu")
    )
    assert rewritten.encoder.encoder[0].hidden_size == 17
    assert rewritten.encoder.encoder[0].depth == 5
    assert [
        layer.out_features
        for layer in rewritten.readout
        if isinstance(layer, torch.nn.Linear)
    ] == [13, 13, 1]


def test_run_training_test_mode_preserves_features_only_architecture(tmp_path: Path):
    source_args = _args(
        "--features_generator",
        "morgan",
        "--features_only",
        "--ffn_hidden_size",
        "13",
        "--ffn_num_layers",
        "3",
    )
    source_data = get_data(
        path=str(DATA_PATH), args=source_args, skip_none_targets=True
    )
    source_args.features_size = source_data.features_size()
    source = MoleculeModel(source_args)
    checkpoint_path = tmp_path / "features-only-source.pt"
    save_checkpoint(
        checkpoint_path,
        source,
        scaler=StandardScaler(
            means=np.array([0.0]), stds=np.array([1.0])
        ),
        features_scaler=StandardScaler(
            means=np.zeros(source_args.features_size),
            stds=np.ones(source_args.features_size),
        ),
        args=source_args,
    )

    output_dir = tmp_path / "features_only_test_output"
    test_args = _args(
        "--features_generator",
        "morgan",
        "--features_only",
        "--checkpoint_path",
        str(checkpoint_path),
        "--test",
        "--save_dir",
        str(output_dir),
        "--num_workers",
        "0",
    )
    test_data = get_data(
        path=str(DATA_PATH), args=test_args, skip_none_targets=True
    )
    test_args.features_size = test_data.features_size()

    valid_scores, test_scores = run_training(test_args, test_data, fold_num=0)

    assert valid_scores.keys() == test_scores.keys() == {"rmse"}
    rewritten = load_checkpoint(
        output_dir / "model_0" / "model.pt", device=torch.device("cpu")
    )
    assert rewritten.encoder.features_only is True
    linears = [
        layer for layer in rewritten.readout if isinstance(layer, torch.nn.Linear)
    ]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (source_args.features_size, 13),
        (13, 13),
        (13, 1),
    ]


def _write_regression_data(path: Path, target_shift: float) -> None:
    smiles = [
        "C",
        "CC",
        "CCC",
        "O",
        "CO",
        "CCO",
        "N",
        "CN",
        "CCN",
        "c1ccccc1",
    ]
    rows = ["smiles,target"]
    for index in range(30):
        rows.append(f"{smiles[index % len(smiles)]},{index / 10 + target_shift}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _run_checkpoint_test(
    data_path: Path, checkpoint_path: Path, save_dir: Path
) -> list:
    args = TrainArgs().parse_args(
        [
            "--data_path",
            str(data_path),
            "--dataset_type",
            "regression",
            "--checkpoint_path",
            str(checkpoint_path),
            "--test",
            "--save_preds",
            "--save_dir",
            str(save_dir),
            "--split_sizes",
            "0.6",
            "0.2",
            "0.2",
            "--num_workers",
            "0",
            "--no_cuda",
        ]
    )
    args.task_names = ["target"]
    data = get_data(path=str(data_path), args=args, skip_none_targets=True)
    args.features_size = data.features_size()
    run_training(args, data, fold_num=0)
    with (save_dir / "test_preds.csv").open(newline="", encoding="utf-8") as file:
        return [float(row["target"]) for row in csv.DictReader(file)]


def test_test_mode_predictions_use_checkpoint_scaler_not_evaluation_labels(
    tmp_path: Path,
):
    source_data_path = tmp_path / "source.csv"
    shifted_data_path = tmp_path / "shifted.csv"
    _write_regression_data(source_data_path, target_shift=0)
    _write_regression_data(shifted_data_path, target_shift=100)

    source_dir = tmp_path / "source_model"
    source_args = TrainArgs().parse_args(
        [
            "--data_path",
            str(source_data_path),
            "--dataset_type",
            "regression",
            "--save_dir",
            str(source_dir),
            "--epochs",
            "1",
            "--hidden_size",
            "17",
            "--ffn_hidden_size",
            "13",
            "--depth",
            "2",
            "--split_sizes",
            "0.6",
            "0.2",
            "0.2",
            "--batch_size",
            "10",
            "--num_workers",
            "0",
            "--no_cuda",
        ]
    )
    source_args.task_names = ["target"]
    source_data = get_data(
        path=str(source_data_path), args=source_args, skip_none_targets=True
    )
    source_args.features_size = source_data.features_size()
    run_training(source_args, source_data, fold_num=0)
    checkpoint_path = source_dir / "model_0" / "model.pt"
    checkpoint_bytes = checkpoint_path.read_bytes()

    unshifted_dir = tmp_path / "unshifted_evaluation"
    shifted_dir = tmp_path / "shifted_evaluation"
    unshifted_predictions = _run_checkpoint_test(
        source_data_path, checkpoint_path, unshifted_dir
    )
    shifted_predictions = _run_checkpoint_test(
        shifted_data_path, checkpoint_path, shifted_dir
    )

    assert checkpoint_path.read_bytes() == checkpoint_bytes
    assert (unshifted_dir / "model_0" / "model.pt").read_bytes() == checkpoint_bytes
    assert (shifted_dir / "model_0" / "model.pt").read_bytes() == checkpoint_bytes
    assert shifted_predictions == pytest.approx(unshifted_predictions, abs=1e-7)


def test_test_mode_rejects_ensemble_with_different_checkpoint_scalers(
    tmp_path: Path,
):
    data_path = tmp_path / "ensemble-data.csv"
    _write_regression_data(data_path, target_shift=0)
    source_args = TrainArgs().parse_args(
        [
            "--data_path",
            str(data_path),
            "--dataset_type",
            "regression",
            "--hidden_size",
            "17",
            "--depth",
            "2",
            "--no_cuda",
        ]
    )
    source_args.task_names = ["target"]
    source = MoleculeModel(source_args)
    first_checkpoint = tmp_path / "first.pt"
    second_checkpoint = tmp_path / "second.pt"
    save_checkpoint(
        first_checkpoint,
        source,
        scaler=StandardScaler(
            means=np.array([0.0]), stds=np.array([1.0])
        ),
        args=source_args,
    )
    save_checkpoint(
        second_checkpoint,
        source,
        scaler=StandardScaler(
            means=np.array([1.0]), stds=np.array([1.0])
        ),
        args=source_args,
    )

    test_args = TrainArgs().parse_args(
        [
            "--data_path",
            str(data_path),
            "--dataset_type",
            "regression",
            "--checkpoint_paths",
            str(first_checkpoint),
            str(second_checkpoint),
            "--test",
            "--save_dir",
            str(tmp_path / "ensemble"),
            "--num_workers",
            "0",
            "--no_cuda",
        ]
    )
    test_args.task_names = ["target"]
    data = get_data(path=str(data_path), args=test_args, skip_none_targets=True)
    test_args.features_size = data.features_size()

    with pytest.raises(ValueError, match="checkpoint scalers do not match"):
        run_training(test_args, data, fold_num=0)


def _checkpoint_scaler_contract_args(
    *,
    dataset_type="regression",
    molecular_features=False,
    molecular_scaling=True,
    atom_descriptors=None,
    atom_scaling=True,
    bond_descriptors=None,
    bond_scaling=True,
    atom_bond_targets=False,
):
    args = _args()
    args.dataset_type = dataset_type
    args.features_generator = ["morgan"] if molecular_features else None
    args.no_features_scaling = not molecular_scaling
    args.atom_descriptors = atom_descriptors
    args.no_atom_descriptor_scaling = not atom_scaling
    args.bond_descriptors = bond_descriptors
    args.no_bond_descriptor_scaling = not bond_scaling
    args.is_atom_bond_targets = atom_bond_targets
    return args


@pytest.mark.parametrize(
    "contract_args,scaler_slots,error_scaler",
    [
        (
            _checkpoint_scaler_contract_args(molecular_features=True),
            (True, False, False, False, False),
            "molecular feature",
        ),
        (
            _checkpoint_scaler_contract_args(),
            (True, True, False, False, False),
            "molecular feature",
        ),
        (
            _checkpoint_scaler_contract_args(
                molecular_features=True, molecular_scaling=False,
            ),
            (True, True, False, False, False),
            "molecular feature",
        ),
        (
            _checkpoint_scaler_contract_args(atom_descriptors="descriptor"),
            (True, False, False, False, False),
            "atom descriptor",
        ),
        (
            _checkpoint_scaler_contract_args(),
            (True, False, True, False, False),
            "atom descriptor",
        ),
        (
            _checkpoint_scaler_contract_args(
                atom_descriptors="feature", atom_scaling=False,
            ),
            (True, False, True, False, False),
            "atom descriptor",
        ),
        (
            _checkpoint_scaler_contract_args(bond_descriptors="descriptor"),
            (True, False, False, False, False),
            "bond descriptor",
        ),
        (
            _checkpoint_scaler_contract_args(),
            (True, False, False, True, False),
            "bond descriptor",
        ),
        (
            _checkpoint_scaler_contract_args(
                bond_descriptors="feature", bond_scaling=False,
            ),
            (True, False, False, True, False),
            "bond descriptor",
        ),
        (
            _checkpoint_scaler_contract_args(),
            (False, False, False, False, False),
            "target",
        ),
        (
            _checkpoint_scaler_contract_args(atom_bond_targets=True),
            (False, False, False, False, False),
            "atom/bond target",
        ),
        (
            _checkpoint_scaler_contract_args(atom_bond_targets=True),
            (True, False, False, False, True),
            "target",
        ),
        (
            _checkpoint_scaler_contract_args(dataset_type="classification"),
            (True, False, False, False, False),
            "target",
        ),
        (
            _checkpoint_scaler_contract_args(dataset_type="multiclass"),
            (False, False, False, False, True),
            "atom/bond target",
        ),
        (
            _checkpoint_scaler_contract_args(dataset_type="spectra"),
            (True, False, False, False, False),
            "target",
        ),
    ],
    ids=[
        "missing-molecular-feature",
        "unexpected-molecular-feature-channel",
        "unexpected-disabled-molecular-feature",
        "missing-atom-descriptor",
        "unexpected-atom-descriptor-channel",
        "unexpected-disabled-atom-descriptor",
        "missing-bond-descriptor",
        "unexpected-bond-descriptor-channel",
        "unexpected-disabled-bond-descriptor",
        "missing-regression-target",
        "missing-atom-bond-target",
        "wrong-regression-target-channel",
        "classification-stray-target",
        "multiclass-stray-atom-bond-target",
        "spectra-stray-target",
    ],
)
def test_test_checkpoint_scaler_presence_must_match_saved_semantics(
    contract_args, scaler_slots, error_scaler,
):
    scalers = tuple(object() if present else None for present in scaler_slots)

    with pytest.raises(
        ValueError,
        match=rf"{error_scaler} scaler which is (missing|unexpectedly present)",
    ):
        _validate_checkpoint_scaler_contract(
            "inconsistent.pt", contract_args, scalers,
        )


def test_test_checkpoint_scaler_contract_uses_v1_defaults_for_missing_flags():
    """Fresh TrainArgs mirrors legacy checkpoints with new flags absent."""
    legacy_args = TrainArgs()
    legacy_args.dataset_type = "classification"
    legacy_args.features_generator = ["morgan"]
    legacy_args.atom_descriptors = "descriptor"

    _validate_checkpoint_scaler_contract(
        "legacy.pt",
        legacy_args,
        (None, object(), object(), None, None),
    )


def test_test_checkpoint_scaler_contract_checks_every_ensemble_member(
    tmp_path: Path,
):
    source_args = _args("--features_generator", "morgan")
    source_args.features_size = 23
    source = MoleculeModel(source_args)
    target_scaler = StandardScaler(
        means=np.array([0.0]), stds=np.array([1.0])
    )
    features_scaler = StandardScaler(
        means=np.zeros(source_args.features_size),
        stds=np.ones(source_args.features_size),
    )
    first_checkpoint = tmp_path / "valid-first.pt"
    second_checkpoint = tmp_path / "missing-second.pt"
    save_checkpoint(
        first_checkpoint,
        source,
        scaler=target_scaler,
        features_scaler=features_scaler,
        args=source_args,
    )
    save_checkpoint(
        second_checkpoint,
        source,
        scaler=target_scaler,
        args=source_args,
    )

    with pytest.raises(
        ValueError,
        match=r"missing-second\.pt.*molecular feature scaler which is missing",
    ):
        _load_test_checkpoint_scalers(
            [str(first_checkpoint), str(second_checkpoint)],
            [load_args(str(first_checkpoint)), load_args(str(second_checkpoint))],
        )


def test_test_mode_does_not_depend_on_unused_training_labels_or_balancing(
    tmp_path: Path, monkeypatch,
):
    """Evaluation-only restore must not sample or inspect the training split."""
    data_path = tmp_path / "classification.csv"
    data_path.write_text("smiles,target\nC,0\nCC,1\n", encoding="utf-8")

    source_args = TrainArgs().parse_args(
        [
            "--data_path",
            str(data_path),
            "--dataset_type",
            "classification",
            "--no_cuda",
        ]
    )
    source_args.task_names = ["target"]
    checkpoint_path = tmp_path / "classification.pt"
    save_checkpoint(
        checkpoint_path, MoleculeModel(source_args), args=source_args,
    )

    test_args = TrainArgs().parse_args(
        [
            "--data_path",
            str(data_path),
            "--dataset_type",
            "classification",
            "--checkpoint_path",
            str(checkpoint_path),
            "--test",
            "--class_balance",
            "--save_dir",
            str(tmp_path / "evaluation"),
            "--num_workers",
            "0",
            "--no_cuda",
        ]
    )
    test_args.task_names = ["target"]
    test_args.features_size = None

    train_data = MoleculeDataset([])
    val_data = MoleculeDataset([
        MoleculeDatapoint(smiles=["C"], targets=[0.0]),
        MoleculeDatapoint(smiles=["CC"], targets=[1.0]),
    ])
    held_out_data = MoleculeDataset([
        MoleculeDatapoint(smiles=["CCC"], targets=[0.0]),
        MoleculeDatapoint(smiles=["O"], targets=[1.0]),
    ])

    run_training_module = importlib.import_module("chemprop.train.run_training")
    monkeypatch.setattr(
        run_training_module,
        "split_data",
        lambda **_kwargs: (train_data, val_data, held_out_data),
    )
    loader_options = []

    def capture_loader(*args, **kwargs):
        if kwargs.get("dataset") is train_data:
            loader_options.append(
                (kwargs.get("class_balance"), kwargs.get("shuffle"))
            )
        return MoleculeDataLoader(*args, **kwargs)

    monkeypatch.setattr(run_training_module, "MoleculeDataLoader", capture_loader)

    validation_scores, test_scores = run_training_module.run_training(
        test_args,
        MoleculeDataset([]),
        fold_num=0,
    )

    assert validation_scores.keys() == test_scores.keys() == {"auc"}
    assert loader_options == [(False, False)]


def test_features_only_warm_start_copies_compatible_ffn_state(tmp_path: Path):
    source_args = _args(
        "--features_generator",
        "morgan",
        "--features_only",
        "--ffn_hidden_size",
        "11",
        "--ffn_num_layers",
        "3",
        task_names=("source",),
    )
    source_args.features_size = 19
    source = MoleculeModel(source_args)
    _fill_parameters(source)
    checkpoint_path = tmp_path / "features_only.pt"
    save_checkpoint(checkpoint_path, source, args=source_args)

    current_args = _args(
        "--features_generator",
        "morgan",
        "--features_only",
        "--ffn_hidden_size",
        "11",
        "--ffn_num_layers",
        "3",
        task_names=("first", "second"),
    )
    current_args.features_size = 19
    torch.manual_seed(97)
    untouched = MoleculeModel(current_args)
    untouched_state = _clone_state(untouched)
    torch.manual_seed(97)

    loaded = load_checkpoint_for_training(checkpoint_path, current_args)
    source_state = source.state_dict()
    loaded_state = loaded.state_dict()

    for name in ("readout.1.weight", "readout.1.bias", "readout.4.weight", "readout.4.bias"):
        torch.testing.assert_close(loaded_state[name], source_state[name])
    for name in ("readout.7.weight", "readout.7.bias"):
        assert loaded_state[name].shape[0] == 2
        torch.testing.assert_close(loaded_state[name], untouched_state[name])


@pytest.mark.parametrize("checkpoint_features_only", [False, True])
def test_warm_start_rejects_features_only_mpn_mode_mismatch(
    tmp_path: Path, checkpoint_features_only: bool
):
    source_extra = (
        ("--features_generator", "morgan", "--features_only")
        if checkpoint_features_only
        else ()
    )
    source_args = _args(*source_extra)
    if checkpoint_features_only:
        source_args.features_size = 19
    checkpoint_path = tmp_path / f"source-{checkpoint_features_only}.pt"
    save_checkpoint(checkpoint_path, MoleculeModel(source_args), args=source_args)

    current_extra = () if checkpoint_features_only else (
        "--features_generator",
        "morgan",
        "--features_only",
    )
    current_args = _args(*current_extra)
    if not checkpoint_features_only:
        current_args.features_size = 19

    with pytest.raises(ValueError, match="both use features_only"):
        load_checkpoint_for_training(checkpoint_path, current_args)


@pytest.mark.parametrize("freeze_first_only", [False, True])
def test_one_encoder_frozen_checkpoint_transfers_to_two_encoders(
    tmp_path: Path, freeze_first_only: bool
):
    source_args = _args("--hidden_size", "17", "--depth", "2")
    source = MoleculeModel(source_args)
    _fill_parameters(source)
    source_state = _clone_state(source)
    checkpoint_path = tmp_path / "one-encoder.pt"
    save_checkpoint(checkpoint_path, source, args=source_args)

    current_args = _args("--hidden_size", "17", "--depth", "2")
    current_args.number_of_molecules = 2
    current_args.freeze_first_only = freeze_first_only
    current_args.frzn_ffn_layers = 0
    torch.manual_seed(101)
    current = MoleculeModel(current_args)
    before_state = _clone_state(current)
    before_requires_grad = {
        name: parameter.requires_grad for name, parameter in current.named_parameters()
    }

    loaded = load_frzn_model(current, checkpoint_path, current_args=current_args)
    loaded_state = loaded.state_dict()

    _assert_prefix_copied(
        source_state, loaded_state, "encoder.encoder.0.", "encoder.encoder.0."
    )
    _assert_frozen(loaded, "encoder.encoder.0.")
    if freeze_first_only:
        _assert_state_unchanged(before_state, loaded_state, "encoder.encoder.1.")
        for name, parameter in loaded.named_parameters():
            if name.startswith("encoder.encoder.1."):
                assert parameter.requires_grad == before_requires_grad[name]
    else:
        _assert_prefix_copied(
            source_state, loaded_state, "encoder.encoder.0.", "encoder.encoder.1."
        )
        _assert_frozen(loaded, "encoder.encoder.1.")


def test_frozen_molecular_ffn_copies_only_requested_leading_layers(tmp_path: Path):
    architecture = (
        "--hidden_size",
        "17",
        "--ffn_hidden_size",
        "13",
        "--ffn_num_layers",
        "3",
    )
    source_args = _args(*architecture)
    source = MoleculeModel(source_args)
    _fill_parameters(source)
    source_state = _clone_state(source)
    checkpoint_path = tmp_path / "molecular-ffn.pt"
    save_checkpoint(checkpoint_path, source, args=source_args)

    current_args = _args(*architecture)
    current_args.frzn_ffn_layers = 2
    torch.manual_seed(103)
    current = MoleculeModel(current_args)
    before_state = _clone_state(current)
    loaded = load_frzn_model(current, checkpoint_path, current_args=current_args)
    loaded_state = loaded.state_dict()

    parameters = dict(loaded.named_parameters())
    for name in (
        "readout.1.weight",
        "readout.1.bias",
        "readout.4.weight",
        "readout.4.bias",
    ):
        torch.testing.assert_close(loaded_state[name], source_state[name])
        assert parameters[name].requires_grad is False
    for name in ("readout.7.weight", "readout.7.bias"):
        torch.testing.assert_close(loaded_state[name], before_state[name])
        assert parameters[name].requires_grad is True


def _reaction_solvent_args() -> TrainArgs:
    args = _args(
        "--hidden_size",
        "17",
        "--hidden_size_solvent",
        "11",
        "--depth",
        "2",
        "--depth_solvent",
        "4",
    )
    args.reaction_solvent = True
    args.reaction = False
    args.number_of_molecules = 2
    args.smiles_columns = ["reaction", "solvent"]
    return args


def test_reaction_solvent_freeze_first_only_leaves_solvent_encoder_trainable(
    tmp_path: Path,
):
    source_args = _reaction_solvent_args()
    source = MoleculeModel(source_args)
    _fill_parameters(source)
    source_state = _clone_state(source)
    checkpoint_path = tmp_path / "reaction-solvent.pt"
    save_checkpoint(checkpoint_path, source, args=source_args)

    current_args = _reaction_solvent_args()
    current_args.freeze_first_only = True
    current_args.frzn_ffn_layers = 0
    torch.manual_seed(107)
    current = MoleculeModel(current_args)
    before_state = _clone_state(current)
    before_requires_grad = {
        name: parameter.requires_grad for name, parameter in current.named_parameters()
    }

    loaded = load_frzn_model(current, checkpoint_path, current_args=current_args)
    loaded_state = loaded.state_dict()

    _assert_prefix_copied(
        source_state, loaded_state, "encoder.encoder.", "encoder.encoder."
    )
    _assert_frozen(loaded, "encoder.encoder.")
    _assert_state_unchanged(before_state, loaded_state, "encoder.encoder_solvent.")
    for name, parameter in loaded.named_parameters():
        if name.startswith("encoder.encoder_solvent."):
            assert parameter.requires_grad == before_requires_grad[name]


def _atom_target_args(task_names) -> TrainArgs:
    args = _args(
        "--hidden_size",
        "17",
        "--ffn_hidden_size",
        "13",
        "--ffn_num_layers",
        "3",
        task_names=task_names,
    )
    args.is_atom_bond_targets = True
    args.atom_targets = list(task_names)
    args.bond_targets = []
    args.target_columns = list(task_names)
    args.atom_constraints = [False] * len(task_names)
    args.bond_constraints = []
    return args


def _atom_bond_target_args(atom_targets, bond_targets) -> TrainArgs:
    task_names = tuple(atom_targets) + tuple(bond_targets)
    args = _args(
        "--hidden_size",
        "17",
        "--ffn_hidden_size",
        "13",
        "--ffn_num_layers",
        "3",
        task_names=task_names,
    )
    args.is_atom_bond_targets = True
    args.atom_targets = list(atom_targets)
    args.bond_targets = list(bond_targets)
    args.target_columns = list(task_names)
    args.atom_constraints = [False] * len(atom_targets)
    args.bond_constraints = [False] * len(bond_targets)
    return args


def test_shared_atom_ffn_expands_frozen_base_to_new_task_alias(tmp_path: Path):
    source_args = _atom_target_args(("charge",))
    source = MoleculeModel(source_args)
    _fill_parameters(source)
    source_state = _clone_state(source)
    checkpoint_path = tmp_path / "one-atom-task.pt"
    save_checkpoint(checkpoint_path, source, args=source_args)

    current_args = _atom_target_args(("charge", "charge_plus_one"))
    current_args.frzn_ffn_layers = 1
    torch.manual_seed(109)
    current = MoleculeModel(current_args)
    before_state = _clone_state(current)

    loaded = load_frzn_model(current, checkpoint_path, current_args=current_args)
    loaded_state = loaded.state_dict()
    canonical_names = (
        "readout.atom_ffn_base.0.1.weight",
        "readout.atom_ffn_base.0.1.bias",
    )
    all_parameters = dict(loaded.named_parameters(remove_duplicate=False))

    for canonical_name in canonical_names:
        suffix = canonical_name.rsplit(".", 1)[-1]
        source_value = source_state[canonical_name]
        torch.testing.assert_close(loaded_state[canonical_name], source_value)
        assert all_parameters[canonical_name].requires_grad is False
        for task_index in (0, 1):
            alias = f"readout.ffn_list.{task_index}.ffn.0.1.{suffix}"
            torch.testing.assert_close(loaded_state[alias], source_value)
            assert all_parameters[alias].requires_grad is False

    for task_index in (0, 1):
        for suffix in ("weight", "bias"):
            head_name = f"readout.ffn_list.{task_index}.ffn_readout.1.{suffix}"
            torch.testing.assert_close(loaded_state[head_name], before_state[head_name])
            assert all_parameters[head_name].requires_grad is True


def test_shared_frozen_ffn_uses_task_kind_when_task_boundary_moves(tmp_path: Path):
    source_args = _atom_bond_target_args(
        ("atom_a", "atom_b"), ("bond_a",),
    )
    source = MoleculeModel(source_args)
    _fill_parameters(source)
    source_state = _clone_state(source)
    checkpoint_path = tmp_path / "shifted-task-boundary.pt"
    save_checkpoint(checkpoint_path, source, args=source_args)

    current_args = _atom_bond_target_args(
        ("atom_a",), ("bond_a", "bond_b"),
    )
    current_args.frzn_ffn_layers = 1
    current = MoleculeModel(current_args)

    loaded = load_frzn_model(current, checkpoint_path, current_args=current_args)
    loaded_state = loaded.state_dict()
    for task_index, task_kind in enumerate(("atom", "bond", "bond")):
        for suffix in ("weight", "bias"):
            alias = f"readout.ffn_list.{task_index}.ffn.0.1.{suffix}"
            canonical = f"readout.{task_kind}_ffn_base.0.1.{suffix}"
            torch.testing.assert_close(loaded_state[alias], source_state[canonical])
