from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import chemprop.utils as utils_module
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.utils import build_lr_scheduler, load_checkpoint, save_smiles_splits


def test_lr_scheduler_uses_nonzero_ceiling_batch_count():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter])
    args = SimpleNamespace(
        warmup_epochs=2,
        epochs=30,
        num_lrs=1,
        train_data_size=51,
        batch_size=50,
        init_lr=1e-4,
        max_lr=1e-3,
        final_lr=1e-4,
    )

    scheduler = build_lr_scheduler(optimizer, args)
    assert scheduler.steps_per_epoch == 2

    args.train_data_size = 1
    scheduler = build_lr_scheduler(optimizer, args)
    assert scheduler.steps_per_epoch == 1


def test_checkpoint_load_rejects_missing_model_parameters(monkeypatch):
    """Normal prediction must never retain random weights silently."""
    checkpoint = {
        "args": Namespace(data_path="unused.csv", dataset_type="regression"),
        "state_dict": {"weight": torch.ones(1, 2)},
    }
    monkeypatch.setattr(utils_module.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(
        utils_module,
        "MoleculeModel",
        lambda args: torch.nn.Linear(2, 1),
    )

    with pytest.raises(ValueError, match="missing parameters=.*bias"):
        load_checkpoint("incomplete.pt", device=torch.device("cpu"))


def test_checkpoint_partial_load_remains_explicitly_opt_in(monkeypatch):
    checkpoint = {
        "args": Namespace(data_path="unused.csv", dataset_type="regression"),
        "state_dict": {"weight": torch.ones(1, 2)},
    }
    monkeypatch.setattr(utils_module.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(
        utils_module,
        "MoleculeModel",
        lambda args: torch.nn.Linear(2, 1),
    )

    model = load_checkpoint(
        "partial.pt", device=torch.device("cpu"), strict=False
    )

    assert torch.equal(model.weight, torch.ones_like(model.weight))
    assert torch.isfinite(model.bias).all()


def test_save_smiles_splits_removes_stale_unrepresentable_indices(tmp_path: Path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCC,2\n', encoding='utf-8')
    split_path = tmp_path / 'split_indices.pckl'
    split_path.write_bytes(b'stale')
    train_data = MoleculeDataset([
        MoleculeDatapoint(smiles=['CC'], targets=[1.0]),
    ])

    save_smiles_splits(
        data_path=str(data_path),
        save_dir=str(tmp_path),
        task_names=['target'],
        train_data=train_data,
        smiles_columns=['smiles'],
    )

    assert not split_path.exists()
    assert not list(tmp_path.glob('.split-indices-*.tmp'))
