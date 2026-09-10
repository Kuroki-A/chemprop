from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chemprop.train.loss_functions import mcc_class_loss
from chemprop.train.run_training import (
    _validate_primary_validation_score,
    _validate_training_split,
)
from chemprop.train.train import train


class _Batch:
    def __init__(self, targets, masks, atom_bond=False):
        self._targets = targets
        self._masks = masks
        self._atom_bond = atom_bond
        self.number_of_atoms = [len(targets[0][0])] if atom_bond else [1]
        self.number_of_bonds = [0]

    def __len__(self):
        return 1

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
        return [[None]] if self._atom_bond else None

    def data_weights(self):
        return [1.0]

    def atom_bond_data_weights(self):
        return [[1.0] * len(self._targets[0][0])]


class _ScalarModel(torch.nn.Module):
    is_atom_bond_targets = False

    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, *args):
        return self.value.reshape(1, 1)


class _AtomClassificationModel(torch.nn.Module):
    is_atom_bond_targets = True

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, *args):
        return [torch.sigmoid(self.logits).reshape(-1, 1)]


class _Split:
    def __init__(self, targets):
        self._targets = targets

    def __len__(self):
        return len(self._targets)

    def __iter__(self):
        return iter(SimpleNamespace(targets=targets) for targets in self._targets)

    def mask(self):
        return list(zip(*[
            [target is not None for target in targets]
            for targets in self._targets
        ]))


def _args(**overrides):
    values = dict(
        device=torch.device("cpu"),
        dataset_type="regression",
        loss_function="mse",
        target_weights=None,
        atom_targets=[],
        bond_targets=[],
        atom_constraints=[],
        bond_constraints=[],
        adding_bond_types=False,
        grad_clip=None,
        batch_size=1,
        log_frequency=100,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_train_skips_an_entirely_unlabeled_batch():
    model = _ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = _Batch(targets=[[None]], masks=[(False,)])

    n_iter = train(
        model=model,
        data_loader=[batch],
        loss_func=torch.nn.MSELoss(reduction="none"),
        optimizer=optimizer,
        scheduler=object(),
        args=_args(),
    )

    assert n_iter == 1
    assert model.value.grad is None
    assert model.value.item() == 0.5


def test_train_fails_before_updating_on_a_nonfinite_loss():
    model = _ScalarModel()
    model.value.data.fill_(float("nan"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = _Batch(targets=[[1.0]], masks=[(True,)])

    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train(
            model=model,
            data_loader=[batch],
            loss_func=torch.nn.MSELoss(reduction="none"),
            optimizer=optimizer,
            scheduler=object(),
            args=_args(),
        )


def test_atom_mcc_is_not_divided_by_the_number_of_labels():
    logits = torch.tensor([-1.5, -0.2, 0.4, 1.1])
    targets = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    weights = torch.ones_like(targets)

    expected_logits = logits.clone().requires_grad_(True)
    expected_loss = mcc_class_loss(
        torch.sigmoid(expected_logits).reshape(-1, 1), targets, weights, mask
    ).sum()
    expected_loss.backward()

    model = _AtomClassificationModel(logits)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    batch = _Batch(
        targets=[[np.asarray([0.0, 0.0, 1.0, 1.0], dtype=object)]],
        masks=[[True, True, True, True]],
        atom_bond=True,
    )
    train(
        model=model,
        data_loader=[batch],
        loss_func=mcc_class_loss,
        optimizer=optimizer,
        scheduler=object(),
        args=_args(
            dataset_type="classification",
            loss_function="mcc",
            atom_targets=["atom_target"],
            atom_constraints=[False],
        ),
    )

    torch.testing.assert_close(model.logits.grad, expected_logits.grad)


def test_training_split_rejects_a_completely_unlabeled_task():
    args = SimpleNamespace(
        num_tasks=2,
        task_names=["measured", "never_measured"],
        class_balance=False,
    )

    with pytest.raises(ValueError, match="never_measured.*randomly initialized"):
        _validate_training_split(
            args,
            _Split([[1.0, None], [2.0, None]]),
            _Split([[3.0, 4.0]]),
        )


def test_training_split_allows_legacy_multitask_class_balance():
    _validate_training_split(
        SimpleNamespace(
            num_tasks=2,
            task_names=["a", "b"],
            class_balance=True,
        ),
        _Split([[0, 1], [0, 0]]),
        _Split([[0, 1]]),
    )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_primary_validation_score_is_never_accepted(score):
    with pytest.raises(ValueError, match="cannot select a trained checkpoint"):
        _validate_primary_validation_score("auc", score)
