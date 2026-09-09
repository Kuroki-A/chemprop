"""Regression tests for row-level constraint input validation."""

import numpy as np
import pytest
import torch

from chemprop.data import get_constraints
from chemprop.train.predict import predict


def test_get_constraints_rejects_duplicate_raw_csv_headers(tmp_path):
    path = tmp_path / 'duplicate.csv'
    path.write_text('task,task\n1,2\n')

    with pytest.raises(ValueError, match='duplicate columns'):
        get_constraints(str(path), ['task'])


@pytest.mark.parametrize('invalid_value', ['not-a-number', '', 'nan', 'inf', '-inf'])
def test_get_constraints_rejects_nonfinite_or_nonnumeric_values(
    tmp_path, invalid_value,
):
    path = tmp_path / 'invalid.csv'
    path.write_text(f'task,other\n1,first\n{invalid_value},second\n')

    with pytest.raises(ValueError, match='numeric finite values'):
        get_constraints(str(path), ['task'])


def test_get_constraints_keeps_missing_target_columns_as_unconstrained(tmp_path):
    path = tmp_path / 'constraints.csv'
    path.write_text('present\n1.5\n2.5\n')

    constraints, raw = get_constraints(
        str(path), ['missing', 'present'], save_raw_data=True,
    )

    assert constraints.shape == (2, 2)
    assert constraints[:, 0].tolist() == [None, None]
    np.testing.assert_array_equal(
        constraints[:, 1].astype(float), np.array([1.5, 2.5])
    )
    np.testing.assert_array_equal(raw, np.array([[1.5], [2.5]]))


class _ConstraintBatch:
    def __init__(self, constraints):
        self._constraints = constraints
        self.number_of_atoms = [[2], [1]]
        self.number_of_bonds = [[1], [0]]

    def __len__(self):
        return 2

    def __iter__(self):
        return iter(())

    def batch_graph(self):
        return None

    def features(self):
        return None

    def atom_descriptors(self):
        return None

    def atom_features(self):
        return None

    def bond_descriptors(self):
        return None

    def bond_features(self):
        return None

    def constraints(self):
        return self._constraints


class _ConstraintCaptureModel(torch.nn.Module):
    def __init__(
        self,
        atom_targets,
        bond_targets,
        atom_constraints,
        bond_constraints,
    ):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.is_atom_bond_targets = True
        self.atom_targets = atom_targets
        self.bond_targets = bond_targets
        self.atom_constraints = atom_constraints
        self.bond_constraints = bond_constraints
        self.adding_bond_types = False
        self.loss_function = 'mse'
        self.classification = False
        self.multiclass = False
        self.received_constraints = None

    def forward(self, *inputs):
        self.received_constraints = inputs[6]
        outputs = []
        outputs.extend(
            torch.zeros((3, 1), device=self.anchor.device)
            for _ in self.atom_targets
        )
        outputs.extend(
            torch.zeros((1, 1), device=self.anchor.device)
            for _ in self.bond_targets
        )
        return outputs


def _identity_atom_bond_scaler(num_tasks):
    return type(
        'IdentityAtomBondScaler',
        (),
        {
            'means': [np.array([0.0])] * num_tasks,
            'stds': [np.array([1.0])] * num_tasks,
            'inverse_transform': lambda self, values: values,
        },
    )()


def test_prediction_empty_constraint_matrix_becomes_task_major_none_values():
    model = _ConstraintCaptureModel(
        atom_targets=['atom'],
        bond_targets=['bond'],
        atom_constraints=[False],
        bond_constraints=[False],
    )
    batch = _ConstraintBatch(np.empty((2, 0), dtype=object))

    predict(model, [batch], disable_progress_bar=True)

    assert model.received_constraints == [None, None]


def test_prediction_transposes_row_major_constraints_to_task_major_tensors():
    model = _ConstraintCaptureModel(
        atom_targets=['atom'],
        bond_targets=['bond'],
        atom_constraints=[True],
        bond_constraints=[True],
    )
    batch = _ConstraintBatch(
        np.array([[10.0, 20.0], [30.0, 40.0]], dtype=object)
    )

    predict(
        model,
        [batch],
        disable_progress_bar=True,
        atom_bond_scaler=_identity_atom_bond_scaler(2),
    )

    assert len(model.received_constraints) == 2
    torch.testing.assert_close(
        model.received_constraints[0], torch.tensor([10.0, 30.0])
    )
    torch.testing.assert_close(
        model.received_constraints[1], torch.tensor([20.0, 40.0])
    )


@pytest.mark.parametrize(
    'constraints',
    [
        np.empty((2, 0), dtype=object),
        np.array([['not-numeric'], ['1.0']], dtype=object),
        np.array([[np.nan], [1.0]], dtype=object),
    ],
    ids=['missing-task', 'non-numeric', 'non-finite'],
)
def test_required_prediction_constraints_fail_with_the_checkpoint_task_name(
    constraints,
):
    model = _ConstraintCaptureModel(
        atom_targets=['partial_charge'],
        bond_targets=[],
        atom_constraints=[True],
        bond_constraints=[],
    )

    with pytest.raises(
        ValueError, match=r"required atom task 'partial_charge'.*numeric finite",
    ):
        predict(
            model,
            [_ConstraintBatch(constraints)],
            disable_progress_bar=True,
            atom_bond_scaler=_identity_atom_bond_scaler(1),
        )

    assert model.received_constraints is None
