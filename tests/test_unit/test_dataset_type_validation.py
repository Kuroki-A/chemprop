"""Validation tests for numerical training targets."""

import numpy as np
import pytest

from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data.utils import get_class_sizes, validate_dataset_type


def _dataset(*targets):
    return MoleculeDataset(
        [
            MoleculeDatapoint(smiles=["C"], targets=list(target_row))
            for target_row in targets
        ]
    )


@pytest.mark.parametrize("target", [float("nan"), float("inf"), -float("inf")])
def test_dataset_targets_must_be_finite(target):
    with pytest.raises(ValueError, match="finite"):
        validate_dataset_type(_dataset([target]), "regression")


@pytest.mark.parametrize("target", [-1, 1.5, 3])
def test_multiclass_targets_must_be_valid_indices(target):
    with pytest.raises(ValueError, match="Multiclass"):
        validate_dataset_type(
            _dataset([target]),
            "multiclass",
            multiclass_num_classes=3,
        )


def test_multiclass_targets_accept_missing_and_in_range_indices():
    validate_dataset_type(
        _dataset([0], [None], [2]),
        "multiclass",
        multiclass_num_classes=3,
    )


def test_multiclass_requires_at_least_two_classes():
    with pytest.raises(ValueError, match="at least 2"):
        validate_dataset_type(
            _dataset([0]),
            "multiclass",
            multiclass_num_classes=1,
        )


def test_atom_class_sizes_ignore_individually_missing_labels():
    dataset = MoleculeDataset(
        [
            MoleculeDatapoint(
                smiles=["CC"],
                targets=[np.asarray([0, None], dtype=object)],
                atom_targets=[np.asarray([0, None], dtype=object)],
            ),
            MoleculeDatapoint(
                smiles=["CO"],
                targets=[np.asarray([1, 1], dtype=object)],
                atom_targets=[np.asarray([1, 1], dtype=object)],
            ),
        ]
    )

    assert get_class_sizes(dataset, proportion=False) == [[1, 2]]


def test_class_sizes_reject_nonbinary_subset_without_both_binary_values():
    with pytest.raises(ValueError, match="only contain 0s and 1s"):
        get_class_sizes(_dataset([0], [2]))


@pytest.mark.parametrize('dataset_type', ['regression', 'classification', 'multiclass'])
def test_training_requires_an_observed_target(dataset_type):
    with pytest.raises(ValueError, match='observed target'):
        validate_dataset_type(_dataset([None], [None]), dataset_type)
