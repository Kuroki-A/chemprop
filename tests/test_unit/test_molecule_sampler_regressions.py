"""Regression tests for class-balanced molecule sampling."""

import pytest

from chemprop.data import MoleculeDatapoint, MoleculeDataset, MoleculeSampler


def _dataset(target_rows):
    return MoleculeDataset([
        MoleculeDatapoint(['C'], targets=targets) for targets in target_rows
    ])


def test_class_balance_excludes_missing_labels_instead_of_treating_as_negative():
    sampler = MoleculeSampler(
        _dataset([[1], [None], [0], [None]]), class_balance=True,
    )

    assert list(sampler) == [0, 2]
    assert len(sampler) == 2


def test_class_balance_rejects_multitask_targets():
    with pytest.raises(ValueError, match='single-task binary'):
        MoleculeSampler(_dataset([[1, 0], [0, 1]]), class_balance=True)


@pytest.mark.parametrize('target_rows', [[[1], [1], [None]], [[0], [None]]])
def test_class_balance_requires_both_observed_classes(target_rows):
    with pytest.raises(ValueError, match='each binary class'):
        MoleculeSampler(_dataset(target_rows), class_balance=True)


def test_class_balance_rejects_nonbinary_observed_targets():
    with pytest.raises(ValueError, match='binary values 0 or 1'):
        MoleculeSampler(_dataset([[0], [1], [2]]), class_balance=True)
