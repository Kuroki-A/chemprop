"""Regression coverage for externally supplied split indices."""

import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from chemprop.data import MoleculeDatapoint, MoleculeDataset, split_data


def _dataset():
    return MoleculeDataset(
        [
            MoleculeDatapoint([smiles], targets=[float(index)])
            for index, smiles in enumerate(['C', 'CC', 'CCC', 'CO', 'CN', 'CF'])
        ]
    )


def _index_args(index_sets):
    return SimpleNamespace(
        folds_file=None,
        val_fold_index=None,
        test_fold_index=None,
        crossval_index_sets=[index_sets],
        seed=0,
    )


def _predetermined_args(folds_file, test_fold_index, val_fold_index):
    return SimpleNamespace(
        folds_file=str(folds_file),
        val_fold_index=val_fold_index,
        test_fold_index=test_fold_index,
    )


def _write_folds(tmp_path, folds):
    path = tmp_path / 'folds.pkl'
    with path.open('wb') as file:
        pickle.dump(folds, file)
    return path


def test_index_predetermined_rejects_overlap_between_external_splits():
    args = _index_args([[0, 1], [1, 2], [3, 4, 5]])

    with pytest.raises(ValueError, match='training and validation splits overlap'):
        split_data(_dataset(), split_type='index_predetermined', args=args)


@pytest.mark.parametrize('invalid_index', [-1, 6])
def test_index_predetermined_rejects_out_of_range_data_indices(invalid_index):
    args = _index_args([[0, 1], [2, 3], [4, invalid_index]])

    with pytest.raises(ValueError, match='outside the valid range'):
        split_data(_dataset(), split_type='index_predetermined', args=args)


def test_predetermined_rejects_overlapping_fold_files(tmp_path):
    folds_file = _write_folds(tmp_path, [[0, 1], [1, 2], [3, 4, 5]])
    args = _predetermined_args(folds_file, test_fold_index=1, val_fold_index=0)

    with pytest.raises(ValueError, match='fold 0 and fold 1 splits overlap'):
        split_data(_dataset(), split_type='predetermined', args=args)


@pytest.mark.parametrize(
    ('test_fold_index', 'val_fold_index', 'name'),
    [(-1, 0, 'test_fold_index'), (1, -1, 'val_fold_index')],
)
def test_predetermined_rejects_negative_fold_selectors(
    tmp_path, test_fold_index, val_fold_index, name,
):
    folds_file = _write_folds(tmp_path, [[0, 1], [2, 3], [4, 5]])
    args = _predetermined_args(
        folds_file,
        test_fold_index=test_fold_index,
        val_fold_index=val_fold_index,
    )

    with pytest.raises(ValueError, match=rf'{name}=-1.*outside'):
        split_data(_dataset(), split_type='predetermined', args=args)


def test_predetermined_accepts_zero_as_an_explicit_validation_fold(tmp_path):
    folds_file = _write_folds(tmp_path, [[0, 1], [2, 3], [4, 5]])
    args = _predetermined_args(folds_file, test_fold_index=1, val_fold_index=0)

    train, validation, test = split_data(
        _dataset(), split_type='predetermined', args=args,
    )

    assert train.smiles() == [['CN'], ['CF']]
    assert validation.smiles() == [['C'], ['CC']]
    assert test.smiles() == [['CCC'], ['CO']]


def test_crossval_warns_when_external_folds_omit_rows(tmp_path):
    for fold_id, indices in enumerate(([0, 1], [2], [3])):
        with (tmp_path / f'{fold_id}.pkl').open('wb') as file:
            pickle.dump(indices, file)
    args = SimpleNamespace(
        folds_file=None,
        val_fold_index=None,
        test_fold_index=None,
        crossval_index_sets=[[[0], [1], [2]]],
        crossval_index_dir=str(tmp_path),
        seed=0,
        skip_test_evaluation=False,
    )
    logger = Mock()

    split_data(_dataset(), split_type='crossval', args=args, logger=logger)

    logger.warning.assert_called_once_with(
        'External split indices omit 2 of 6 data rows; those rows will not be used.'
    )
