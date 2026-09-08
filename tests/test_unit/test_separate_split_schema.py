from copy import deepcopy
import importlib
from types import SimpleNamespace

import pytest

from chemprop.data import MoleculeDataset
from chemprop.train.run_training import validate_features_source_metadata


def _feature_metadata():
    return {
        'schema_version': 1,
        'external_features': [{
            'dimension': 2,
            'dtype': 'float32',
            'csv_header': ['descriptor_a', 'descriptor_b'],
            'feature_manifest': {
                'schema_version': 1,
                'generator': 'morgan',
                'dimension': 2,
                'dtype': 'float32',
                'feature_names': ['descriptor_a', 'descriptor_b'],
            },
        }],
        'phase_features': None,
        'generated_dimension': 0,
        'total_dimension': 2,
    }


def _empty_dataset(metadata=None):
    data = MoleculeDataset([])
    data._features_source_metadata = (
        _feature_metadata() if metadata is None else metadata
    )
    return data


def _separate_args():
    return SimpleNamespace(
        pytorch_seed=0,
        seed=0,
        task_names=['task_b', 'task_a'],
        separate_test_path='test.csv',
        separate_test_features_path=None,
        separate_test_atom_descriptors_path=None,
        separate_test_bond_descriptors_path=None,
        separate_test_phase_features_path=None,
        separate_test_constraints_path=None,
        separate_val_path='val.csv',
        separate_val_features_path=None,
        separate_val_atom_descriptors_path=None,
        separate_val_bond_descriptors_path=None,
        separate_val_phase_features_path=None,
        separate_val_constraints_path=None,
        smiles_columns=['smiles'],
        loss_function='mse',
        skip_test_evaluation=False,
        dataset_type='regression',
        save_smiles_splits=False,
        features_scaling=False,
        atom_descriptor_scaling=False,
        bond_descriptor_scaling=False,
    )


@pytest.mark.parametrize(
    ('mutate', 'difference_path'),
    [
        (
            lambda metadata: metadata['external_features'][0].update(
                csv_header=['descriptor_b', 'descriptor_a']
            ),
            'csv_header[0]',
        ),
        (
            lambda metadata: metadata['external_features'][0][
                'feature_manifest'
            ].update(generator='maccs'),
            'feature_manifest.generator',
        ),
    ],
)
def test_feature_schema_rejects_same_width_semantic_mismatch(
    mutate, difference_path
):
    expected = _feature_metadata()
    actual = deepcopy(expected)
    mutate(actual)

    with pytest.raises(ValueError, match=difference_path.replace('[', r'\[')):
        validate_features_source_metadata(
            _empty_dataset(expected),
            _empty_dataset(actual),
            'separate validation data',
        )


def test_ffn_separate_splits_request_training_task_order(monkeypatch):
    run_training_module = importlib.import_module('chemprop.train.run_training')
    calls = []

    def fake_get_data(**kwargs):
        calls.append(kwargs)
        return _empty_dataset()

    monkeypatch.setattr(run_training_module, 'get_data', fake_get_data)

    with pytest.raises(ValueError, match='validation data split is empty'):
        run_training_module.run_training(
            _separate_args(), _empty_dataset(), fold_num=0
        )

    assert [call['path'] for call in calls] == ['test.csv', 'val.csv']
    assert all(
        call['target_columns'] == ['task_b', 'task_a'] for call in calls
    )


def test_lgbm_separate_splits_request_training_task_order(monkeypatch):
    pytest.importorskip('lightgbm')
    lgbm_module = importlib.import_module('chemprop.train.run_training_lgbm')
    calls = []

    def fake_get_data(**kwargs):
        calls.append(kwargs)
        return _empty_dataset()

    monkeypatch.setattr(lgbm_module, 'get_data', fake_get_data)
    train_data, val_data, test_data = lgbm_module._load_split_data(
        _separate_args(), _empty_dataset(), logger=None
    )

    assert len(train_data) == len(val_data) == len(test_data) == 0
    assert [call['path'] for call in calls] == ['test.csv', 'val.csv']
    assert all(
        call['target_columns'] == ['task_b', 'task_a'] for call in calls
    )
