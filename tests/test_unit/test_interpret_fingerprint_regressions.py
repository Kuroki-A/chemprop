from collections import OrderedDict
import csv
import importlib
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from chemprop.data import get_data_from_smiles, MoleculeDatapoint, MoleculeDataset
from chemprop.features import get_features_generators_metadata
from chemprop.interpret import ChempropModel
from chemprop.train import molecule_fingerprint as fingerprint_module
from chemprop.train.molecule_fingerprint import (
    molecule_fingerprint,
    restore_checkpoint_featurization,
    validate_checkpoint_feature_schema,
)
from chemprop.train.make_predictions import load_model
from chemprop.utils import update_prediction_args


make_predictions_module = importlib.import_module('chemprop.train.make_predictions')


def _empty_feature_metadata(generated_dimension=0):
    return {
        'schema_version': 1,
        'external_features': [],
        'phase_features': None,
        'generated_dimension': generated_dimension,
        'total_dimension': generated_dimension,
    }


def test_restore_checkpoint_featurization_uses_checkpoint_values():
    train_args = SimpleNamespace(
        explicit_h=True,
        adding_h=True,
        keeping_atom_map=True,
        reaction=True,
        reaction_solvent=False,
        reaction_mode='reac_diff',
        atom_descriptors='feature',
        atom_features_size=7,
        bond_descriptors='feature',
        bond_features_size=5,
    )

    with mock.patch.multiple(
        fingerprint_module,
        reset_featurization_parameters=mock.DEFAULT,
        set_explicit_h=mock.DEFAULT,
        set_adding_hs=mock.DEFAULT,
        set_keeping_atom_map=mock.DEFAULT,
        set_reaction=mock.DEFAULT,
        set_extra_atom_fdim=mock.DEFAULT,
        set_extra_bond_fdim=mock.DEFAULT,
    ) as calls:
        restore_checkpoint_featurization(train_args)

    calls['reset_featurization_parameters'].assert_called_once_with()
    calls['set_explicit_h'].assert_called_once_with(True)
    calls['set_adding_hs'].assert_called_once_with(True)
    calls['set_keeping_atom_map'].assert_called_once_with(True)
    calls['set_reaction'].assert_called_once_with(True, 'reac_diff')
    calls['set_extra_atom_fdim'].assert_called_once_with(7)
    calls['set_extra_bond_fdim'].assert_called_once_with(5)


def test_feature_schema_validation_rejects_same_width_generator_change():
    expected_metadata = get_features_generators_metadata(
        ['morgan'], total_dimension=2048,
    )
    train_args = SimpleNamespace(
        features_generator=['morgan'],
        features_size=2048,
        features_generator_metadata=expected_metadata,
        features_source_metadata=_empty_feature_metadata(2048),
    )
    args = SimpleNamespace(
        features_generator=['fcfp'], selected_features_path=None,
    )
    data = MoleculeDataset([
        MoleculeDatapoint(smiles=['CCO'], features=np.zeros(2048)),
    ])
    data._features_source_metadata = _empty_feature_metadata(2048)

    with pytest.raises(ValueError, match='Feature generators do not match'):
        validate_checkpoint_feature_schema(args, train_args, data, data)


def test_feature_schema_all_invalid_uses_generator_metadata_not_placeholder_width():
    data = get_data_from_smiles(
        [['not-a-smiles']],
        skip_invalid_smiles=False,
        features_generator=['morgan'],
    )
    valid_data = MoleculeDataset([])
    train_args = SimpleNamespace(
        features_generator=['morgan'],
        features_size=2048,
        features_generator_metadata=get_features_generators_metadata(
            ['morgan'], total_dimension=2048,
        ),
        features_source_metadata=data._features_source_metadata,
    )
    args = SimpleNamespace(
        features_generator=['morgan'], selected_features_path=None,
    )

    validate_checkpoint_feature_schema(args, train_args, data, valid_data)


def _fingerprint_args(tmp_path, output_name):
    return SimpleNamespace(
        checkpoint_paths=['model.pt'],
        fingerprint_type='MPN',
        features_generator=None,
        selected_features_path=None,
        features_path=None,
        features_only=False,
        atom_descriptors=None,
        bond_descriptors=None,
        hidden_size=2,
        hidden_size_solvent=2,
        reaction_solvent=False,
        number_of_molecules=1,
        batch_size=4,
        num_workers=0,
        features_scaling=False,
        atom_descriptor_scaling=False,
        bond_descriptor_scaling=False,
        smiles_columns=['smiles'],
        preds_path=str(tmp_path / output_name),
        device='cpu',
    )


def _fingerprint_train_args():
    return SimpleNamespace(
        features_generator=None,
        features_path=None,
        features_size=None,
        features_generator_metadata=None,
        features_source_metadata=None,
        atom_descriptors=None,
        atom_descriptors_size=0,
        atom_features_size=0,
        bond_descriptors=None,
        bond_descriptors_size=0,
        bond_features_size=0,
    )


def _prediction_feature_args(features_generator, features_path):
    return SimpleNamespace(
        number_of_molecules=1,
        features_scaling=False,
        atom_descriptors=None,
        bond_descriptors=None,
        constraints_path=None,
        features_generator=features_generator,
        features_path=features_path,
    )


def test_legacy_prediction_args_reject_reordered_feature_generators():
    train_args = _prediction_feature_args(
        ['morgan', 'rdkit_2d'], ['training_features.npz'],
    )
    predict_args = _prediction_feature_args(
        ['rdkit_2d', 'morgan'], ['prediction_features.npz'],
    )

    with pytest.raises(ValueError, match='ordered feature generators'):
        update_prediction_args(
            predict_args,
            train_args,
            missing_to_defaults=False,
        )


def test_legacy_prediction_args_compare_feature_paths_by_presence_only():
    train_args = _prediction_feature_args(['morgan'], ['training_features.npz'])
    predict_args = _prediction_feature_args(['morgan'], ['prediction_features.npz'])

    update_prediction_args(
        predict_args,
        train_args,
        missing_to_defaults=False,
    )


@pytest.mark.parametrize(
    ('compatibility_field', 'first_value', 'second_value'),
    [
        ('features_scaling', False, True),
        ('atom_descriptor_scaling', False, True),
        ('bond_descriptor_scaling', False, True),
        ('atom_constraints', [True, False], [False, True]),
        ('bond_constraints', [True], [False]),
        ('weights_ffn_num_layers', 2, 3),
        (
            'spectra_phase_mask',
            np.array([[True, False], [False, True]]),
            np.array([[False, True], [True, False]]),
        ),
        ('quantile_loss_alpha', 0.1, 0.2),
        ('quantiles', [0.05, 0.95], [0.1, 0.9]),
    ],
)
@pytest.mark.parametrize('entrypoint', ['predict', 'fingerprint', 'interpret'])
def test_ffn_ensemble_rejects_incompatible_fields_before_model_loading(
    compatibility_field, first_value, second_value, entrypoint,
):
    first = SimpleNamespace(**{compatibility_field: first_value})
    second = SimpleNamespace(**{compatibility_field: second_value})
    args = SimpleNamespace(checkpoint_paths=['model_0.pt', 'model_1.pt'])

    if entrypoint == 'predict':
        with mock.patch.object(
            make_predictions_module, 'load_args', side_effect=[first, second],
        ), mock.patch.object(
            make_predictions_module, 'load_checkpoint',
        ) as load_checkpoint_mock, pytest.raises(ValueError, match=compatibility_field):
            load_model(args)
    elif entrypoint == 'fingerprint':
        with mock.patch.object(
            fingerprint_module, 'load_args', side_effect=[first, second],
        ), mock.patch.object(
            fingerprint_module, 'load_checkpoint',
        ) as load_checkpoint_mock, pytest.raises(ValueError, match=compatibility_field):
            molecule_fingerprint(args, smiles=[])
    else:
        with mock.patch(
            'chemprop.interpret.load_args', side_effect=[first, second],
        ), mock.patch(
            'chemprop.interpret.load_checkpoint',
        ) as load_checkpoint_mock, pytest.raises(ValueError, match=compatibility_field):
            ChempropModel(args)

    load_checkpoint_mock.assert_not_called()


@pytest.mark.parametrize('all_invalid', [False, True])
def test_fingerprint_zero_valid_rows_still_writes_csv(tmp_path, all_invalid):
    args = _fingerprint_args(tmp_path, 'fingerprints.csv')
    train_args = _fingerprint_train_args()
    if all_invalid:
        args.features_generator = ['morgan']
        full_data = get_data_from_smiles(
            [['not-a-smiles']],
            skip_invalid_smiles=False,
            features_generator=args.features_generator,
        )
        full_data[0].row = OrderedDict([('smiles', 'not-a-smiles')])
        train_args.features_generator = ['morgan']
        train_args.features_size = 2048
        train_args.features_generator_metadata = get_features_generators_metadata(
            ['morgan'], total_dimension=2048,
        )
        train_args.features_source_metadata = full_data._features_source_metadata
    else:
        full_data = MoleculeDataset([])
        full_data._features_source_metadata = _empty_feature_metadata(0)

    with mock.patch.object(
        fingerprint_module, 'load_args', return_value=train_args,
    ), mock.patch.object(
        fingerprint_module, 'update_prediction_args', return_value=None,
    ), mock.patch.object(
        fingerprint_module, 'restore_checkpoint_featurization', return_value=None,
    ), mock.patch.object(
        fingerprint_module, 'get_data_from_smiles', return_value=full_data,
    ):
        result = molecule_fingerprint(args, smiles=[])

    lines = (tmp_path / 'fingerprints.csv').read_text().splitlines()
    assert lines[0] == 'smiles,fp_0_mol_0,fp_1_mol_0'
    assert result.shape == (len(full_data), 2, 1)
    if all_invalid:
        assert lines[1] == 'not-a-smiles,Invalid SMILES,Invalid SMILES'
        assert np.all(result == 'Invalid SMILES')
    else:
        assert len(lines) == 1


def test_mpn_fingerprint_truncates_phase_only_input_features(tmp_path):
    args = _fingerprint_args(tmp_path, 'phase_fingerprints.csv')
    args.phase_features_path = 'prediction_phase_features.csv'
    train_args = _fingerprint_train_args()
    train_args.phase_features_path = 'training_phase_features.csv'
    train_args.atom_descriptor_scaling = False
    train_args.bond_descriptor_scaling = False
    full_data = MoleculeDataset([
        MoleculeDatapoint(
            smiles=['CCO'],
            row=OrderedDict([('smiles', 'CCO')]),
            phase_features=np.array([1.0]),
        ),
    ])

    with mock.patch.object(
        fingerprint_module, 'load_args', return_value=train_args,
    ), mock.patch.object(
        fingerprint_module, 'update_prediction_args', return_value=None,
    ), mock.patch.object(
        fingerprint_module, 'restore_checkpoint_featurization', return_value=None,
    ), mock.patch.object(
        fingerprint_module, 'get_data_from_smiles', return_value=full_data,
    ), mock.patch.object(
        fingerprint_module, 'validate_checkpoint_feature_schema', return_value=None,
    ), mock.patch.object(
        fingerprint_module, 'load_checkpoint', return_value=object(),
    ), mock.patch.object(
        fingerprint_module, 'load_scalers', return_value=(None,) * 5,
    ), mock.patch.object(
        fingerprint_module,
        'model_fingerprint',
        return_value=[[10.0, 11.0, 99.0]],
    ):
        result = molecule_fingerprint(args, smiles=[['CCO']])

    assert result.shape == (1, 2, 1)
    np.testing.assert_allclose(result.astype(float), [[[10.0], [11.0]]])


def test_multimolecule_ensemble_csv_columns_follow_c_order_flatten(tmp_path):
    args = _fingerprint_args(tmp_path, 'ensemble_fingerprints.csv')
    args.checkpoint_paths = ['model_0.pt', 'model_1.pt']
    args.number_of_molecules = 2
    args.smiles_columns = ['solute', 'solvent']
    train_args = _fingerprint_train_args()
    train_args.atom_descriptor_scaling = False
    train_args.bond_descriptor_scaling = False
    full_data = MoleculeDataset([
        MoleculeDatapoint(
            smiles=['CCO', 'O'],
            row=OrderedDict([('solute', 'CCO'), ('solvent', 'O')]),
        ),
    ])

    with mock.patch.object(
        fingerprint_module, 'load_args', return_value=train_args,
    ), mock.patch.object(
        fingerprint_module, 'update_prediction_args', return_value=None,
    ), mock.patch.object(
        fingerprint_module, 'restore_checkpoint_featurization', return_value=None,
    ), mock.patch.object(
        fingerprint_module, 'get_data_from_smiles', return_value=full_data,
    ), mock.patch.object(
        fingerprint_module, 'validate_checkpoint_feature_schema', return_value=None,
    ), mock.patch.object(
        fingerprint_module, 'load_checkpoint', side_effect=[object(), object()],
    ), mock.patch.object(
        fingerprint_module, 'load_scalers', return_value=(None,) * 5,
    ), mock.patch.object(
        fingerprint_module,
        'model_fingerprint',
        side_effect=[
            [[100.0, 110.0, 200.0, 210.0]],
            [[101.0, 111.0, 201.0, 211.0]],
        ],
    ):
        result = molecule_fingerprint(args, smiles=[['CCO', 'O']])

    expected_columns = [
        'fp_0_mol_0_model_0',
        'fp_0_mol_0_model_1',
        'fp_1_mol_0_model_0',
        'fp_1_mol_0_model_1',
        'fp_0_mol_1_model_0',
        'fp_0_mol_1_model_1',
        'fp_1_mol_1_model_0',
        'fp_1_mol_1_model_1',
    ]
    with open(args.preds_path, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        row = next(reader)

    assert reader.fieldnames == args.smiles_columns + expected_columns
    assert [float(row[column]) for column in expected_columns] == [
        100.0, 101.0, 110.0, 111.0, 200.0, 201.0, 210.0, 211.0,
    ]
    np.testing.assert_allclose(
        result.astype(float).reshape(-1),
        [100.0, 101.0, 110.0, 111.0, 200.0, 201.0, 210.0, 211.0],
    )


def test_interpret_restores_settings_and_validates_generated_data():
    args = SimpleNamespace(
        checkpoint_paths=['model.pt'],
        features_generator=None,
        selected_features_path=None,
        atom_descriptors=None,
        bond_descriptors_size=0,
        device='cpu',
        num_workers=0,
    )
    train_args = _fingerprint_train_args()
    train_args.features_scaling = False
    train_args.atom_descriptor_scaling = False
    train_args.bond_descriptor_scaling = False
    data = MoleculeDataset([
        MoleculeDatapoint(smiles=['CCO'], row=OrderedDict([('smiles', 'CCO')])),
    ])
    data._features_source_metadata = _empty_feature_metadata(0)

    with mock.patch('chemprop.interpret.load_args', return_value=train_args), \
            mock.patch('chemprop.interpret.restore_checkpoint_featurization') as restore, \
            mock.patch('chemprop.interpret.load_scalers', return_value=(None,) * 5), \
            mock.patch('chemprop.interpret.load_checkpoint', return_value=object()), \
            mock.patch('chemprop.interpret.get_data_from_smiles', return_value=data), \
            mock.patch('chemprop.interpret.validate_checkpoint_feature_schema') as validate, \
            mock.patch('chemprop.interpret.predict', return_value=[[1.25]]):
        model = ChempropModel(args)
        predictions = model([['CCO']])

    restore.assert_called_once_with(train_args)
    validate.assert_called_once()
    np.testing.assert_allclose(predictions, [[1.25]])


def test_interpret_uses_each_ensemble_members_target_scaler():
    args = SimpleNamespace(
        checkpoint_paths=['model_0.pt', 'model_1.pt'],
        features_generator=None,
        selected_features_path=None,
        atom_descriptors=None,
        bond_descriptors_size=0,
        device='cpu',
        num_workers=0,
    )
    train_args = _fingerprint_train_args()
    train_args.features_scaling = False
    train_args.atom_descriptor_scaling = False
    train_args.bond_descriptor_scaling = False
    data = MoleculeDataset([
        MoleculeDatapoint(smiles=['CCO'], row=OrderedDict([('smiles', 'CCO')])),
    ])
    data._features_source_metadata = _empty_feature_metadata(0)
    target_scalers = [object(), object()]

    with mock.patch('chemprop.interpret.load_args', return_value=train_args), \
            mock.patch('chemprop.interpret.restore_checkpoint_featurization'), \
            mock.patch(
                'chemprop.interpret.load_scalers',
                side_effect=[
                    (target_scalers[0], None, None, None, None),
                    (target_scalers[1], None, None, None, None),
                ],
            ), \
            mock.patch(
                'chemprop.interpret.load_checkpoint',
                side_effect=[object(), object()],
            ), \
            mock.patch('chemprop.interpret.get_data_from_smiles', return_value=data), \
            mock.patch('chemprop.interpret.validate_checkpoint_feature_schema'), \
            mock.patch(
                'chemprop.interpret.predict',
                side_effect=[[[1.0]], [[3.0]]],
            ) as predict_mock:
        model = ChempropModel(args)
        predictions = model([['CCO']])

    assert [call.kwargs['scaler'] for call in predict_mock.call_args_list] == target_scalers
    np.testing.assert_allclose(predictions, [[2.0]])
