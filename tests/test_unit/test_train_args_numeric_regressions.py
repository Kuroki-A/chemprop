import json
from pathlib import Path

import pytest

from chemprop.args import HyperoptArgs, InterpretArgs, TrainArgs


DATA_PATH = Path(__file__).parents[1] / 'data' / 'regression.csv'


def _parse_train_args(*extra_args):
    return TrainArgs().parse_args([
        '--data_path', str(DATA_PATH),
        '--dataset_type', 'regression',
        *extra_args,
    ])


@pytest.mark.parametrize(
    ('flags', 'message'),
    [
        (['--hidden_size', '0'], 'hidden_size'),
        (['--depth', '0'], 'depth'),
        (['--ffn_hidden_size', '0'], 'ffn_hidden_size'),
        (['--ffn_num_layers', '0'], 'ffn_num_layers'),
        (['--log_frequency', '0'], 'log_frequency'),
        (['--multiclass_num_classes', '1'], 'multiclass_num_classes'),
        (['--dropout', 'nan'], 'dropout'),
        (['--dropout', '1'], 'dropout'),
        (['--aggregation_norm', '0'], 'aggregation_norm'),
        (['--warmup_epochs', 'nan'], 'warmup_epochs'),
        (['--warmup_epochs', '-1'], 'warmup_epochs'),
        (['--init_lr', '0'], 'init_lr'),
        (['--max_lr', 'inf'], 'max_lr'),
        (['--final_lr', '-0.1'], 'final_lr'),
        (['--init_lr', '0.002'], 'max_lr must be greater'),
        (['--grad_clip', '0'], 'grad_clip'),
        (['--cache_cutoff', 'nan'], 'cache_cutoff'),
        (['--cache_cutoff', '-1'], 'cache_cutoff'),
        (['--evidential_regularization', '-1'], 'evidential_regularization'),
        (['--quantile_loss_alpha', 'nan'], 'quantile_loss_alpha'),
        (['--split_key_molecule', '-1'], 'split_key_molecule'),
        (['--split_sizes', 'nan', '0', '1'], 'split_sizes'),
    ],
)
def test_train_args_reject_invalid_numeric_settings(flags, message):
    with pytest.raises(ValueError, match=message):
        _parse_train_args(*flags)


def test_train_args_validates_numeric_config_overrides(tmp_path):
    config_path = tmp_path / 'invalid.json'
    config_path.write_text(json.dumps({'hidden_size': True}), encoding='utf-8')

    with pytest.raises(ValueError, match='hidden_size'):
        _parse_train_args('--config_path', str(config_path))


@pytest.mark.parametrize(
    'contents',
    [
        [],
        {'hidden_szie': 300},
        {'_train_data_size': 10},
        {'config_path': 'another.json'},
    ],
)
def test_train_args_rejects_non_object_unknown_or_internal_config_keys(tmp_path, contents):
    config_path = tmp_path / 'invalid.json'
    config_path.write_text(json.dumps(contents), encoding='utf-8')

    with pytest.raises(ValueError, match='JSON object|Unknown or unsafe config'):
        _parse_train_args('--config_path', str(config_path))


@pytest.mark.parametrize(
    ('contents', 'message'),
    [
        ({'model_type': 'typo'}, 'Invalid config value'),
        ({'dataset_type': None}, 'cannot be null'),
        ({'extra_metrics': ['mae', 'not-a-metric']}, 'Invalid config value'),
        ({'features_generator': ['not-a-generator']}, 'Unknown feature generator'),
        ({'hidden_size': '300'}, 'Invalid config value'),
        ({'quiet': 'false'}, 'Invalid config value'),
        ({'gpu': -1}, 'available CUDA device'),
    ],
)
def test_train_args_config_cannot_bypass_cli_types_or_choices(tmp_path, contents, message):
    config_path = tmp_path / 'invalid-value.json'
    config_path.write_text(json.dumps(contents), encoding='utf-8')

    with pytest.raises(ValueError, match=message):
        _parse_train_args('--config_path', str(config_path))


def test_train_config_overrides_are_applied_before_derived_state(tmp_path):
    config_path = tmp_path / 'override.json'
    multimolecule_path = DATA_PATH.parent / 'regression_multimolecule.csv'
    config_path.write_text(
        json.dumps({
            'data_path': str(multimolecule_path),
            'number_of_molecules': 2,
            'test': True,
            'checkpoint_path': 'configured-model.pt',
        }),
        encoding='utf-8',
    )

    args = _parse_train_args('--config_path', str(config_path))

    assert args.data_path == str(multimolecule_path)
    assert args.number_of_molecules == 2
    assert len(args.smiles_columns) == 2
    assert args.checkpoint_paths == ['configured-model.pt']
    assert args.epochs == 0


def test_train_args_retains_zero_epoch_evaluation_compatibility():
    args = _parse_train_args('--epochs', '0')

    assert args.epochs == 0


def test_train_args_test_mode_requires_an_existing_checkpoint_source():
    with pytest.raises(ValueError, match='--test skips optimization'):
        _parse_train_args('--test')

    args = _parse_train_args('--test', '--checkpoint_path', 'existing-model.pt')
    assert args.epochs == 0
    assert args.checkpoint_paths == ['existing-model.pt']


def _parse_interpret_args(*extra_args):
    return InterpretArgs().parse_args([
        '--data_path', str(DATA_PATH),
        '--checkpoint_path', 'existing-model.pt',
        *extra_args,
    ])


@pytest.mark.parametrize(
    ('flags', 'message'),
    [
        (['--property_id', '0'], 'property_id'),
        (['--rollout', '0'], 'rollout'),
        (['--max_atoms', '0'], 'max_atoms'),
        (['--min_atoms', '0'], 'min_atoms'),
        (['--min_atoms', '3', '--max_atoms', '2'], 'min_atoms must be less'),
        (['--c_puct', 'nan'], 'c_puct'),
        (['--prop_delta', 'inf'], 'prop_delta'),
    ],
)
def test_interpret_args_reject_invalid_numeric_settings(flags, message):
    with pytest.raises(ValueError, match=message):
        _parse_interpret_args(*flags)


def _parse_hyperopt_args(*extra_args):
    return HyperoptArgs().parse_args([
        '--data_path', str(DATA_PATH),
        '--dataset_type', 'regression',
        '--config_save_path', 'best.json',
        *extra_args,
    ])


@pytest.mark.parametrize(
    ('flags', 'message'),
    [
        (['--num_iters', '0'], 'num_iters'),
        (['--num_iters', '2', '--startup_random_iters', '-1'], 'startup_random_iters'),
        (['--num_iters', '2', '--startup_random_iters', '3'], 'startup_random_iters'),
        (['--epochs', '0'], 'epochs.*greater than 0'),
    ],
)
def test_hyperopt_args_reject_invalid_iteration_settings(flags, message):
    with pytest.raises(ValueError, match=message):
        _parse_hyperopt_args(*flags)
