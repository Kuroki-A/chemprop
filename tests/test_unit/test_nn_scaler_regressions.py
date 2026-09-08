import warnings

import numpy as np
import pytest
import torch

from chemprop.data.scaler import StandardScaler
from chemprop.nn_utils import NoamLR


def _make_optimizer():
    parameter = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.Adam([parameter], lr=1e-4)


def _make_scheduler(**overrides):
    arguments = {
        'optimizer': _make_optimizer(),
        'warmup_epochs': [0],
        'total_epochs': [2],
        'steps_per_epoch': 2,
        'init_lr': [1e-4],
        'max_lr': [1e-3],
        'final_lr': [1e-4],
    }
    arguments.update(overrides)
    return NoamLR(**arguments)


def test_noam_lr_zero_warmup_is_finite_and_reaches_final_lr():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        scheduler = _make_scheduler()
        scheduler.step(0)
        observed = [scheduler.get_lr()[0]]
        for step in range(1, 5):
            scheduler.step(step)
            observed.append(scheduler.get_lr()[0])

    assert np.isfinite(observed).all()
    assert observed[0] == pytest.approx(1e-4)
    assert observed[-1] == pytest.approx(1e-4)
    assert not any(isinstance(item.message, RuntimeWarning) for item in caught)


def test_noam_lr_zero_epochs_is_finite_for_evaluation_only_mode():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        scheduler = _make_scheduler(total_epochs=[0], warmup_epochs=[2])
        scheduler.step(0)
        initial_lr = scheduler.get_lr()[0]
        scheduler.step(1)
        post_schedule_lr = scheduler.get_lr()[0]

    assert initial_lr == pytest.approx(1e-4)
    assert post_schedule_lr == pytest.approx(1e-4)
    assert not any(isinstance(item.message, RuntimeWarning) for item in caught)


def test_noam_lr_clips_warmup_longer_than_short_training_run():
    scheduler = _make_scheduler(total_epochs=[1], warmup_epochs=[2])

    scheduler.step(2)

    assert scheduler.warmup_steps.tolist() == [2]
    assert scheduler.get_lr()[0] == pytest.approx(1e-3)


@pytest.mark.parametrize(
    ('override', 'message'),
    [
        ({'steps_per_epoch': 0}, 'steps_per_epoch'),
        ({'total_epochs': [-1]}, 'total_epochs'),
        ({'warmup_epochs': [-1]}, 'warmup_epochs'),
        ({'warmup_epochs': [np.nan]}, 'warmup_epochs'),
        ({'init_lr': [0]}, 'init_lr'),
        ({'max_lr': [np.inf]}, 'max_lr'),
        ({'final_lr': [-1e-4]}, 'final_lr'),
        ({'init_lr': [2e-3]}, 'greater than or equal'),
    ],
)
def test_noam_lr_rejects_invalid_schedule_parameters(override, message):
    with pytest.raises(ValueError, match=message):
        _make_scheduler(**override)


def test_noam_lr_rejects_invalid_explicit_step():
    scheduler = _make_scheduler()

    with pytest.raises(ValueError, match='current_step'):
        scheduler.step(-1)


def test_standard_scaler_handles_an_entirely_missing_column_without_warnings():
    values = [[1.0, None], [3.0, None]]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        scaler = StandardScaler(replace_nan_token=0).fit(values)
        transformed = scaler.transform(values)

    assert scaler.means.tolist() == [2.0, 0.0]
    assert scaler.stds.tolist() == [1.0, 1.0]
    assert transformed.tolist() == [[-1.0, 0.0], [1.0, 0.0]]
    assert not any(isinstance(item.message, RuntimeWarning) for item in caught)


@pytest.mark.parametrize(
    'values',
    [
        [],
        [1.0, 2.0],
        [[1.0], [2.0, 3.0]],
        [[1.0, np.inf]],
        [[1.0, -np.inf]],
    ],
)
def test_standard_scaler_rejects_empty_malformed_or_infinite_fit_input(values):
    with pytest.raises(ValueError, match='StandardScaler.fit'):
        StandardScaler().fit(values)


def test_standard_scaler_validates_state_and_transform_width():
    with pytest.raises(ValueError, match='must be fitted'):
        StandardScaler().transform([[1.0]])

    scaler = StandardScaler(means=np.array([0.0]), stds=np.array([0.0]))
    with pytest.raises(ValueError, match='finite and positive'):
        scaler.transform([[1.0]])

    scaler = StandardScaler().fit([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match='feature width'):
        scaler.transform([[1.0]])
    with pytest.raises(ValueError, match='infinite'):
        scaler.inverse_transform([[np.inf, 0.0]])
