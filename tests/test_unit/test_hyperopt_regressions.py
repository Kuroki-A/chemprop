import json
from pathlib import Path

import pytest

from chemprop.hyperparameter_optimization import _score_to_hyperopt_loss
from chemprop.hyperopt_utils import _load_manual_validation_scores


def test_manual_hyperopt_trial_uses_only_validation_scores(tmp_path: Path):
    trial_args = {
        "metric": "rmse",
        "num_folds": 2,
        "ignore_nan_metrics": False,
    }
    for fold_num, scores in enumerate(([2.0, 8.0], [4.0, 4.0])):
        fold_dir = tmp_path / f"fold_{fold_num}"
        fold_dir.mkdir()
        (fold_dir / "valid_scores.json").write_text(json.dumps({"rmse": scores}))

    # A deliberately conflicting held-out score must have no influence.
    (tmp_path / "test_scores.csv").write_text(
        "Task,Mean rmse,Standard deviation rmse,Fold 0 rmse\n"
        "target,999,0,999\n"
    )

    mean_score, std_score = _load_manual_validation_scores(str(tmp_path), trial_args)

    assert mean_score == pytest.approx(4.0)
    assert std_score == pytest.approx(0.0)


def test_manual_hyperopt_trial_never_falls_back_to_test_scores(tmp_path: Path):
    (tmp_path / "test_scores.csv").write_text(
        "Task,Mean rmse,Standard deviation rmse,Fold 0 rmse\n"
        "target,1,0,1\n"
    )

    with pytest.raises(FileNotFoundError, match="Test scores cannot be used"):
        _load_manual_validation_scores(
            str(tmp_path),
            {"metric": "rmse", "num_folds": 1, "ignore_nan_metrics": False},
        )


@pytest.mark.parametrize('minimize_score', [False, True])
@pytest.mark.parametrize('score', [float('nan'), float('inf'), -float('inf')])
def test_invalid_validation_score_is_always_worst_hyperopt_loss(
    score, minimize_score,
):
    assert _score_to_hyperopt_loss(score, minimize_score) == float('inf')


def test_finite_hyperopt_loss_respects_metric_direction():
    assert _score_to_hyperopt_loss(0.25, minimize_score=True) == 0.25
    assert _score_to_hyperopt_loss(0.25, minimize_score=False) == -0.25
