import json
from pathlib import Path

import pytest
from hyperopt import Trials

from chemprop.hyperparameter_optimization import _score_to_hyperopt_loss
from chemprop.hyperopt_utils import (
    _load_manual_validation_scores,
    build_search_space,
    get_hyperopt_seed,
    load_trials,
    save_trials,
)


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


@pytest.mark.parametrize('invalid_score', [float('nan'), float('inf'), -float('inf')])
def test_manual_hyperopt_trial_rejects_non_finite_validation_scores(
    tmp_path: Path, invalid_score: float,
):
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    (fold_dir / "valid_scores.json").write_text(
        json.dumps({"rmse": [invalid_score]})
    )

    with pytest.raises(ValueError, match="non-finite validation rmse"):
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


def test_hyperopt_space_handles_no_or_short_warmup_search():
    assert set(build_search_space(["depth"])) == {"depth"}
    assert set(build_search_space(["warmup_epochs"], train_epochs=0)) == {
        "warmup_epochs"
    }
    assert set(build_search_space(["warmup_epochs"], train_epochs=1)) == {
        "warmup_epochs"
    }

    with pytest.raises(ValueError, match="train_epochs is required"):
        build_search_space(["warmup_epochs"])
    with pytest.raises(ValueError, match="Unsupported hyperparameter"):
        build_search_space(["typo"], train_epochs=10)


def test_hyperopt_seed_file_handles_empty_state_and_reserves_unique_seeds(
    tmp_path: Path,
):
    assert get_hyperopt_seed(7, str(tmp_path)) == 7
    assert get_hyperopt_seed(7, str(tmp_path)) == 8

    seed_files = [path for path in tmp_path.iterdir() if path.name != "7.pkl"]
    assert len(seed_files) == 1
    assert seed_files[0].read_text().strip() == "7 8"


def test_hyperopt_trial_publish_is_atomic_and_never_overwrites(
    tmp_path: Path,
):
    first = Trials()
    first.marker = "first"
    second = Trials()
    second.marker = "second"

    save_trials(str(tmp_path), first, hyperopt_seed=3)
    original_bytes = (tmp_path / "3.pkl").read_bytes()
    save_trials(str(tmp_path), second, hyperopt_seed=3)

    assert (tmp_path / "3.pkl").read_bytes() == original_bytes
    assert not list(tmp_path.glob("*.tmp"))
    loaded = load_trials(str(tmp_path))
    assert isinstance(loaded, Trials)


def test_load_trials_ignores_unpublished_temporary_pickle(tmp_path: Path):
    (tmp_path / "unfinished.pkl.tmp").write_bytes(b"not a pickle")
    assert len(load_trials(str(tmp_path)).trials) == 0
