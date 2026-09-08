"""Regression tests for row-level constraint input validation."""

import numpy as np
import pytest

from chemprop.data import get_constraints


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
