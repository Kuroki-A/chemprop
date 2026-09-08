"""Regression tests for atom/bond target schema inference."""

import pytest

from chemprop.data import get_mixed_task_names


def test_mixed_task_schema_skips_invalid_leading_smiles(tmp_path):
    path = tmp_path / 'mixed.csv'
    path.write_text(
        'smiles,atom,bond,molecule\n'
        'not-a-smiles,"[1,2,3]","[1,2]",1\n'
        'CCO,"[1,2,3]","[1,2]",1\n'
    )

    atom, bond, molecule = get_mixed_task_names(str(path))

    assert atom == ['atom']
    assert bond == ['bond']
    assert molecule == ['molecule']


def test_mixed_task_schema_defers_equal_atom_bond_count_rows(tmp_path):
    path = tmp_path / 'cycles.csv'
    path.write_text(
        'smiles,atom,bond\n'
        'C1CC1,"[1,2,3]","[4,5,6]"\n'
        'CCC,"[1,2,3]","[4,5]"\n'
    )

    atom, bond, molecule = get_mixed_task_names(str(path))

    assert atom == ['atom']
    assert bond == ['bond']
    assert molecule == []


def test_mixed_task_schema_requires_one_smiles_column(tmp_path):
    path = tmp_path / 'multi.csv'
    path.write_text('left,right,target\nC,CC,"[1]"\n')

    with pytest.raises(ValueError, match='exactly one SMILES column'):
        get_mixed_task_names(
            str(path), smiles_columns=['left', 'right'],
            target_columns=['target'],
        )


def test_mixed_task_schema_reports_all_invalid_smiles(tmp_path):
    path = tmp_path / 'invalid.csv'
    path.write_text('smiles,target\nnot-a-smiles,"[1]"\n')

    with pytest.raises(ValueError, match='no valid SMILES rows'):
        get_mixed_task_names(str(path))
