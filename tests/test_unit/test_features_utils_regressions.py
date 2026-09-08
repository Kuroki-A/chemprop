"""Regression tests for feature-file loading."""

import numpy as np
import pandas as pd
import pytest

from chemprop.features import load_features, load_valid_atom_or_bond_features


def test_load_features_accepts_case_insensitive_numpy_extensions(tmp_path):
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]])

    lower_npz = tmp_path / 'features.npz'
    upper_npz = tmp_path / 'features.NPZ'
    np.savez_compressed(lower_npz, features=values)
    lower_npz.rename(upper_npz)
    np.testing.assert_array_equal(load_features(str(upper_npz)), values)

    lower_npy = tmp_path / 'features.npy'
    upper_npy = tmp_path / 'features.NPY'
    np.save(lower_npy, values)
    lower_npy.rename(upper_npy)
    np.testing.assert_array_equal(load_features(str(upper_npy)), values)


def test_load_features_requires_the_documented_npz_array_name(tmp_path):
    path = tmp_path / 'features.npz'
    np.savez_compressed(path, unexpected=np.ones((2, 3)))

    with pytest.raises(ValueError, match='array named "features"'):
        load_features(str(path))


def test_atom_or_bond_npz_loading_preserves_archive_order(tmp_path):
    path = tmp_path / 'atom_features.NPZ'
    lower_path = tmp_path / 'atom_features.npz'
    first = np.asarray([[1.0], [2.0]])
    second = np.asarray([[3.0, 4.0]])
    np.savez_compressed(lower_path, row_0=first, row_1=second)
    lower_path.rename(path)

    loaded = load_valid_atom_or_bond_features(str(path), ['CC', 'C'])
    assert len(loaded) == 2
    np.testing.assert_array_equal(loaded[0], first)
    np.testing.assert_array_equal(loaded[1], second)


def test_empty_atom_or_bond_dataframe_has_an_explicit_error(tmp_path):
    path = tmp_path / 'empty.pkl'
    pd.DataFrame().to_pickle(path)

    with pytest.raises(ValueError, match='is empty'):
        load_valid_atom_or_bond_features(str(path), [])


def test_pickle_atom_or_bond_features_are_aligned_by_smiles(tmp_path):
    path = tmp_path / 'descriptors.pkl'
    pd.DataFrame(
        {
            'first': [np.array([30.0]), np.array([10.0, 11.0])],
            'second': [np.array([31.0]), np.array([12.0, 13.0])],
        },
        index=['C', 'CC'],
    ).to_pickle(path)

    loaded = load_valid_atom_or_bond_features(str(path), ['CC', 'C'])

    np.testing.assert_array_equal(
        loaded[0], np.array([[10.0, 12.0], [11.0, 13.0]])
    )
    np.testing.assert_array_equal(loaded[1], np.array([[30.0, 31.0]]))


def test_pickle_atom_or_bond_features_reject_ambiguous_duplicate_smiles(
    tmp_path,
):
    path = tmp_path / 'duplicate_descriptors.pkl'
    pd.DataFrame(
        {'descriptor': [np.array([1.0]), np.array([2.0]), np.array([3.0])]},
        index=['C', 'CC', 'C'],
    ).to_pickle(path)

    with pytest.raises(ValueError, match='duplicate SMILES.*ambiguous'):
        load_valid_atom_or_bond_features(str(path), ['C', 'C', 'CC'])


def test_sdf_descriptor_columns_are_detected_across_all_rows(
    monkeypatch, tmp_path,
):
    path = tmp_path / 'descriptors.sdf'
    sdf_frame = pd.DataFrame(
        {
            'ID': ['first', 'second'],
            'SMILES': ['C', 'CC'],
            'atomic_descriptor': ['1.0', '2.0,3.0'],
            'scalar_metadata': ['100', '200'],
            'ROMol': [object(), object()],
        }
    )
    monkeypatch.setattr(
        'chemprop.features.utils.PandasTools.LoadSDF',
        lambda _: sdf_frame.copy(),
    )

    # Even a filtered request containing only the scalar first record must use
    # the column schema discovered from all SDF records.
    scalar_only = load_valid_atom_or_bond_features(str(path), ['C'])
    np.testing.assert_array_equal(scalar_only[0], np.array([[1.0]]))

    loaded = load_valid_atom_or_bond_features(str(path), ['C', 'CC'])

    np.testing.assert_array_equal(loaded[0], np.array([[1.0]]))
    np.testing.assert_array_equal(loaded[1], np.array([[2.0], [3.0]]))


def test_sdf_descriptor_loader_rejects_duplicate_requested_smiles(
    monkeypatch, tmp_path,
):
    path = tmp_path / 'descriptors.sdf'
    sdf_frame = pd.DataFrame(
        {
            'SMILES': ['CC', 'CC'],
            'descriptor': ['1.0,2.0', '3.0,4.0'],
        }
    )
    monkeypatch.setattr(
        'chemprop.features.utils.PandasTools.LoadSDF',
        lambda _: sdf_frame.copy(),
    )

    with pytest.raises(ValueError, match='multiple SDF records'):
        load_valid_atom_or_bond_features(str(path), ['CC'])
