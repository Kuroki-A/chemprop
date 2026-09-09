"""Regression tests for molecular feature generation and cache isolation."""

import inspect
import json
import weakref

import numpy as np
import pandas as pd
import pytest

from chemprop.args import TrainArgs
from chemprop.data import (
    MoleculeDatapoint,
    MoleculeDataset,
    empty_cache,
    get_data,
    get_data_from_smiles,
    load_selected_feature_columns,
)
from chemprop.data.data import (
    SMILES_TO_GRAPH,
    cache_mol,
    generate_features_for_smiles_batch,
    make_mols,
    set_cache_mol,
)
from chemprop.data.utils import (
    _feature_manifest_schema,
    _ordered_smiles_sha256,
    get_data as get_data_function,
)
from chemprop.features import (
    register_features_generator,
    reset_featurization_parameters,
    set_extra_atom_fdim,
    set_keeping_atom_map,
    set_reaction,
)
from chemprop.features import features_generators as feature_generators_module


CALLS = 0


@register_features_generator('_test_selected_features')
def selected_features_test_generator(mol, selected_feature_columns=None):
    global CALLS
    CALLS += 1
    size = 3 if selected_feature_columns is None else len(selected_feature_columns)
    return np.arange(size, dtype=float)


@register_features_generator('_test_molecule_values')
def molecule_values_test_generator(mol, selected_feature_columns=None):
    return np.array([mol.GetNumHeavyAtoms(), mol.GetNumAtoms()], dtype=float)


@register_features_generator('_test_first_atom')
def first_atom_test_generator(mol, selected_feature_columns=None):
    return np.array([mol.GetAtomWithIdx(0).GetAtomicNum()], dtype=float)


@register_features_generator('_test_integer_values')
def integer_values_test_generator(mol, selected_feature_columns=None):
    return np.array([mol.GetNumHeavyAtoms(), mol.GetNumAtoms()], dtype=np.int16)


def _selected_features_csv(tmp_path):
    path = tmp_path / 'selected_features.csv'
    pd.DataFrame({'_test_selected_features': ['f0', 'f1']}).to_csv(path, index=False)
    return str(path)


def _write_feature_manifest(feature_path, smiles, dimension, dtype='float64'):
    manifest = {
        'schema_version': 1,
        'generator': '_test_external',
        'generator_config': {},
        'versions': {},
        'feature_names': [f'feature_{index}' for index in range(dimension)],
        'dimension': dimension,
        'dtype': dtype,
        'implementation_sha256': '0' * 64,
        'input': {
            'ordered_smiles_sha256': _ordered_smiles_sha256(smiles),
            'ordered_smiles_encoding': 'chemprop-length-prefixed-utf8-v1',
            'num_smiles': len(smiles),
        },
        'num_molecules': len(smiles),
        'num_molecules_completed': len(smiles),
        'status': 'complete',
        'storage': 'npz',
    }
    manifest_path = feature_path.with_name(feature_path.name + '.manifest.json')
    manifest_path.write_text(json.dumps(manifest))
    return manifest


def test_selected_feature_csv_is_loaded_once(monkeypatch, tmp_path):
    empty_cache()
    path = _selected_features_csv(tmp_path)
    reads = 0
    original_read_csv = pd.read_csv

    def counted_read_csv(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr('chemprop.data.data.pd.read_csv', counted_read_csv)
    for smiles in ['CC', 'CCC', 'CCCC']:
        datapoint = MoleculeDatapoint(
            [smiles],
            features_generator=['_test_selected_features'],
            selected_features_path=path,
        )
        assert datapoint.features.shape == (2,)

    assert reads == 1


def test_selected_feature_cache_is_not_mutable_by_callers(tmp_path):
    empty_cache()
    path = _selected_features_csv(tmp_path)

    first = load_selected_feature_columns(path)
    first['_test_selected_features'] = ('changed',)
    first['new_generator'] = ('unexpected',)

    second = load_selected_feature_columns(path)
    assert second == {'_test_selected_features': ('f0', 'f1')}


@pytest.mark.parametrize('prefix', ['', '\ufeff'])
def test_selected_feature_csv_rejects_duplicate_generator_columns(tmp_path, prefix):
    path = tmp_path / 'duplicate_selected_features.csv'
    path.write_text(f'{prefix}morgan,morgan\nbit_1,bit_2\n', encoding='utf-8')

    with pytest.raises(ValueError, match='duplicate generator columns'):
        load_selected_feature_columns(str(path))


@pytest.mark.parametrize('header', ['', 'morgan,', ' ,rdkit'])
def test_selected_feature_csv_rejects_blank_generator_columns(tmp_path, header):
    path = tmp_path / 'blank_selected_features.csv'
    path.write_text(f'{header}\nbit_1,bit_2\n', encoding='utf-8')

    with pytest.raises(ValueError, match='non-blank generator names'):
        load_selected_feature_columns(str(path))


def test_empty_dataset_is_not_an_atom_or_bond_target_dataset():
    dataset = MoleculeDataset([])
    assert dataset.is_atom_bond_targets is False
    assert dataset.batch_graph() == []
    assert dataset.data_weights() == []
    assert dataset.atom_bond_data_weights() == []
    assert dataset.gt_targets() is None
    assert dataset.lt_targets() is None


@pytest.mark.parametrize(
    ('argument', 'label'),
    [
        ('features', 'Molecular features'),
        ('atom_features', 'Atom features'),
        ('atom_descriptors', 'Atom descriptors'),
        ('bond_features', 'Bond features'),
        ('bond_descriptors', 'Bond descriptors'),
    ],
)
def test_datapoint_rejects_infinite_feature_inputs(argument, label):
    value = (
        np.array([np.inf, 1.0])
        if argument == 'features'
        else np.array([[np.inf], [1.0]])
    )

    with pytest.raises(ValueError, match=rf'{label} contains an infinite value'):
        MoleculeDatapoint(['CC'], **{argument: value})


def test_datapoint_replaces_nan_in_every_feature_input():
    datapoint = MoleculeDatapoint(
        ['CC'],
        features=np.array([np.nan, 1.0]),
        atom_features=np.array([[np.nan], [1.0]]),
        atom_descriptors=np.array([[np.nan], [1.0]]),
        bond_features=np.array([[np.nan]]),
        bond_descriptors=np.array([[np.nan]]),
    )

    assert datapoint.features[0] == 0
    assert datapoint.atom_features[0, 0] == 0
    assert datapoint.atom_descriptors[0, 0] == 0
    assert datapoint.bond_features[0, 0] == 0
    assert datapoint.bond_descriptors[0, 0] == 0


def test_precomputed_generated_features_are_not_copied_when_already_valid():
    features = np.array([1.0, 2.0], dtype=np.float32)

    datapoint = MoleculeDatapoint(
        ['CC'],
        features=features,
        features_generator=['_test_selected_features'],
        features_generator_precomputed=True,
    )

    assert datapoint.features is features


def test_direct_feature_input_retains_historical_copy_semantics():
    features = np.array([1.0, 2.0], dtype=np.float32)

    datapoint = MoleculeDatapoint(['CC'], features=features)

    assert datapoint.features is not features
    np.testing.assert_array_equal(datapoint.features, features)


def test_datapoint_rejects_non_numeric_feature_inputs():
    with pytest.raises(ValueError, match='real-valued numeric array'):
        MoleculeDatapoint(['CC'], features=np.array(['not-a-number']))


def test_zero_bond_feature_matrices_retain_their_feature_width():
    dataset = MoleculeDataset([
        MoleculeDatapoint(
            ['C'],
            bond_features=np.empty((0, 3)),
            bond_descriptors=np.empty((0, 4)),
        )
    ])

    assert dataset.bond_features_size() == 3
    assert dataset.bond_descriptors_size() == 4


def test_reaction_honors_selected_feature_columns(tmp_path):
    empty_cache()
    reset_featurization_parameters()
    set_reaction(True, 'reac_diff')
    path = _selected_features_csv(tmp_path)
    try:
        datapoint = MoleculeDatapoint(
            ['CC>>CCC'],
            features_generator=['_test_selected_features'],
            selected_features_path=path,
        )
        assert datapoint.features.shape == (2,)
    finally:
        reset_featurization_parameters()
        empty_cache()


def test_duplicate_smiles_feature_generation_is_cached():
    global CALLS
    empty_cache()
    CALLS = 0
    MoleculeDatapoint(['CC'], features_generator=['_test_selected_features'])
    MoleculeDatapoint(['CC'], features_generator=['_test_selected_features'])
    assert CALLS == 1


def test_molecule_cache_key_includes_hydrogen_settings():
    empty_cache()
    implicit = make_mols(['C'], [False], [False], [False], [False])[0]
    explicit = make_mols(['C'], [False], [False], [True], [False])[0]
    assert implicit.GetNumAtoms() == 1
    assert explicit.GetNumAtoms() == 5


def test_graph_cache_does_not_reuse_row_specific_atom_features():
    empty_cache()
    reset_featurization_parameters()
    set_extra_atom_fdim(1)
    try:
        first = MoleculeDatapoint(['CC'], atom_features=np.ones((2, 1)))
        second = MoleculeDatapoint(['CC'], atom_features=np.zeros((2, 1)))
        MoleculeDataset([first, second]).batch_graph()
        assert not SMILES_TO_GRAPH
    finally:
        reset_featurization_parameters()
        empty_cache()


def test_graph_cache_key_matches_atom_target_hydrogen_parsing():
    empty_cache()
    reset_featurization_parameters()
    set_keeping_atom_map(True)
    try:
        mapped_methane = '[H:2][C:1]([H:3])([H:4])[H:5]'
        plain = MoleculeDatapoint([mapped_methane])
        atomic = MoleculeDatapoint(
            [mapped_methane],
            atom_targets=[np.zeros(5)],
        )
        MoleculeDataset([plain]).batch_graph()
        MoleculeDataset([atomic]).batch_graph()

        assert sorted(graph.n_atoms for graph in SMILES_TO_GRAPH.values()) == [1, 5]
    finally:
        reset_featurization_parameters()
        empty_cache()


def test_batch_generation_matches_scalar_generator_order():
    empty_cache()
    reset_featurization_parameters()
    smiles = [['C', 'CC'], ['CCC', 'CCCC']]
    generators = ['_test_molecule_values', '_test_selected_features']

    expected = [
        MoleculeDatapoint(row, features_generator=generators).features
        for row in smiles
    ]
    empty_cache()
    actual = get_data_from_smiles(
        smiles,
        features_generator=generators,
        skip_invalid_smiles=False,
    )

    for expected_row, actual_row in zip(expected, actual.features()):
        np.testing.assert_array_equal(actual_row, expected_row)


def test_batch_vector_parts_preserve_selected_reaction_h_invalid_and_dtype(tmp_path):
    empty_cache()
    reset_featurization_parameters()
    set_reaction(True, 'reac_diff')
    selected_path = _selected_features_csv(tmp_path)
    smiles = [
        ['CC', 'CO'],
        ['[H][H]', 'N'],
        ['CC>>CO', 'CO'],
        ['not-a-smiles', 'CC'],
    ]
    generators = ['_test_integer_values', '_test_selected_features']
    try:
        expected = [
            MoleculeDatapoint(
                row,
                features_generator=generators,
                selected_features_path=selected_path,
            ).features
            for row in smiles
        ]
        empty_cache()
        actual = get_data_from_smiles(
            smiles,
            features_generator=generators,
            selected_features_path=selected_path,
            skip_invalid_smiles=False,
        )

        for expected_row, actual_row in zip(expected, actual.features()):
            np.testing.assert_array_equal(actual_row, expected_row)
            assert actual_row.dtype == expected_row.dtype
        assert actual._features_source_metadata == {
            'schema_version': 1,
            'external_features': [],
            'phase_features': None,
            'generated_dimension': len(expected[0]),
            'total_dimension': len(expected[0]),
        }
    finally:
        reset_featurization_parameters()
        empty_cache()


def test_batch_vector_parts_preserve_integer_h_and_empty_dtypes():
    empty_cache()
    reset_featurization_parameters()
    smiles = [['CC'], ['[H][H]'], ['not-a-smiles']]
    expected = [
        MoleculeDatapoint(
            row, features_generator=['_test_integer_values'],
        ).features
        for row in smiles
    ]
    empty_cache()
    actual = generate_features_for_smiles_batch(
        smiles, ['_test_integer_values'],
    )

    for expected_row, actual_row in zip(expected, actual):
        np.testing.assert_array_equal(actual_row, expected_row)
        assert actual_row.dtype == expected_row.dtype


def test_batch_duplicate_keys_are_computed_once_for_all_generators(monkeypatch):
    empty_cache()
    reset_featurization_parameters()
    mol_to_smiles_calls = 0
    original_mol_to_smiles = feature_generators_module.Chem.MolToSmiles

    def counted_mol_to_smiles(*args, **kwargs):
        nonlocal mol_to_smiles_calls
        if kwargs.get('canonical') is False:
            mol_to_smiles_calls += 1
        return original_mol_to_smiles(*args, **kwargs)

    monkeypatch.setattr(
        'chemprop.data.data.Chem.MolToSmiles', counted_mol_to_smiles,
    )
    generate_features_for_smiles_batch(
        [['CC', 'CO'], ['CC', '[H][H]']],
        ['_test_integer_values', '_test_molecule_values'],
    )

    # Three heavy-atom positions are serialized once each, rather than once
    # per requested generator. Hydrogen-only positions use the sentinel path.
    assert mol_to_smiles_calls == 3


def test_builtin_batch_deduplicates_noncanonical_and_atom_mapped_smiles(monkeypatch):
    empty_cache()
    reset_featurization_parameters()
    set_keeping_atom_map(True)
    generator = feature_generators_module.get_features_generator('morgan')
    original_batch_transform = generator.batch_transform
    generated_batch_sizes = []

    def counted_batch_transform(mols, **kwargs):
        generated_batch_sizes.append(len(mols))
        return original_batch_transform(mols, **kwargs)

    monkeypatch.setattr(generator, 'batch_transform', counted_batch_transform)
    try:
        result = generate_features_for_smiles_batch(
            [
                ['CO'],
                ['OC'],
                ['[CH3:1][OH:2]'],
                ['[OH:9][CH3:8]'],
            ],
            ['morgan'],
        )

        # All four strings describe methanol. Managed generators ignore atom
        # ordering and atom-map identifiers, so only one vector is computed.
        assert generated_batch_sizes == [1]
        for row in result[1:]:
            np.testing.assert_array_equal(row, result[0])
    finally:
        reset_featurization_parameters()
        empty_cache()


def test_custom_generator_retains_atom_order_preserving_deduplication():
    generated_batch_sizes = []

    def generator(mol, selected_feature_columns=None):
        return np.array([mol.GetAtomWithIdx(0).GetAtomicNum()], dtype=np.int16)

    def batch_transform(mols):
        generated_batch_sizes.append(len(mols))
        return [generator(mol) for mol in mols]

    generator.batch_transform = batch_transform

    result = generate_features_for_smiles_batch(
        [['CO'], ['OC']],
        ['_test_atom_order_generator'],
        generator_overrides={'_test_atom_order_generator': generator},
    )

    assert generated_batch_sizes == [2]
    assert [row.tolist() for row in result] == [[6], [8]]


def test_batch_finalization_releases_source_rows_incrementally(monkeypatch):
    """Final output allocation must not retain a second full feature matrix."""
    source_references = []

    def generator(_mol, selected_feature_columns=None):
        pytest.fail('The native batch transform should be used.')

    def batch_transform(mols):
        rows = [
            np.full(4096, index, dtype=np.float32)
            for index in range(len(mols))
        ]
        source_references.extend(weakref.ref(row) for row in rows)
        return rows

    generator.batch_transform = batch_transform
    generator.preferred_batch_size = 256

    original_concatenate = np.concatenate
    live_source_counts = []

    def recording_concatenate(parts, *args, **kwargs):
        live_source_counts.append(sum(ref() is not None for ref in source_references))
        return original_concatenate(parts, *args, **kwargs)

    monkeypatch.setattr('chemprop.data.data.np.concatenate', recording_concatenate)

    result = generate_features_for_smiles_batch(
        [['C'], ['CC'], ['CCC'], ['CO']],
        ['_test_memory_generator'],
        generator_overrides={'_test_memory_generator': generator},
    )

    assert live_source_counts == [4, 3, 2, 1]
    assert all(ref() is None for ref in source_references)
    assert [row.shape for row in result] == [(4096,)] * 4


def test_batch_parsing_releases_unused_reaction_products_before_generation(monkeypatch):
    empty_cache()
    previous_cache_mol = cache_mol()
    set_cache_mol(False)
    product_references = []
    original_make_mols = make_mols

    def tracked_make_mols(*args, **kwargs):
        mols = original_make_mols(*args, **kwargs)
        product_references.extend(
            weakref.ref(mol[1])
            for mol in mols
            if isinstance(mol, tuple) and mol[1] is not None
        )
        return mols

    def generator(mol, selected_feature_columns=None):
        return np.array([mol.GetNumHeavyAtoms()], dtype=np.int16)

    def batch_transform(mols):
        assert all(ref() is None for ref in product_references)
        return [generator(mol) for mol in mols]

    generator.batch_transform = batch_transform
    monkeypatch.setattr('chemprop.data.data.make_mols', tracked_make_mols)
    try:
        result = generate_features_for_smiles_batch(
            [['C>>CC'], ['C>>CCC'], ['C>>CO'], ['C>>CN']],
            ['_test_reaction_memory_generator'],
            generator_overrides={'_test_reaction_memory_generator': generator},
            auto_detect_reactions=True,
        )
    finally:
        set_cache_mol(previous_cache_mol)
        empty_cache()

    assert [row.tolist() for row in result] == [[1], [1], [1], [1]]


def test_runtime_batching_chunks_unique_molecules_and_scatters_duplicates():
    batch_sizes = []

    def generator(_mol, selected_feature_columns=None):
        pytest.fail('The native batch transform should be used.')

    def batch_transform(mols):
        batch_sizes.append(len(mols))
        return [
            np.array([mol.GetNumHeavyAtoms()], dtype=np.int16)
            for mol in mols
        ]

    generator.batch_transform = batch_transform
    generator.preferred_batch_size = 2
    smiles = [
        ['C', 'CC'],
        ['CCC', 'C'],
        ['CO', 'CN'],
        ['CC', 'CO'],
    ]

    result = generate_features_for_smiles_batch(
        smiles,
        ['_test_chunked_generator'],
        generator_overrides={'_test_chunked_generator': generator},
    )

    # Five unique atom-order-preserving molecules are generated in bounded
    # native batches, then duplicates are restored in their original columns.
    assert batch_sizes == [2, 2, 1]
    assert [row.tolist() for row in result] == [
        [1, 2],
        [3, 1],
        [2, 2],
        [2, 2],
    ]
    assert all(row.dtype == np.int16 for row in result)


def test_get_data_batches_duplicates_and_preserves_external_feature_alignment(tmp_path):
    global CALLS
    empty_cache()
    CALLS = 0
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCC,\nCCC,2\n')
    external_path = tmp_path / 'external.npy'
    np.save(external_path, np.array([[10.0], [20.0], [30.0]]))

    data = get_data(
        path=str(data_path),
        features_path=[str(external_path)],
        features_generator=['_test_selected_features'],
        skip_none_targets=True,
    )

    assert len(data) == 2
    # Duplicate CC is skipped because its target is missing, while external
    # features remain aligned to the original CSV row indices.
    np.testing.assert_array_equal(data[0].features, [10.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(data[1].features, [30.0, 0.0, 1.0, 2.0])
    assert CALLS == 2


def test_feature_source_metadata_distinguishes_external_and_phase_layouts(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,2\n')
    external_two = tmp_path / 'external_two.csv'
    external_two.write_text('descriptor_a,descriptor_b\n1,2\n3,4\n')
    external_one = tmp_path / 'external_one.csv'
    external_one.write_text('descriptor_a\n1\n3\n')
    phase_one = tmp_path / 'phase.csv'
    phase_one.write_text('liquid\n1\n1\n')

    two_external_columns = get_data(
        path=str(data_path),
        features_path=[str(external_two)],
    )
    external_plus_phase = get_data(
        path=str(data_path),
        features_path=[str(external_one)],
        phase_features_path=str(phase_one),
    )

    assert two_external_columns.features_size() == 2
    assert external_plus_phase.features_size() == 2
    first_schema = two_external_columns._features_source_metadata
    second_schema = external_plus_phase._features_source_metadata
    assert first_schema != second_schema
    assert first_schema == {
        'schema_version': 1,
        'external_features': [{
            'dimension': 2,
            'dtype': 'float64',
            'csv_header': ['descriptor_a', 'descriptor_b'],
        }],
        'phase_features': None,
        'generated_dimension': 0,
        'total_dimension': 2,
    }
    assert second_schema == {
        'schema_version': 1,
        'external_features': [{
            'dimension': 1,
            'dtype': 'float64',
            'csv_header': ['descriptor_a'],
        }],
        'phase_features': {
            'dimension': 1,
            'dtype': 'float64',
            'csv_header': ['liquid'],
        },
        'generated_dimension': 0,
        'total_dimension': 2,
    }


def test_empty_feature_source_metadata_uses_static_or_unknown_generator_width(tmp_path):
    data_path = tmp_path / 'empty.csv'
    data_path.write_text('smiles,target\n')

    empty_morgan_file = get_data(
        path=str(data_path),
        features_generator=['morgan'],
    )
    empty_maccs_file = get_data(
        path=str(data_path),
        features_generator=['maccs'],
    )
    empty_smiles_api = get_data_from_smiles(
        [],
        features_generator=['morgan'],
    )

    assert empty_morgan_file._features_source_metadata['generated_dimension'] == 2048
    assert empty_morgan_file._features_source_metadata['total_dimension'] == 2048
    assert empty_maccs_file._features_source_metadata['generated_dimension'] == 167
    assert empty_maccs_file._features_source_metadata['total_dimension'] == 167
    assert empty_smiles_api._features_source_metadata['generated_dimension'] is None
    assert empty_smiles_api._features_source_metadata['total_dimension'] is None


def test_feature_source_metadata_uses_semantic_save_features_manifest(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,2\n')
    values = np.array([[1, 2], [3, 4]], dtype=np.float32)

    schemas = []
    for generator_name in ('morgan', 'maccs'):
        feature_path = tmp_path / f'{generator_name}.npz'
        np.savez_compressed(feature_path, features=values)
        manifest = {
            'schema_version': 2,
            'generator': generator_name,
            'generator_config': {'example': generator_name},
            'versions': {'rdkit': 'test'},
            'feature_names': ['first', 'second'],
            'dimension': 2,
            'dtype': 'float32',
            'semantic_revision': 1,
            # Operational and row-dependent fields must not become model
            # compatibility constraints.
            'input': {'data_sha256': generator_name},
            'num_molecules': 2,
            'num_molecules_completed': 2,
            'status': 'complete',
            'storage': 'npz',
            'temporary_file_count': 0,
            'feature_file': feature_path.name,
            'manifest_file': feature_path.name + '.manifest.json',
        }
        manifest_path = tmp_path / (feature_path.name + '.manifest.json')
        manifest_path.write_text(json.dumps(manifest))

        dataset = get_data(
            path=str(data_path),
            features_path=[str(feature_path)],
        )
        source = dataset._features_source_metadata['external_features'][0]
        schemas.append(source)
        assert source['feature_manifest'] == {
            field: manifest[field]
            for field in (
                'schema_version', 'generator', 'generator_config', 'versions',
                'feature_names', 'dimension', 'dtype', 'semantic_revision',
            )
        }
        assert 'input' not in source['feature_manifest']
        assert 'status' not in source['feature_manifest']

    assert schemas[0]['dimension'] == schemas[1]['dimension'] == 2
    assert schemas[0]['dtype'] == schemas[1]['dtype'] == 'float32'
    assert schemas[0] != schemas[1]


@pytest.mark.parametrize(
    ('identity', 'error'),
    [
        ({}, 'exactly one'),
        (
            {'semantic_revision': 1, 'implementation_sha256': '0' * 64},
            'exactly one',
        ),
        ({'semantic_revision': 0}, 'positive integer'),
        ({'implementation_sha256': 'z' * 64}, 'hexadecimal string'),
    ],
)
def test_v2_feature_manifest_requires_one_valid_generator_identity(
    tmp_path, identity, error,
):
    feature_path = tmp_path / 'features.npz'
    manifest = {
        'schema_version': 2,
        'generator': 'test',
        'generator_config': {},
        'versions': {},
        'feature_names': ['value'],
        'dimension': 1,
        'dtype': 'float64',
        **identity,
    }

    with pytest.raises(ValueError, match=error):
        _feature_manifest_schema(
            str(feature_path), np.asarray([[1.0]]), manifest=manifest,
        )


def test_empty_save_features_archive_recovers_width_from_complete_manifest(tmp_path):
    data_path = tmp_path / 'empty.csv'
    data_path.write_text('smiles,target\n')
    feature_path = tmp_path / 'empty_features.npz'
    np.savez_compressed(feature_path, features=np.array([], dtype=np.float64))
    _write_feature_manifest(feature_path, smiles=[], dimension=3)

    dataset = get_data(
        path=str(data_path),
        features_path=[str(feature_path)],
    )

    assert len(dataset) == 0
    assert dataset._features_source_metadata['total_dimension'] == 3
    source = dataset._features_source_metadata['external_features'][0]
    assert source['dimension'] == 3
    assert source['dtype'] == 'float64'
    assert source['smiles_column_indices'] == [0]


def test_one_dimensional_nonempty_feature_array_remains_invalid(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\n')
    feature_path = tmp_path / 'features.npz'
    np.savez_compressed(feature_path, features=np.array([1.0, 2.0]))
    _write_feature_manifest(feature_path, smiles=['CC'], dimension=2)

    with pytest.raises(ValueError, match=r'2-D matrix.*shape \(2,\)'):
        get_data(path=str(data_path), features_path=[str(feature_path)])


def test_feature_manifest_rejects_same_length_reordered_smiles(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCCC,1\nCC,2\n')
    feature_path = tmp_path / 'features.npz'
    np.savez_compressed(
        feature_path,
        features=np.array([[1.0], [2.0]], dtype=np.float64),
    )
    _write_feature_manifest(feature_path, smiles=['CC', 'CCC'], dimension=1)

    with pytest.raises(
        ValueError,
        match='ordered_smiles_sha256.*SMILES values or row order changed',
    ):
        get_data(path=str(data_path), features_path=[str(feature_path)])


def test_feature_manifest_rejects_num_smiles_mismatch(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,2\n')
    feature_path = tmp_path / 'features.npz'
    np.savez_compressed(
        feature_path,
        features=np.array([[1.0], [2.0]], dtype=np.float64),
    )
    _write_feature_manifest(feature_path, smiles=['CC'], dimension=1)

    with pytest.raises(ValueError, match=r'input.num_smiles 1.*2 rows'):
        get_data(path=str(data_path), features_path=[str(feature_path)])


def test_manifest_identity_matches_each_configured_smiles_column(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('left,right,target\nCC,O,1\nCCC,N,2\n')
    left_path = tmp_path / 'left.npz'
    right_path = tmp_path / 'right.npz'
    values = np.array([[1.0], [2.0]], dtype=np.float64)
    np.savez_compressed(left_path, features=values)
    np.savez_compressed(right_path, features=values)
    _write_feature_manifest(left_path, smiles=['CC', 'CCC'], dimension=1)
    _write_feature_manifest(right_path, smiles=['O', 'N'], dimension=1)

    left_then_right = get_data(
        path=str(data_path),
        smiles_columns=['left', 'right'],
        features_path=[str(left_path), str(right_path)],
    )
    right_then_left = get_data(
        path=str(data_path),
        smiles_columns=['left', 'right'],
        features_path=[str(right_path), str(left_path)],
    )

    first_sources = left_then_right._features_source_metadata['external_features']
    second_sources = right_then_left._features_source_metadata['external_features']
    assert [source['smiles_column_indices'] for source in first_sources] == [[0], [1]]
    assert [source['smiles_column_indices'] for source in second_sources] == [[1], [0]]
    assert left_then_right._features_source_metadata != \
        right_then_left._features_source_metadata
    assert all('input' not in source['feature_manifest'] for source in first_sources)

    duplicate_data_path = tmp_path / 'duplicate_columns.csv'
    duplicate_data_path.write_text('left,right,target\nCC,CC,1\nCCC,CCC,2\n')
    duplicate_columns = get_data(
        path=str(duplicate_data_path),
        smiles_columns=['left', 'right'],
        features_path=[str(left_path)],
    )
    duplicate_source = duplicate_columns._features_source_metadata[
        'external_features'
    ][0]
    assert duplicate_source['smiles_column_indices'] == [0, 1]


def test_manifest_row_count_is_checked_against_full_csv_before_max_size(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,2\nCCCC,3\n')
    feature_path = tmp_path / 'features.npz'
    np.savez_compressed(
        feature_path,
        features=np.array([[1.0]], dtype=np.float64),
    )
    _write_feature_manifest(
        feature_path,
        smiles=['CC', 'CCC', 'CCCC'],
        dimension=1,
    )

    with pytest.raises(ValueError, match=r'has 1 rows.*3 data rows'):
        get_data(
            path=str(data_path),
            features_path=[str(feature_path)],
            max_data_size=1,
        )


def test_manifest_identity_uses_raw_rows_before_target_and_size_filtering(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,\nCCC,2\nCCCC,3\n')
    feature_path = tmp_path / 'features.npz'
    np.savez_compressed(
        feature_path,
        features=np.array([[10.0], [20.0], [30.0]], dtype=np.float64),
    )
    _write_feature_manifest(
        feature_path,
        smiles=['CC', 'CCC', 'CCCC'],
        dimension=1,
    )

    dataset = get_data(
        path=str(data_path),
        features_path=[str(feature_path)],
        skip_none_targets=True,
        max_data_size=1,
    )

    assert len(dataset) == 1
    assert dataset[0].smiles == ['CCC']
    np.testing.assert_array_equal(dataset[0].features, [20.0])


def test_manifestless_features_still_cover_raw_rows_with_max_size(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,2\nCCCC,3\n')
    feature_path = tmp_path / 'features.npy'
    np.save(feature_path, np.array([[7.0]], dtype=np.float64))

    with pytest.raises(ValueError, match=r'1 rows.*3 data rows'):
        get_data(
            path=str(data_path),
            features_path=[str(feature_path)],
            max_data_size=1,
        )


def test_manifestless_features_report_raw_row_count_mismatch(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,\nCCC,2\n')
    feature_path = tmp_path / 'features.npy'
    np.save(feature_path, np.array([[7.0]], dtype=np.float64))

    with pytest.raises(ValueError, match=r'1 rows.*2 data rows'):
        get_data(
            path=str(data_path),
            features_path=[str(feature_path)],
            skip_none_targets=True,
        )


def test_feature_source_manifest_rejects_invalid_json_and_dimension(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,2\n')
    feature_path = tmp_path / 'features.npz'
    np.savez_compressed(
        feature_path,
        features=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    manifest_path = tmp_path / (feature_path.name + '.manifest.json')

    manifest_path.write_text('{not-json')
    with pytest.raises(ValueError, match='Invalid feature manifest'):
        get_data(path=str(data_path), features_path=[str(feature_path)])

    manifest_path.write_text(json.dumps({
        'schema_version': 1,
        'generator': 'morgan',
        'dimension': 3,
        'dtype': 'float64',
    }))
    with pytest.raises(ValueError, match='dimension 3.*width 2'):
        get_data(path=str(data_path), features_path=[str(feature_path)])


def test_invalid_reaction_component_is_filtered_without_crashing():
    empty_cache()
    reset_featurization_parameters()
    set_reaction(True, 'reac_diff')
    try:
        data = get_data_from_smiles(
            [['CC>>not-a-smiles']],
            features_generator=['_test_molecule_values'],
        )
        assert len(data) == 0
    finally:
        reset_featurization_parameters()
        empty_cache()


def test_new_batch_parameters_do_not_shift_legacy_positional_arguments():
    datapoint_parameters = list(inspect.signature(MoleculeDatapoint).parameters)
    assert datapoint_parameters[10:12] == ['selected_features_path', 'phase_features']

    get_data_parameters = list(inspect.signature(get_data_function).parameters)
    assert get_data_parameters[8:10] == ['features_generator', 'phase_features_path']


def test_feature_cache_preserves_atom_order_for_custom_generators():
    empty_cache()
    data = get_data_from_smiles(
        [['CO'], ['OC']],
        features_generator=['_test_first_atom'],
        skip_invalid_smiles=False,
    )
    np.testing.assert_array_equal(data.features(), [[6.0], [8.0]])

    empty_cache()
    first = MoleculeDatapoint(['CO'], features_generator=['_test_first_atom'])
    second = MoleculeDatapoint(['OC'], features_generator=['_test_first_atom'])
    np.testing.assert_array_equal(first.features, [6.0])
    np.testing.assert_array_equal(second.features, [8.0])


def test_replacing_registered_generator_invalidates_feature_cache():
    empty_cache()

    def first_implementation(mol, selected_feature_columns=None):
        return np.array([1.0])

    def second_implementation(mol, selected_feature_columns=None):
        return np.array([2.0])

    registry = feature_generators_module.FEATURES_GENERATOR_REGISTRY
    previous = registry.get('_test_replaceable')
    try:
        register_features_generator('_test_replaceable')(first_implementation)
        first = MoleculeDatapoint(['CC'], features_generator=['_test_replaceable'])
        register_features_generator('_test_replaceable')(second_implementation)
        second = MoleculeDatapoint(['CC'], features_generator=['_test_replaceable'])

        np.testing.assert_array_equal(first.features, [1.0])
        np.testing.assert_array_equal(second.features, [2.0])
    finally:
        if previous is None:
            registry.pop('_test_replaceable', None)
        else:
            registry['_test_replaceable'] = previous


def test_feature_cache_supports_unhashable_callable_generators():
    class UnhashableGenerator:
        __hash__ = None

        def __eq__(self, other):
            return self is other

        def __call__(self, mol, selected_feature_columns=None):
            return np.array([mol.GetNumHeavyAtoms()], dtype=float)

    registry = feature_generators_module.FEATURES_GENERATOR_REGISTRY
    previous = registry.get('_test_unhashable')
    try:
        registry['_test_unhashable'] = UnhashableGenerator()
        first = MoleculeDatapoint(['CC'], features_generator=['_test_unhashable'])
        second = MoleculeDatapoint(['CC'], features_generator=['_test_unhashable'])
        np.testing.assert_array_equal(first.features, [2.0])
        np.testing.assert_array_equal(second.features, [2.0])
    finally:
        if previous is None:
            registry.pop('_test_unhashable', None)
        else:
            registry['_test_unhashable'] = previous


def test_get_data_without_args_resolves_targets_before_constraints(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,2\n')
    constraints_path = tmp_path / 'constraints.csv'
    constraints_path.write_text('target\n0.25\n0.5\n')

    data = get_data(
        path=str(data_path),
        constraints_path=str(constraints_path),
    )

    np.testing.assert_array_equal(data[0].constraints, [0.25])
    np.testing.assert_array_equal(data[1].constraints, [0.5])


def test_quantile_get_data_expands_explicit_target_columns(tmp_path):
    data_path = tmp_path / 'quantile.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,2\n')

    data = get_data(
        path=str(data_path),
        target_columns=['target'],
        loss_function='quantile_interval',
    )

    assert data.targets() == [[1.0, 1.0], [2.0, 2.0]]


@pytest.mark.parametrize('phase', [False, True])
def test_external_features_require_exact_raw_csv_row_count(tmp_path, phase):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,2\n')
    features_path = tmp_path / 'too_many.npz'
    values = np.eye(3) if phase else np.arange(6).reshape(3, 2)
    np.savez_compressed(features_path, features=values)

    kwargs = (
        {'phase_features_path': str(features_path)}
        if phase
        else {'features_path': [str(features_path)]}
    )
    with pytest.raises(ValueError, match='3 rows.*2 data rows'):
        get_data(path=str(data_path), **kwargs)


@pytest.mark.parametrize('auxiliary', ['weights', 'constraints'])
def test_rowwise_training_inputs_require_exact_raw_csv_row_count(
    tmp_path, auxiliary,
):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,2\n')
    auxiliary_path = tmp_path / f'{auxiliary}.csv'
    if auxiliary == 'weights':
        auxiliary_path.write_text('weight\n1\n2\n3\n')
        kwargs = {'data_weights_path': str(auxiliary_path)}
    else:
        auxiliary_path.write_text('target\n0.1\n0.2\n0.3\n')
        kwargs = {'constraints_path': str(auxiliary_path)}

    with pytest.raises(ValueError, match='3 rows.*2 data rows'):
        get_data(path=str(data_path), **kwargs)


def test_ordered_atom_features_follow_rows_skipped_for_missing_targets(tmp_path):
    data_path = tmp_path / 'atom_features.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,\nCO,3\n')
    atom_features_path = tmp_path / 'atom_features.npz'
    np.savez(
        atom_features_path,
        first=np.full((2, 1), 10.0),
        skipped=np.full((3, 1), 20.0),
        third=np.full((2, 1), 30.0),
    )
    args = TrainArgs().parse_args(
        [
            '--data_path', str(data_path),
            '--dataset_type', 'regression',
            '--atom_descriptors', 'feature',
            '--atom_descriptors_path', str(atom_features_path),
            '--no_cuda',
        ]
    )

    data = get_data(path=str(data_path), args=args, skip_none_targets=True)

    assert len(data) == 2
    np.testing.assert_array_equal(data[0].atom_features, np.full((2, 1), 10.0))
    np.testing.assert_array_equal(data[1].atom_features, np.full((2, 1), 30.0))


def test_pickle_atom_features_align_before_skipping_missing_targets(tmp_path):
    data_path = tmp_path / 'atom_features.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,\nCO,3\n')
    atom_features_path = tmp_path / 'atom_features.pkl'
    pd.DataFrame(
        {
            'descriptor': [
                np.full(2, 30.0),
                np.full(2, 10.0),
                np.full(3, 20.0),
            ]
        },
        index=['CO', 'CC', 'CCC'],
    ).to_pickle(atom_features_path)
    args = TrainArgs().parse_args(
        [
            '--data_path', str(data_path),
            '--dataset_type', 'regression',
            '--atom_descriptors', 'feature',
            '--atom_descriptors_path', str(atom_features_path),
            '--no_cuda',
        ]
    )

    data = get_data(path=str(data_path), args=args, skip_none_targets=True)

    assert len(data) == 2
    np.testing.assert_array_equal(data[0].atom_features, np.full((2, 1), 10.0))
    np.testing.assert_array_equal(data[1].atom_features, np.full((2, 1), 30.0))


def test_get_data_normalizes_single_feature_path_and_generator_strings(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,2\n')
    feature_path = tmp_path / 'features.npz'
    np.savez_compressed(feature_path, features=np.array([[10.0], [20.0]]))

    data = get_data(
        path=str(data_path),
        features_path=str(feature_path),
        features_generator='morgan',
    )

    assert len(data) == 2
    assert data[0].features.shape == (2049,)
    assert data[0].features[0] == 10
    assert data[1].features[0] == 20


def test_get_data_honors_zero_and_rejects_invalid_max_data_size(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,2\n')

    assert len(get_data(path=str(data_path), max_data_size=0)) == 0
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError, match='max_data_size'):
            get_data(path=str(data_path), max_data_size=invalid)


def test_get_data_rejects_an_incomplete_external_feature_manifest(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('smiles,target\nCC,1\nCCC,2\n')
    feature_path = tmp_path / 'features.npz'
    np.savez_compressed(feature_path, features=np.array([[10.0], [20.0]]))
    payload = _write_feature_manifest(
        feature_path, ['CC', 'CCC'], dimension=1,
    )
    payload['status'] = 'in_progress'
    manifest_path = feature_path.with_name(feature_path.name + '.manifest.json')
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match='only complete feature archives'):
        get_data(path=str(data_path), features_path=[str(feature_path)])
