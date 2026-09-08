import json
from types import SimpleNamespace

import numpy as np
import pytest

from chemprop.features import load_features
from scripts import save_features as save_features_script


def _schema(_name, feature_vector=None, selected_feature_columns=None):
    dtype = str(np.asarray(feature_vector).dtype) if feature_vector is not None else 'float32'
    return {
        'schema_version': 1,
        'generator': 'streaming_test',
        'implementation_sha256': 'a' * 64,
        'generator_config': {},
        'versions': {},
        'dimension': 1,
        'dtype': dtype,
        'feature_names': ['value'],
    }


def _args(tmp_path, row_count, save_frequency=17):
    data_path = tmp_path / 'molecules.csv'
    data_path.write_text(
        'smiles\n' + ''.join(f'{index}\n' for index in range(row_count)),
        encoding='utf-8',
    )
    return SimpleNamespace(
        data_path=str(data_path),
        smiles_column='smiles',
        features_generator='streaming_test',
        selected_features_path=None,
        save_path=str(tmp_path / 'features.npz'),
        save_frequency=save_frequency,
        restart=False,
        sequential=True,
        num_workers=None,
        chunksize=1,
        batch_size=None,
    )


def _install_generator(monkeypatch):
    monkeypatch.setattr(
        save_features_script,
        'get_features_generator_schema',
        _schema,
    )
    monkeypatch.setattr(
        save_features_script,
        'get_features_generator',
        lambda _name: lambda value: np.asarray([int(value)], dtype=np.float32),
    )


def test_streaming_generation_never_passes_all_rows_to_save_features(
    tmp_path, monkeypatch,
):
    args = _args(tmp_path, row_count=1003, save_frequency=17)
    _install_generator(monkeypatch)
    original_save_features = save_features_script.save_features
    persisted_chunk_sizes = []

    def record_bounded_chunk(path, rows):
        persisted_chunk_sizes.append(len(rows))
        return original_save_features(path, rows)

    monkeypatch.setattr(save_features_script, 'save_features', record_bounded_chunk)
    monkeypatch.setattr(
        save_features_script,
        'load_temp',
        lambda _path: pytest.fail('generate_and_save_features must not materialize load_temp'),
    )

    save_features_script.generate_and_save_features(args)

    assert sum(persisted_chunk_sizes) == 1003
    assert max(persisted_chunk_sizes) <= 17
    result = load_features(args.save_path)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[:, 0], np.arange(1003, dtype=np.float32))
    manifest = json.loads((tmp_path / 'features.npz.manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['storage'] == 'npz'
    assert manifest['num_molecules_completed'] == 1003


def test_streaming_empty_input_preserves_historical_archive_shape(
    tmp_path, monkeypatch,
):
    args = _args(tmp_path, row_count=0)
    _install_generator(monkeypatch)

    save_features_script.generate_and_save_features(args)

    result = load_features(args.save_path)
    assert result.shape == (0,)
    assert result.dtype == np.float32


def test_explicit_worker_count_can_override_a_pseudo_batch_generator():
    generator = lambda value: np.asarray([value])
    generator.batch_transform = lambda values: values
    generator.prefer_process_pool_when_requested = True
    args = SimpleNamespace(sequential=False, num_workers=None)

    assert not save_features_script._prefer_requested_process_pool(args, generator)
    args.num_workers = 1
    assert not save_features_script._prefer_requested_process_pool(args, generator)
    args.num_workers = 4
    assert save_features_script._prefer_requested_process_pool(args, generator)
    args.sequential = True
    assert not save_features_script._prefer_requested_process_pool(args, generator)


def test_resume_without_identity_manifest_is_rejected(tmp_path, monkeypatch):
    args = _args(tmp_path, row_count=2, save_frequency=1)
    _install_generator(monkeypatch)
    temp_dir = tmp_path / 'features.npz_temp'
    temp_dir.mkdir()
    save_features_script._atomic_save_feature_chunk(
        str(temp_dir / '0.npz'),
        [np.asarray([999], dtype=np.float32)],
    )

    with pytest.raises(ValueError, match='Cannot safely resume.*without.*manifest'):
        save_features_script.generate_and_save_features(args)


@pytest.mark.parametrize(
    ('sequential', 'num_workers'),
    [(True, None), (False, 2)],
    ids=['sequential', 'process-pool'],
)
def test_save_features_applies_selected_fixed_fingerprint_columns(
    tmp_path, sequential, num_workers,
):
    data_path = tmp_path / 'selected.csv'
    data_path.write_text('smiles\nCCO\nCC\n')
    selected_path = tmp_path / 'selected_features.csv'
    selected_path.write_text('morgan\nbit_17\nbit_3\nbit_17\n')
    args = SimpleNamespace(
        data_path=str(data_path),
        smiles_column='smiles',
        features_generator='morgan',
        selected_features_path=str(selected_path),
        save_path=str(tmp_path / 'selected_features.npz'),
        save_frequency=10,
        restart=False,
        sequential=sequential,
        num_workers=num_workers,
        chunksize=1,
        batch_size=None,
    )

    save_features_script.generate_and_save_features(args)

    values = load_features(args.save_path)
    assert values.shape == (2, 3)
    assert values[:, 0].tolist() == values[:, 2].tolist()
    manifest = json.loads(
        (tmp_path / 'selected_features.npz.manifest.json').read_text()
    )
    assert manifest['feature_names'] == ['bit_17', 'bit_3', 'bit_17']
    assert manifest['generator_config']['selected_feature_columns'] == [
        'bit_17', 'bit_3', 'bit_17',
    ]


def test_streaming_resume_adopts_atomic_chunk_newer_than_manifest(
    tmp_path, monkeypatch,
):
    args = _args(tmp_path, row_count=6, save_frequency=2)
    _install_generator(monkeypatch)
    temp_dir = args.save_path + '_temp'
    save_features_script.makedirs(temp_dir)
    first = [np.asarray([0], dtype=np.float32)]
    second = [np.asarray([1], dtype=np.float32)]
    save_features_script._atomic_save_feature_chunk(
        f'{temp_dir}/0.npz', first,
    )
    save_features_script._atomic_save_feature_chunk(
        f'{temp_dir}/1.npz', second,
    )
    identity = save_features_script._input_identity(
        args.data_path, [str(index) for index in range(6)],
    )
    # Simulate interruption after chunk 1 was atomically published but before
    # its manifest update. The scanner can safely adopt that complete chunk.
    save_features_script._save_manifest(
        args,
        first,
        storage='chunk_directory',
        temporary_file_count=1,
        input_identity=identity,
        total_molecules=6,
        status='in_progress',
    )

    save_features_script.generate_and_save_features(args)

    result = load_features(args.save_path)
    np.testing.assert_array_equal(result[:, 0], np.arange(6, dtype=np.float32))
    assert not (tmp_path / 'features.npz_temp').exists()


def test_atomic_consolidation_failure_preserves_existing_output(
    tmp_path, monkeypatch,
):
    temp_dir = tmp_path / 'chunks'
    temp_dir.mkdir()
    save_features_script._atomic_save_feature_chunk(
        str(temp_dir / '0.npz'),
        [np.asarray([1], dtype=np.float32)],
    )
    output_path = tmp_path / 'features.npz'
    output_path.write_bytes(b'previous-complete-output')

    def fail_write(*_args, **_kwargs):
        raise RuntimeError('simulated archive failure')

    monkeypatch.setattr(save_features_script.zipfile.ZipFile, 'write', fail_write)
    with pytest.raises(RuntimeError, match='simulated archive failure'):
        save_features_script._atomic_consolidate_chunks(
            str(output_path),
            str(temp_dir),
            temporary_file_count=1,
            total_rows=1,
            dimension=1,
            dtype='float32',
        )

    assert output_path.read_bytes() == b'previous-complete-output'
    assert sorted(path.name for path in tmp_path.iterdir()) == ['chunks', 'features.npz']
