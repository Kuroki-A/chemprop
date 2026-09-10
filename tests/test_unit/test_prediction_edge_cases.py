import csv
from collections import OrderedDict
import importlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from chemprop.data import get_data_from_smiles, MoleculeDatapoint, MoleculeDataset
from chemprop.features import get_features_generators_metadata


make_predictions_module = importlib.import_module("chemprop.train.make_predictions")
make_predictions = make_predictions_module.make_predictions
validate_prediction_feature_schema = (
    make_predictions_module.validate_prediction_feature_schema
)


def _prediction_args(preds_path, test_path=None):
    return SimpleNamespace(
        checkpoint_paths=["model.pt"],
        uncertainty_method=None,
        calibration_method=None,
        evaluation_methods=None,
        calibration_path=None,
        dataset_type="regression",
        loss_function="mse",
        preds_path=str(preds_path),
        test_path=None if test_path is None else str(test_path),
        smiles_columns=["smiles"],
        drop_extra_columns=False,
        individual_ensemble_predictions=False,
    )


def _run_without_valid_molecules(args, full_data, return_index_dict=False):
    model_result = (args, SimpleNamespace(), [], [], 1, ["target"])
    data_result = (full_data, [], None, {})
    with patch.object(
        make_predictions_module, "load_model", return_value=model_result,
    ), patch.object(make_predictions_module, "set_features"), patch.object(
        make_predictions_module, "load_data", return_value=data_result,
    ):
        return make_predictions(
            args,
            return_invalid_smiles=True,
            return_index_dict=return_index_dict,
        )


def test_ffn_prediction_writes_header_for_empty_input(tmp_path):
    test_path = tmp_path / "empty.csv"
    test_path.write_text("smiles,source\n")
    preds_path = tmp_path / "predictions.csv"

    result = _run_without_valid_molecules(
        _prediction_args(preds_path, test_path), [],
    )

    assert result == []
    with preds_path.open(newline="") as predictions_file:
        assert list(csv.reader(predictions_file)) == [["smiles", "source", "target"]]


def test_ffn_prediction_marks_all_invalid_rows_and_index_result(tmp_path):
    preds_path = tmp_path / "invalid_predictions.csv"
    args = _prediction_args(preds_path)
    datapoint = SimpleNamespace(
        row=OrderedDict([("smiles", "not-a-smiles"), ("source", "test")]),
        smiles=["not-a-smiles"],
    )

    result = _run_without_valid_molecules(args, [datapoint], return_index_dict=True)

    assert result == {0: ["Invalid SMILES"]}
    with preds_path.open(newline="") as predictions_file:
        rows = list(csv.DictReader(predictions_file))
    assert rows == [{
        "smiles": "not-a-smiles",
        "source": "test",
        "target": "Invalid SMILES",
    }]


def _empty_feature_metadata(generated_dimension):
    return {
        "schema_version": 1,
        "external_features": [],
        "phase_features": None,
        "generated_dimension": generated_dimension,
        "total_dimension": generated_dimension,
    }


def _external_feature_metadata(columns):
    return {
        "schema_version": 1,
        "external_features": [{
            "dimension": len(columns),
            "dtype": "float64",
            "csv_header": list(columns),
        }],
        "phase_features": None,
        "generated_dimension": 0,
        "total_dimension": len(columns),
    }


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("width", "Calibration feature width"),
        ("generator", "Calibration feature schema"),
        ("source", "Calibration feature sources"),
    ],
)
def test_calibration_data_uses_checkpoint_feature_schema_validation(
    tmp_path, mismatch, message,
):
    args = _prediction_args(tmp_path / "predictions.csv")
    args.uncertainty_method = "ensemble"
    args.calibration_method = "isotonic"
    args.calibration_path = "calibration.csv"
    args.calibration_features_path = ["calibration_features.npz"]
    args.calibration_phase_features_path = None
    args.calibration_atom_descriptors_path = None
    args.calibration_bond_descriptors_path = None
    args.features_generator = None
    args.selected_features_path = None
    args.max_data_size = None
    args.batch_size = 1
    args.num_workers = 0

    expected_width = 2
    actual_width = 2
    expected_generator_metadata = None
    expected_source_metadata = None
    actual_source_metadata = None
    if mismatch == "width":
        actual_width = 3
    elif mismatch == "generator":
        expected_width = actual_width = 2048
        args.features_generator = ["fcfp"]
        expected_generator_metadata = get_features_generators_metadata(
            ["morgan"], total_dimension=2048,
        )
        expected_source_metadata = actual_source_metadata = (
            _empty_feature_metadata(2048)
        )
    else:
        expected_source_metadata = _external_feature_metadata(["a", "b"])
        actual_source_metadata = _external_feature_metadata(["b", "a"])

    calibration_data = MoleculeDataset([
        MoleculeDatapoint(smiles=["CC"], features=np.zeros(actual_width)),
    ])
    calibration_data._features_source_metadata = actual_source_metadata
    train_args = SimpleNamespace(
        features_size=expected_width,
        features_generator_metadata=expected_generator_metadata,
        features_source_metadata=expected_source_metadata,
    )
    model_objects = (args, train_args, [], [], 1, ["target"])
    empty_data = MoleculeDataset([])

    with patch.object(make_predictions_module, "set_features"), patch.object(
        make_predictions_module,
        "load_data",
        return_value=(empty_data, empty_data, None, {}),
    ), patch.object(
        make_predictions_module, "get_data", return_value=calibration_data,
    ), patch.object(
        make_predictions_module, "MoleculeDataLoader",
    ) as loader_mock, patch.object(
        make_predictions_module, "build_uncertainty_calibrator",
    ) as calibrator_mock, pytest.raises(ValueError, match=message):
        make_predictions(args, model_objects=model_objects)

    loader_mock.assert_not_called()
    calibrator_mock.assert_not_called()


@pytest.mark.parametrize("smiles", [[], [["not-a-smiles"]]])
def test_calibration_schema_validation_accepts_empty_or_all_invalid(smiles):
    raw_data = get_data_from_smiles(
        smiles,
        skip_invalid_smiles=False,
        features_generator=["morgan"],
    )
    calibration_data = MoleculeDataset([])
    calibration_data._features_source_metadata = raw_data._features_source_metadata
    args = SimpleNamespace(
        features_generator=["morgan"], selected_features_path=None,
    )
    train_args = SimpleNamespace(
        features_size=2048,
        features_generator_metadata=get_features_generators_metadata(
            ["morgan"], total_dimension=2048,
        ),
        features_source_metadata=_empty_feature_metadata(2048),
    )

    validate_prediction_feature_schema(
        args,
        train_args,
        calibration_data,
        calibration_data,
        input_label="Calibration",
    )


def test_prediction_output_validation_rejects_bad_rows_and_nonfinite_values():
    validate = make_predictions_module._validate_prediction_values

    with pytest.raises(ValueError, match="row count"):
        validate([[1.0]], "Prediction", expected_rows=2)
    with pytest.raises(ValueError, match="non-finite"):
        validate([[np.nan]], "Prediction", expected_rows=1)
    with pytest.raises(ValueError, match="non-finite"):
        validate([[np.inf]], "Prediction", expected_rows=1, allow_nan=True)

    validate([[np.nan]], "Spectrum", expected_rows=1, allow_nan=True)


def test_empty_conformal_regression_has_one_uncertainty_per_task(tmp_path):
    args = SimpleNamespace(
        loss_function="mse",
        dataset_type="regression",
        uncertainty_method="conformal_regression",
        calibration_method="conformal_regression",
        conformal_alpha=0.1,
        drop_extra_columns=False,
        individual_ensemble_predictions=False,
        smiles_columns=["smiles"],
        checkpoint_paths=["model.pt"],
        preds_path=str(tmp_path / "predictions.csv"),
    )
    datapoint = SimpleNamespace(
        row=OrderedDict([("smiles", "invalid")]),
        smiles=["invalid"],
    )

    predictions, uncertainties = (
        make_predictions_module._save_no_valid_ffn_predictions(
            args=args,
            full_data=[datapoint],
            task_names=["a", "b"],
            calibrator=SimpleNamespace(label="conformal_interval"),
            return_invalid_smiles=True,
        )
    )

    assert predictions == [["Invalid SMILES", "Invalid SMILES"]]
    assert uncertainties == [["Invalid SMILES", "Invalid SMILES"]]


@pytest.mark.parametrize(
    ('atom_constraints', 'bond_constraints'),
    [([True], []), ([], [True])],
)
def test_in_memory_smiles_reject_checkpoints_with_row_aligned_constraints(
    atom_constraints, bond_constraints,
):
    args = SimpleNamespace()
    train_args = SimpleNamespace(
        atom_constraints=atom_constraints,
        bond_constraints=bond_constraints,
    )

    with pytest.raises(ValueError, match='In-memory SMILES.*constraints'):
        make_predictions_module.load_data(
            args,
            smiles=[['CC']],
            train_args=train_args,
        )


def test_prediction_input_does_not_inherit_checkpoint_training_row_weights(
    monkeypatch,
):
    class ExpectedStop(Exception):
        pass

    captured = {}

    def capture_prediction_data(**kwargs):
        captured.update(kwargs)
        raise ExpectedStop

    args = SimpleNamespace(
        test_path='prediction.csv',
        smiles_columns=['smiles'],
        drop_extra_columns=False,
        data_weights_path='training_weights.csv',
    )
    monkeypatch.setattr(
        make_predictions_module, 'get_data', capture_prediction_data,
    )

    with pytest.raises(ExpectedStop):
        make_predictions_module.load_data(args, smiles=None)

    assert captured['args'] is args
    assert captured['use_args_data_weights'] is False


def test_quantile_calibration_loads_each_observed_target_once(monkeypatch):
    class ExpectedStop(Exception):
        pass

    captured = {}

    def capture_calibration_data(**kwargs):
        captured.update(kwargs)
        raise ExpectedStop

    args = SimpleNamespace(
        checkpoint_paths=['model.pt'],
        uncertainty_method=None,
        calibration_method='conformal_quantile_regression',
        calibration_path='calibration.csv',
        calibration_features_path=None,
        calibration_phase_features_path=None,
        calibration_atom_descriptors_path=None,
        calibration_bond_descriptors_path=None,
        calibration_constraints_path=None,
        smiles_columns=['smiles'],
        features_generator=None,
        max_data_size=None,
        dataset_type='regression',
        loss_function='quantile_interval',
        evaluation_methods=['conformal_coverage'],
    )
    train_args = SimpleNamespace(is_atom_bond_targets=False)
    model_objects = (
        args,
        train_args,
        [],
        [],
        4,
        ['target_a', 'target_b', 'target_a', 'target_b'],
    )

    monkeypatch.setattr(make_predictions_module, 'set_features', lambda *_args: None)
    monkeypatch.setattr(
        make_predictions_module,
        'load_data',
        lambda *_args, **_kwargs: (None, None, None, {}),
    )
    monkeypatch.setattr(
        make_predictions_module, 'get_data', capture_calibration_data,
    )

    with pytest.raises(ExpectedStop):
        make_predictions(args, model_objects=model_objects)

    assert captured['target_columns'] == ['target_a', 'target_b']
    assert captured['expand_quantile_targets'] is False


def test_uncertainty_evaluation_does_not_inherit_training_row_weights(
    monkeypatch,
):
    class ExpectedStop(Exception):
        pass

    class DummyEstimator:
        def __init__(self, **_kwargs):
            pass

        def calculate_uncertainty(self, calibrator=None):
            return [[0.5]], [[0.1]]

    captured = {}

    def capture_evaluation_data(**kwargs):
        captured.update(kwargs)
        raise ExpectedStop

    args = SimpleNamespace(
        uncertainty_method='mve',
        dataset_type='regression',
        loss_function='mve',
        uncertainty_dropout_p=0.0,
        conformal_alpha=0.1,
        dropout_sampling_size=2,
        individual_ensemble_predictions=False,
        is_atom_bond_targets=False,
        calibration_method=None,
        evaluation_methods=['nll'],
        test_path='evaluation.csv',
        smiles_columns=['smiles'],
        features_path=None,
        features_generator=None,
        phase_features_path=None,
        atom_descriptors_path=None,
        bond_descriptors_path=None,
        max_data_size=None,
        data_weights_path='training_weights.csv',
    )
    data = MoleculeDataset([
        MoleculeDatapoint(smiles=['CC'], targets=[1.0]),
    ])
    monkeypatch.setattr(
        make_predictions_module, 'UncertaintyEstimator', DummyEstimator,
    )
    monkeypatch.setattr(
        make_predictions_module, 'get_data', capture_evaluation_data,
    )

    with pytest.raises(ExpectedStop):
        make_predictions_module.predict_and_save(
            args=args,
            train_args=SimpleNamespace(spectra_phase_mask=None),
            test_data=data,
            task_names=['target'],
            num_tasks=1,
            test_data_loader=object(),
            full_data=data,
            full_to_valid_indices={0: 0},
            models=[],
            scalers=[],
            num_models=1,
            save_results=False,
        )

    assert captured['args'] is args
    assert captured['use_args_data_weights'] is False
    assert captured['expand_quantile_targets'] is False


def test_calibration_constraints_are_loaded_with_checkpoint_task_columns(monkeypatch):
    class ExpectedStop(Exception):
        pass

    captured = {}

    def capture_calibration_data(**kwargs):
        captured.update(kwargs)
        raise ExpectedStop

    args = SimpleNamespace(
        checkpoint_paths=['model.pt'],
        uncertainty_method='mve',
        calibration_method='zscaling',
        calibration_path='calibration.csv',
        calibration_features_path=None,
        calibration_phase_features_path=None,
        calibration_atom_descriptors_path=None,
        calibration_bond_descriptors_path=None,
        calibration_constraints_path='calibration_constraints.csv',
        smiles_columns=['smiles'],
        features_generator=None,
        max_data_size=None,
        dataset_type='regression',
        loss_function='mve',
        evaluation_methods=None,
    )
    train_args = SimpleNamespace(
        is_atom_bond_targets=True,
        atom_targets=['atom_a'],
        bond_targets=['bond_b'],
    )
    model_objects = (args, train_args, [], [], 2, ['atom_a', 'bond_b'])

    monkeypatch.setattr(make_predictions_module, 'set_features', lambda *_args: None)
    monkeypatch.setattr(
        make_predictions_module,
        'load_data',
        lambda *_args, **_kwargs: (None, None, None, {}),
    )
    monkeypatch.setattr(make_predictions_module, 'get_data', capture_calibration_data)

    with pytest.raises(ExpectedStop):
        make_predictions(args, model_objects=model_objects)

    assert captured['constraints_path'] == 'calibration_constraints.csv'
    assert captured['constraints_target_columns'] == ['atom_a', 'bond_b']
    assert captured['use_args_data_weights'] is False


@pytest.mark.parametrize(
    'value',
    [
        np.array([1.25, 2.5]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        [5.0, 6.0, 7.0],
    ],
)
def test_atom_bond_prediction_csv_cells_round_trip_as_json(value):
    encoded = make_predictions_module._prediction_csv_cell(
        value, is_atom_bond_targets=True,
    )

    assert json.loads(encoded) == np.asarray(value).tolist()
