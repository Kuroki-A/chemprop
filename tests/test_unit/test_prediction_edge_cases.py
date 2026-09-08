import csv
from collections import OrderedDict
import importlib
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
