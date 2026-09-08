import csv

import numpy as np
from tqdm import tqdm

from chemprop.args import SklearnPredictArgs, SklearnTrainArgs
from chemprop.data import get_data, load_selected_feature_columns
from chemprop.features import (
    get_features_generator,
    get_features_generators_metadata,
)
from chemprop.sklearn_train import load_sklearn_checkpoint, predict
from chemprop.utils import makedirs, timeit


def _feature_source_metadata_matches(expected: dict, actual: dict) -> bool:
    """Compares feature layouts while permitting unknown widths for empty input."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return expected == actual
    normalized_actual = dict(actual)
    for field in ('generated_dimension', 'total_dimension'):
        if normalized_actual.get(field) is None:
            normalized_actual[field] = expected.get(field)
    return normalized_actual == expected


@timeit()
def predict_sklearn(args: SklearnPredictArgs) -> None:
    """
    Loads data and a trained scikit-learn model and uses the model to make predictions on the data.

   :param args: A :class:`~chemprop.args.SklearnPredictArgs` object containing arguments for
                 loading data, loading a trained scikit-learn model, and making predictions with the model.
    """
    print('Loading training arguments')
    if not args.checkpoint_paths:
        raise ValueError('At least one sklearn checkpoint is required for prediction.')
    bundles = [load_sklearn_checkpoint(path) for path in args.checkpoint_paths]
    reference_args = bundles[0].train_args
    compatibility_fields = (
        'model_type', 'dataset_type', 'task_names', 'num_tasks', 'radius',
        'num_bits', 'number_of_molecules', 'features_size',
        'features_generator', 'features_generator_metadata',
        'features_source_metadata',
    )
    for checkpoint_index, bundle in enumerate(bundles[1:], start=1):
        incompatible = [
            field for field in compatibility_fields
            if bundle.train_args.get(field) != reference_args.get(field)
        ]
        if incompatible:
            raise ValueError(
                f'Sklearn ensemble checkpoint {checkpoint_index} is incompatible '
                f'with checkpoint 0: {", ".join(incompatible)}.'
            )

    train_args = SklearnTrainArgs()
    train_args.from_dict(reference_args, skip_unsettable=True)

    if args.number_of_molecules != train_args.number_of_molecules:
        raise ValueError(
            'Prediction molecule-column count does not match the sklearn checkpoint.'
        )
    checkpoint_generators = list(train_args.features_generator or [])
    if (
        args.features_generator is not None
        and list(args.features_generator) != checkpoint_generators
    ):
        raise ValueError(
            'Prediction feature generators do not match the sklearn checkpoint.'
        )
    selected_features_path = (
        args.selected_features_path
        if args.selected_features_path is not None
        else train_args.selected_features_path
    )

    print('Loading data')
    data = get_data(
        path=args.test_path,
        features_path=args.features_path,
        features_generator=checkpoint_generators or None,
        selected_features_path=selected_features_path,
        phase_features_path=args.phase_features_path,
        smiles_columns=args.smiles_columns,
        target_columns=[],
        ignore_columns=[],
        store_row=True,
    )

    actual_features_size = data.features_size() if len(data) > 0 else None
    expected_features_size = reference_args.get('features_size')
    if (
        actual_features_size is not None
        and expected_features_size is not None
        and actual_features_size != expected_features_size
    ):
        raise ValueError(
            'Prediction feature width does not match the sklearn checkpoint: '
            f'generated {actual_features_size}, expected {expected_features_size}.'
        )

    expected_generator_metadata = reference_args.get('features_generator_metadata')
    if expected_generator_metadata is not None:
        selected_feature_columns = (
            load_selected_feature_columns(selected_features_path)
            if selected_features_path is not None
            else {}
        )
        actual_generator_metadata = get_features_generators_metadata(
            checkpoint_generators,
            selected_feature_columns=selected_feature_columns,
            total_dimension=(
                actual_features_size
                if actual_features_size is not None
                else expected_generator_metadata.get('total_dimension')
            ),
        )
        if actual_generator_metadata != expected_generator_metadata:
            raise ValueError(
                'Prediction feature schema does not match the sklearn checkpoint.'
            )

    expected_source_metadata = reference_args.get('features_source_metadata')
    actual_source_metadata = getattr(data, '_features_source_metadata', None)
    if (
        expected_source_metadata is not None
        and not _feature_source_metadata_matches(
            expected_source_metadata, actual_source_metadata
        )
    ):
        raise ValueError(
            'Prediction feature sources do not match the sklearn checkpoint.'
        )

    print('Computing morgan fingerprints')
    morgan_fingerprint = get_features_generator('morgan')
    for datapoint in tqdm(data, total=len(data)):
        for s in datapoint.smiles:
            datapoint.extend_features(morgan_fingerprint(mol=s, radius=train_args.radius, num_bits=train_args.num_bits))

    print(f'Predicting with an ensemble of {len(bundles)} models')
    sum_preds = np.zeros((len(data), train_args.num_tasks))

    for bundle in tqdm(bundles, total=len(bundles)):
        model_preds = predict(
            model=bundle,
            model_type=train_args.model_type,
            dataset_type=train_args.dataset_type,
            features=data.features()
        )
        if len(data) > 0:
            model_preds = np.asarray(model_preds, dtype=float)
            if model_preds.shape != sum_preds.shape:
                raise ValueError(
                    'Sklearn checkpoint prediction shape does not match its training metadata.'
                )
            sum_preds += model_preds

    # Ensemble predictions
    avg_preds = sum_preds / len(bundles)
    avg_preds = avg_preds.tolist()

    print(f'Saving predictions to {args.preds_path}')
    makedirs(args.preds_path, isfile=True)

    # Copy predictions over to data
    for datapoint, preds in zip(data, avg_preds):
        for pred_name, pred in zip(train_args.task_names, preds):
            datapoint.row[pred_name] = pred

    fieldnames = (
        list(data[0].row.keys())
        if len(data) > 0
        else list(args.smiles_columns) + list(train_args.task_names)
    )

    # Save, including a header-only file for empty input.
    with open(args.preds_path, 'w', newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for datapoint in data:
            writer.writerow(datapoint.row)


def sklearn_predict() -> None:
    """Parses scikit-learn predicting arguments and runs prediction using a trained scikit-learn model.

    This is the entry point for the command line command :code:`sklearn_predict`.
    """
    predict_sklearn(args=SklearnPredictArgs().parse_args())
