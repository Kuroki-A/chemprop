from collections import OrderedDict
import csv
import os
from typing import List, Optional, Union, Tuple

import numpy as np

from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, get_header, is_valid_datapoint, load_selected_feature_columns, MoleculeDataLoader, MoleculeDataset, StandardScaler, AtomBondScaler
from chemprop.utils import (
    LightGBMModelBundle,
    load_args,
    load_checkpoint,
    load_checkpoint_lgbm,
    load_scalers,
    makedirs,
    timeit,
    update_prediction_args,
)
from chemprop.features import get_features_generators_metadata, set_extra_atom_fdim, set_extra_bond_fdim, set_reaction, set_explicit_h, set_adding_hs, set_keeping_atom_map, reset_featurization_parameters
from chemprop.models import MoleculeModel
from chemprop.uncertainty import UncertaintyCalibrator, build_uncertainty_calibrator, UncertaintyEstimator, build_uncertainty_evaluator
from chemprop.multitask_utils import reshape_values


_LGBM_ENSEMBLE_COMPATIBILITY_FIELDS = (
    "dataset_type",
    "number_of_molecules",
    "reaction",
    "reaction_solvent",
    "reaction_mode",
    "explicit_h",
    "adding_h",
    "keeping_atom_map",
    "features_generator",
    "features_generator_metadata",
    "features_source_metadata",
    "features_size",
    "features_only",
    "use_input_features",
    "atom_descriptors",
    "atom_descriptors_size",
    "atom_features_size",
    "bond_descriptors",
    "bond_descriptors_size",
    "bond_features_size",
    "overwrite_default_atom_features",
    "overwrite_default_bond_features",
    "atom_messages",
    "hidden_size",
    "hidden_size_solvent",
    "bias",
    "bias_solvent",
    "depth",
    "depth_solvent",
    "undirected",
    "aggregation",
    "aggregation_norm",
    "activation",
    "mpn_shared",
)

_FFN_ENSEMBLE_COMPATIBILITY_FIELDS = tuple(dict.fromkeys(
    _LGBM_ENSEMBLE_COMPATIBILITY_FIELDS + (
        "model_type",
        "task_names",
        "num_tasks",
        "loss_function",
        "multiclass_num_classes",
        "ffn_hidden_size",
        "ffn_num_layers",
        "dropout",
        "features_scaling",
        "atom_descriptor_scaling",
        "bond_descriptor_scaling",
        "is_atom_bond_targets",
        "atom_targets",
        "bond_targets",
        "atom_constraints",
        "bond_constraints",
        "shared_atom_bond_ffn",
        "adding_bond_types",
        "weights_ffn_num_layers",
        "spectra_activation",
        "spectra_phase_mask",
        "quantile_loss_alpha",
        "quantiles",
    )
))


def _validate_prediction_values(
    values, label: str, expected_rows: int, allow_nan: bool = False
) -> None:
    """Validates prediction row counts and rejects invalid numeric output."""
    try:
        actual_rows = len(values)
    except TypeError as error:
        raise ValueError(f'{label} output must contain one row per valid input.') from error
    if actual_rows != expected_rows:
        raise ValueError(
            f'{label} output row count ({actual_rows}) does not match the '
            f'number of valid inputs ({expected_rows}).'
        )

    pending = [values]
    while pending:
        value = pending.pop()
        if isinstance(value, np.ndarray):
            pending.extend(value.flat)
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        else:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f'{label} output must contain numeric values.') from error
            if np.isinf(numeric_value) or (np.isnan(numeric_value) and not allow_nan):
                raise ValueError(f'{label} output contains non-finite values.')


def _lgbm_compatibility_signature(train_args: TrainArgs) -> dict:
    """Captures every setting that changes encoder inputs or architecture."""
    return {
        field: getattr(train_args, field, None)
        for field in _LGBM_ENSEMBLE_COMPATIBILITY_FIELDS
    }


def _validate_ensemble_train_args(
    train_args_list: List[TrainArgs],
    fields: Tuple[str, ...],
    model_label: str,
) -> None:
    """Rejects checkpoints which cannot safely share one prediction dataset."""
    if not train_args_list:
        raise ValueError(f"No {model_label} checkpoint arguments were loaded.")

    def values_equal(expected, actual) -> bool:
        """Compares ordinary checkpoint values and array-valued schemas."""
        if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
            try:
                return np.array_equal(
                    np.asarray(expected), np.asarray(actual), equal_nan=True,
                )
            except TypeError:
                # ``equal_nan`` is unsupported for some non-numeric arrays.
                return np.array_equal(np.asarray(expected), np.asarray(actual))
        return expected == actual

    reference = {
        field: getattr(train_args_list[0], field, None) for field in fields
    }
    for checkpoint_index, candidate in enumerate(train_args_list[1:], start=1):
        incompatible = [
            field for field, expected in reference.items()
            if not values_equal(expected, getattr(candidate, field, None))
        ]
        if incompatible:
            raise ValueError(
                f"{model_label} ensemble checkpoint {checkpoint_index} is incompatible "
                f"with checkpoint 0: {', '.join(incompatible)}."
            )


def _feature_source_metadata_matches(expected: dict, actual: dict) -> bool:
    """Compares source schemas while allowing unknown widths for empty API input.

    Generator identity and configuration are validated separately. An empty
    in-memory ``smiles=[]`` call has no molecule-column count from which to
    realize the generated width, but its external and phase layouts remain
    fully checkable.
    """
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return expected == actual
    normalized_actual = dict(actual)
    for field in ("generated_dimension", "total_dimension"):
        if normalized_actual.get(field) is None:
            normalized_actual[field] = expected.get(field)
    return normalized_actual == expected


def validate_prediction_feature_schema(
    args: PredictArgs,
    train_args: TrainArgs,
    full_data: MoleculeDataset,
    valid_data: MoleculeDataset,
    input_label: str = "Prediction",
) -> None:
    """Validates the complete molecular-feature schema for a model input.

    ``full_data`` retains source metadata even when invalid molecules were
    filtered from ``valid_data``. A realized row is required only for the width
    check; generator and source metadata remain checkable for empty inputs.
    """
    expected_features_size = getattr(train_args, 'features_size', None)
    actual_features_size = valid_data.features_size() if len(valid_data) > 0 else None
    if (
        actual_features_size is not None
        and expected_features_size is not None
        and actual_features_size != expected_features_size
    ):
        raise ValueError(
            f'{input_label} feature width does not match the checkpoint: '
            f'generated {actual_features_size}, expected {expected_features_size}. '
            'Use the same feature generators, selected-feature file, and '
            'dependency versions as training. Legacy reaction checkpoints '
            'trained with selected features may need to be retrained.'
        )

    expected_metadata = getattr(train_args, 'features_generator_metadata', None)
    if expected_metadata is not None:
        selected_features_path = getattr(args, 'selected_features_path', None)
        selected_feature_columns = (
            load_selected_feature_columns(selected_features_path)
            if selected_features_path is not None
            else {}
        )
        metadata_dimension = (
            actual_features_size
            if actual_features_size is not None
            else expected_metadata.get('total_dimension')
        )
        actual_metadata = get_features_generators_metadata(
            getattr(args, 'features_generator', None) or [],
            selected_feature_columns=selected_feature_columns,
            total_dimension=metadata_dimension,
        )
        if actual_metadata != expected_metadata:
            changed = sorted(
                key for key in set(actual_metadata).union(expected_metadata)
                if actual_metadata.get(key) != expected_metadata.get(key)
            )
            raise ValueError(
                f'{input_label} feature schema does not match the checkpoint '
                f'({", ".join(changed)} changed). Use the same ordered '
                'generators, selected columns, and dependency versions as training.'
            )

    expected_source_metadata = getattr(train_args, 'features_source_metadata', None)
    actual_source_metadata = getattr(full_data, '_features_source_metadata', None)
    if (
        expected_source_metadata is not None
        and not _feature_source_metadata_matches(
            expected_source_metadata, actual_source_metadata
        )
    ):
        raise ValueError(
            f'{input_label} feature sources do not match the checkpoint. Use the same '
            'ordered external feature widths/columns, phase-feature width, and '
            'generated-feature layout as training.'
        )


def load_model(args: PredictArgs, generator: bool = False):
    """
    Function to load a model or ensemble of models from file. If generator is True, a generator of the respective model and scaler 
    objects is returned (memory efficient), else the full list (holding all models in memory, necessary for preloading).

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param generator: A boolean to return a generator instead of a list of models and scalers.
    :return: A tuple of updated prediction arguments, training arguments, a list or generator object of models, a list or 
                 generator object of scalers, the number of tasks and their respective names.
    """
    print('Loading training args')
    checkpoint_train_args = [load_args(path) for path in args.checkpoint_paths]
    _validate_ensemble_train_args(
        checkpoint_train_args,
        _FFN_ENSEMBLE_COMPATIBILITY_FIELDS,
        "FFN",
    )
    train_args = checkpoint_train_args[0]
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    update_prediction_args(predict_args=args, train_args=train_args)
    args: Union[PredictArgs, TrainArgs]

    # Load model and scalers
    models = (
        load_checkpoint(checkpoint_path, device=args.device) for checkpoint_path in args.checkpoint_paths
    )
    scalers = (
        load_scalers(checkpoint_path) for checkpoint_path in args.checkpoint_paths
    )
    if not generator:
        models = list(models)
        scalers = list(scalers)

    return args, train_args, models, scalers, num_tasks, task_names

def load_model_lgbm(args: PredictArgs, generator: bool = False):
    """Loads versioned LightGBM bundles and validates ensemble compatibility."""
    print('Loading training args')
    checkpoint_paths = sorted(os.path.abspath(path) for path in args.checkpoint_paths)
    if not checkpoint_paths:
        raise ValueError("No LightGBM bundle checkpoints were provided.")

    bundles = [load_checkpoint_lgbm(path, device=args.device) for path in checkpoint_paths]
    bundles.sort(key=lambda bundle: (bundle.model_index, bundle.checkpoint_path))
    for bundle in bundles:
        _require_safe_lgbm_representation(bundle)
    first_bundle = bundles[0]
    _validate_ensemble_train_args(
        [bundle.train_args for bundle in bundles],
        _LGBM_ENSEMBLE_COMPATIBILITY_FIELDS,
        "LightGBM",
    )
    train_args = first_bundle.train_args
    num_tasks, task_names = train_args.num_tasks, list(first_bundle.task_names)
    reference_signature = _lgbm_compatibility_signature(train_args)

    encoder_groups = [(first_bundle.encoder.encoder.state_dict(), first_bundle.encoder)]
    for bundle in bundles[1:]:
        if bundle.dataset_type != first_bundle.dataset_type or bundle.task_names != task_names:
            raise ValueError(
                "All LightGBM ensemble bundles must have the same dataset type and task names."
            )
        bundle_signature = _lgbm_compatibility_signature(bundle.train_args)
        incompatible_fields = [
            field
            for field, reference_value in reference_signature.items()
            if bundle_signature[field] != reference_value
        ]
        if incompatible_fields:
            raise ValueError(
                "LightGBM ensemble bundles have incompatible encoder/feature settings: "
                + ", ".join(incompatible_fields)
                + "."
            )
        bundle_state = bundle.encoder.encoder.state_dict()
        for group_state, group_encoder in encoder_groups:
            same_encoder = bundle_state.keys() == group_state.keys() and all(
                np.array_equal(
                    bundle_state[name].detach().cpu().numpy(),
                    group_state[name].detach().cpu().numpy(),
                )
                for name in group_state
            )
            if same_encoder:
                # A fold's ensemble shares one frozen encoder in memory as well as on disk.
                bundle.encoder = group_encoder
                break
        else:
            encoder_groups.append((bundle_state, bundle.encoder))

    update_prediction_args(predict_args=args, train_args=train_args)
    args: Union[PredictArgs, TrainArgs]

    scalers = [bundle.scalers for bundle in bundles]
    models = bundles
    if generator:
        models = iter(models)
        scalers = iter(scalers)

    return args, train_args, models, scalers, num_tasks, task_names


def load_data(
    args: PredictArgs,
    smiles: List[List[str]],
    train_args: TrainArgs = None,
):
    """
    Function to load data from a list of smiles or a file.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: A list of list of smiles, or None if data is to be read from file
    :return: A tuple of a :class:`~chemprop.data.MoleculeDataset` containing all datapoints, a :class:`~chemprop.data.MoleculeDataset` containing only valid datapoints,
                 a :class:`~chemprop.data.MoleculeDataLoader` and a dictionary mapping full to valid indices.
    """
    print("Loading data")
    if smiles is not None:
        full_data = get_data_from_smiles(
            smiles=smiles,
            skip_invalid_smiles=False,
            features_generator=args.features_generator,
            selected_features_path=args.selected_features_path,
        )
    else:
        full_data = get_data(
            path=args.test_path,
            smiles_columns=args.smiles_columns,
            target_columns=[],
            ignore_columns=[],
            skip_invalid_smiles=False,
            args=args,
            store_row=not args.drop_extra_columns,
        )

    print("Validating SMILES")
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if is_valid_datapoint(full_data[full_index]):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset(
        [full_data[i] for i in sorted(full_to_valid_indices.keys())]
    )

    validate_prediction_feature_schema(
        args=args,
        train_args=train_args,
        full_data=full_data,
        valid_data=test_data,
    )

    print(f"Test size = {len(test_data):,}")

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data, batch_size=args.batch_size, num_workers=args.num_workers
    )

    return full_data, test_data, test_data_loader, full_to_valid_indices


def set_features(args: PredictArgs, train_args: TrainArgs):
    """
    Function to set extra options.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param train_args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    """
    reset_featurization_parameters()

    if args.atom_descriptors == "feature":
        set_extra_atom_fdim(train_args.atom_features_size)

    if args.bond_descriptors == "feature":
        set_extra_bond_fdim(train_args.bond_features_size)

    # set explicit H option and reaction option
    set_explicit_h(train_args.explicit_h)
    set_adding_hs(args.adding_h)
    set_keeping_atom_map(args.keeping_atom_map)
    if train_args.reaction:
        set_reaction(train_args.reaction, train_args.reaction_mode)
    elif train_args.reaction_solvent:
        set_reaction(True, train_args.reaction_mode)


_UNCERTAINTY_OUTPUT_LABELS = {
    None: "no_uncertainty_method",
    "mve": "mve_uncal_var",
    "ensemble": "ensemble_uncal_var",
    "classification": "classification_uncal_confidence",
    "evidential_total": "evidential_total_uncal_var",
    "evidential_epistemic": "evidential_epistemic_uncal_var",
    "evidential_aleatoric": "evidential_aleatoric_uncal_var",
    "dropout": "dropout_uncal_var",
    "spectra_roundrobin": "roundrobin_sid",
    "dirichlet": "dirichlet_uncal_uncertainty",
    "conformal_quantile_regression": "no_uncertainty_method",
    "conformal_regression": "no_uncertainty_method",
}


def _save_no_valid_ffn_predictions(
    args: PredictArgs,
    full_data: MoleculeDataset,
    task_names: List[str],
    calibrator: UncertaintyCalibrator = None,
    return_invalid_smiles: bool = False,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Writes deterministic output when an FFN input has no valid molecules."""
    output_task_names = list(task_names)
    if args.loss_function == "quantile_interval":
        output_task_names = output_task_names[:len(output_task_names) // 2]
    if args.dataset_type == "multiclass":
        output_task_names = [
            f"{name}_class_{class_index}"
            for name in output_task_names
            for class_index in range(args.multiclass_num_classes)
        ]

    num_prediction_tasks = len(output_task_names)
    if args.uncertainty_method == "spectra_roundrobin":
        num_uncertainty_tasks = 1
    elif args.uncertainty_method == "dirichlet" and args.dataset_type == "multiclass":
        num_uncertainty_tasks = (
            num_prediction_tasks // args.multiclass_num_classes
        )
    elif args.calibration_method == "conformal" and args.dataset_type == "classification":
        num_uncertainty_tasks = 2 * num_prediction_tasks
    else:
        num_uncertainty_tasks = num_prediction_tasks

    uncertainty_label = (
        calibrator.label
        if calibrator is not None
        else _UNCERTAINTY_OUTPUT_LABELS.get(
            args.uncertainty_method, str(args.uncertainty_method),
        )
    )
    if args.uncertainty_method == "spectra_roundrobin":
        uncertainty_names = [uncertainty_label]
    elif (
        args.uncertainty_method == "conformal_quantile_regression"
        and args.calibration_method is None
    ):
        uncertainty_names = [
            f"{name}_{args.conformal_alpha}_half_interval"
            for name in output_task_names
        ]
    elif (
        args.calibration_method == "conformal_regression"
        and calibrator is None
    ):
        uncertainty_names = []
    elif args.calibration_method == "conformal" and args.dataset_type == "classification":
        uncertainty_names = [
            f"{name}_{uncertainty_label}_in_set" for name in output_task_names
        ] + [
            f"{name}_{uncertainty_label}_out_set" for name in output_task_names
        ]
    else:
        uncertainty_names = [
            f"{name}_{uncertainty_label}" for name in output_task_names
        ]
    if args.uncertainty_method is None and args.calibration_method is None:
        uncertainty_names = []

    invalid_predictions = [
        ["Invalid SMILES"] * num_prediction_tasks for _ in range(len(full_data))
    ]
    invalid_uncertainties = [
        ["Invalid SMILES"] * num_uncertainty_tasks for _ in range(len(full_data))
    ]

    for datapoint in full_data:
        if args.drop_extra_columns:
            datapoint.row = OrderedDict(
                (column, smiles)
                for column, smiles in zip(args.smiles_columns, datapoint.smiles)
            )
        for name in output_task_names:
            datapoint.row[name] = "Invalid SMILES"
        for name in uncertainty_names:
            datapoint.row[name] = "Invalid SMILES"
        if args.individual_ensemble_predictions:
            for name in output_task_names:
                for model_index in range(len(args.checkpoint_paths)):
                    datapoint.row[f"{name}_model_{model_index}"] = "Invalid SMILES"

    if len(full_data) > 0:
        fieldnames = list(full_data[0].row.keys())
    else:
        if (
            not args.drop_extra_columns
            and args.test_path is not None
            and os.path.isfile(args.test_path)
        ):
            fieldnames = list(get_header(args.test_path))
        else:
            fieldnames = list(args.smiles_columns)
        for name in output_task_names + uncertainty_names:
            if name not in fieldnames:
                fieldnames.append(name)
        if args.individual_ensemble_predictions:
            fieldnames.extend(
                f"{name}_model_{model_index}"
                for name in output_task_names
                for model_index in range(len(args.checkpoint_paths))
            )

    print(f"Saving predictions to {args.preds_path}")
    makedirs(args.preds_path, isfile=True)
    with open(args.preds_path, "w", newline="") as predictions_file:
        writer = csv.DictWriter(predictions_file, fieldnames=fieldnames)
        writer.writeheader()
        for datapoint in full_data:
            writer.writerow(datapoint.row)

    if return_invalid_smiles:
        return invalid_predictions, invalid_uncertainties
    return [], []


def predict_and_save(
    args: PredictArgs,
    train_args: TrainArgs,
    test_data: MoleculeDataset,
    task_names: List[str],
    num_tasks: int,
    test_data_loader: MoleculeDataLoader,
    full_data: MoleculeDataset,
    full_to_valid_indices: dict,
    models: List[MoleculeModel],
    scalers: List[Union[StandardScaler, AtomBondScaler]],
    num_models: int,
    calibrator: UncertaintyCalibrator = None,
    return_invalid_smiles: bool = False,
    save_results: bool = True,
):
    """
    Function to predict with a model and save the predictions to file.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param train_args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    :param test_data: A :class:`~chemprop.data.MoleculeDataset` containing valid datapoints.
    :param task_names: A list of task names.
    :param num_tasks: Number of tasks.
    :param test_data_loader: A :class:`~chemprop.data.MoleculeDataLoader` to load the test data.
    :param full_data:  A :class:`~chemprop.data.MoleculeDataset` containing all (valid and invalid) datapoints.
    :param full_to_valid_indices: A dictionary dictionary mapping full to valid indices.
    :param models: A list or generator object of :class:`~chemprop.models.MoleculeModel` objects.
    :param scalers: A list or generator object of :class:`~chemprop.features.scaler.StandardScaler` objects.
    :param num_models: The number of models included in the models and scalers input.
    :param calibrator: A :class: `~chemprop.uncertainty.UncertaintyCalibrator` object, for use in calibrating uncertainty predictions.
    :param return_invalid_smiles: Whether to return predictions of "Invalid SMILES" for invalid SMILES, otherwise will skip them in returned predictions.
    :param save_results: Whether to save the predictions in a csv. Function returns the predictions regardless.
    :return: A list of lists of target predictions.
    """
    estimator = UncertaintyEstimator(
        test_data=test_data,
        test_data_loader=test_data_loader,
        uncertainty_method=args.uncertainty_method,
        models=models,
        scalers=scalers,
        num_models=num_models,
        dataset_type=args.dataset_type,
        loss_function=args.loss_function,
        uncertainty_dropout_p=args.uncertainty_dropout_p,
        conformal_alpha=args.conformal_alpha,
        dropout_sampling_size=args.dropout_sampling_size,
        individual_ensemble_predictions=args.individual_ensemble_predictions,
        spectra_phase_mask=getattr(train_args, "spectra_phase_mask", None),
    )

    preds, unc = estimator.calculate_uncertainty(
        calibrator=calibrator
    )  # preds and unc are lists of shape(data,tasks)

    _validate_prediction_values(
        preds,
        label='Prediction',
        expected_rows=len(test_data),
        allow_nan=args.dataset_type == 'spectra',
    )
    if args.uncertainty_method is not None or calibrator is not None:
        _validate_prediction_values(
            unc,
            label='Uncertainty',
            expected_rows=len(test_data),
        )

    if args.loss_function == "quantile_interval":
        task_names = task_names[:len(task_names) // 2]

    if calibrator is not None and args.is_atom_bond_targets and args.calibration_method == "isotonic":
        unc = reshape_values(unc, test_data, len(args.atom_targets), len(args.bond_targets))

    if args.individual_ensemble_predictions:
        individual_preds = (
            estimator.individual_predictions()
        )  # shape(data, tasks, ensemble) or (data, tasks, classes, ensemble)

    if args.evaluation_methods is not None:

        evaluation_data = get_data(
            path=args.test_path,
            smiles_columns=args.smiles_columns,
            target_columns=task_names,
            args=args,
            features_path=args.features_path,
            features_generator=args.features_generator,
            phase_features_path=args.phase_features_path,
            atom_descriptors_path=args.atom_descriptors_path,
            bond_descriptors_path=args.bond_descriptors_path,
            max_data_size=args.max_data_size,
            loss_function=args.loss_function,
        )

        evaluators = []
        for evaluation_method in args.evaluation_methods:
            evaluator = build_uncertainty_evaluator(
                evaluation_method=evaluation_method,
                calibration_method=args.calibration_method,
                uncertainty_method=args.uncertainty_method,
                dataset_type=args.dataset_type,
                loss_function=args.loss_function,
                calibrator=calibrator,
                is_atom_bond_targets=args.is_atom_bond_targets,
            )
            evaluators.append(evaluator)
    else:
        evaluators = None

    if evaluators is not None:
        evaluations = []
        print(f"Evaluating uncertainty for tasks {task_names}")
        for evaluator in evaluators:
            evaluation = evaluator.evaluate(
                targets=evaluation_data.targets(), preds=preds, uncertainties=unc, mask=evaluation_data.mask()
            )
            evaluations.append(evaluation)
            print(
                f"Using evaluation method {evaluator.evaluation_method}: {evaluation}"
            )
    else:
        evaluations = None

    if args.dataset_type == "multiclass":
        num_tasks = num_tasks * args.multiclass_num_classes

    if args.uncertainty_method == "spectra_roundrobin":
        num_unc_tasks = 1
    elif args.uncertainty_method == "dirichlet" and args.dataset_type == "multiclass":
        num_unc_tasks = num_tasks // args.multiclass_num_classes # dirichlet only returns an uncertainty for each task rather than each class
    elif args.calibration_method == "conformal" and args.dataset_type == "classification":
        num_unc_tasks = 2 * num_tasks
    else:
        num_unc_tasks = num_tasks

    # Save results
    if save_results:
        print(f"Saving predictions to {args.preds_path}")

        makedirs(args.preds_path, isfile=True)

        # Set multiclass column names, update num_tasks definitions
        if args.dataset_type == "multiclass":
            original_task_names = task_names
            task_names = [
                f"{name}_class_{i}"
                for name in task_names
                for i in range(args.multiclass_num_classes)
            ]

        # Copy predictions over to full_data
        for full_index, datapoint in enumerate(full_data):
            valid_index = full_to_valid_indices.get(full_index, None)
            if valid_index is not None:
                d_preds = preds[valid_index]
                d_unc = unc[valid_index]
                if args.individual_ensemble_predictions:
                    ind_preds = individual_preds[valid_index]
            else:
                d_preds = ["Invalid SMILES"] * num_tasks
                d_unc = ["Invalid SMILES"] * num_unc_tasks
                if args.individual_ensemble_predictions:
                    ind_preds = [["Invalid SMILES"] * len(args.checkpoint_paths)] * num_tasks
            # Reshape multiclass to merge task and class dimension, with updated num_tasks
            if args.dataset_type == "multiclass":
                d_preds = np.array(d_preds).reshape((num_tasks))
                d_unc = np.array(d_unc).reshape((num_unc_tasks))
                if args.individual_ensemble_predictions:
                    ind_preds = ind_preds.reshape(
                        (num_tasks, len(args.checkpoint_paths))
                    )

            # If extra columns have been dropped, add back in SMILES columns
            if args.drop_extra_columns:
                datapoint.row = OrderedDict()

                smiles_columns = args.smiles_columns

                for column, smiles in zip(smiles_columns, datapoint.smiles):
                    datapoint.row[column] = smiles

            # Add predictions columns
            if args.uncertainty_method == "spectra_roundrobin":
                unc_names = [estimator.label]
            elif args.uncertainty_method == "conformal_quantile_regression" and args.calibration_method is None:
                unc_names = [f"{name}_{args.conformal_alpha}_half_interval" for name in task_names]
            elif args.calibration_method == "conformal_regression" and calibrator is None:
                unc_names = []
            elif args.calibration_method == "conformal" and args.dataset_type == "classification":
                unc_names = [f"{name}_{estimator.label}_in_set" for name in task_names] + [
                    f"{name}_{estimator.label}_out_set" for name in task_names
                ]
            else:
                unc_names = [name + f"_{estimator.label}" for name in task_names]
            
            for pred_name, pred in zip(task_names, d_preds):
                datapoint.row[pred_name] = pred
            

            for unc_name, un in zip(unc_names, d_unc):
                if (
                    args.uncertainty_method is not None or args.calibration_method is not None
                ):
                    datapoint.row[unc_name] = un
            if args.individual_ensemble_predictions:
                for pred_name, model_preds in zip(task_names, ind_preds):
                    for idx, pred in enumerate(model_preds):
                        datapoint.row[pred_name + f"_model_{idx}"] = pred

        # Save
        with open(args.preds_path, 'w', newline="") as f:
            writer = csv.DictWriter(f, fieldnames=full_data[0].row.keys())
            writer.writeheader()

            for datapoint in full_data:
                writer.writerow(datapoint.row)

        if evaluations is not None and args.evaluation_scores_path is not None:
            print(f"Saving uncertainty evaluations to {args.evaluation_scores_path}")
            if args.dataset_type == "multiclass":
                task_names = original_task_names
            with open(args.evaluation_scores_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["evaluation_method"] + task_names)
                for i, evaluation_method in enumerate(args.evaluation_methods):
                    writer.writerow([evaluation_method] + evaluations[i])

    if return_invalid_smiles:
        full_preds = []
        full_unc = []
        for full_index in range(len(full_data)):
            valid_index = full_to_valid_indices.get(full_index, None)
            if valid_index is not None:
                pred = preds[valid_index]
                un = unc[valid_index]
            else:
                pred = ["Invalid SMILES"] * num_tasks
                un = ["Invalid SMILES"] * num_unc_tasks
            full_preds.append(pred)
            full_unc.append(un)
        return full_preds, full_unc
    else:
        return preds, unc
    

def predict_lgbm(
    args: PredictArgs,
    model: LightGBMModelBundle,
    scaler,
    test_data: MoleculeDataset,
    encoded_features: np.ndarray = None,
) -> np.ndarray:
    """Predicts with a versioned bundle and its restored frozen encoder."""
    # Keep LightGBM optional for standard FFN prediction imports.
    from chemprop.train.run_training_lgbm import (
        encode_lgbm_features,
        predict_task_boosters,
    )

    if not isinstance(model, LightGBMModelBundle):
        raise TypeError(
            "LightGBM prediction requires a versioned Chemprop LightGBM bundle."
        )
    _require_safe_lgbm_representation(model)

    if encoded_features is None:
        _apply_lgbm_feature_scalers(test_data, model.scalers)
        encoded_features = encode_lgbm_features(
            encoder=model.encoder,
            data=test_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    predictions = predict_task_boosters(model.task_boosters, encoded_features)
    data_scaler = model.scalers[0]
    if data_scaler is not None:
        predictions = data_scaler.inverse_transform(predictions)
    return np.asarray(predictions, dtype=float)


def _require_safe_lgbm_representation(model: LightGBMModelBundle) -> None:
    """Rejects bundles fitted to an untrained random neural representation."""
    if not getattr(model.train_args, "features_only", False):
        raise ValueError(
            f'LightGBM bundle "{model.checkpoint_path}" was trained without '
            "--features_only and therefore uses an untrained random MPN "
            "representation. Prediction is disabled because its model quality "
            "is unreliable. Retrain with this Chemprop release using, for "
            "example, --features_generator morgan --features_only."
        )


def _apply_lgbm_feature_scalers(test_data: MoleculeDataset, scalers) -> None:
    """Resets raw inputs and applies the exact feature scalers from training."""
    test_data.reset_features_and_targets()
    _, features_scaler, atom_descriptor_scaler, bond_descriptor_scaler, _ = scalers
    if features_scaler is not None:
        test_data.normalize_features(features_scaler)
    if atom_descriptor_scaler is not None:
        test_data.normalize_features(
            atom_descriptor_scaler, scale_atom_descriptors=True
        )
    if bond_descriptor_scaler is not None:
        test_data.normalize_features(
            bond_descriptor_scaler, scale_bond_descriptors=True
        )


def _lgbm_input_scaler_key(scalers) -> tuple:
    """Returns a stable key for the scalers that affect encoder inputs."""
    key = []
    for scaler in scalers[1:4]:
        if scaler is None:
            key.append(None)
        else:
            means = np.asarray(scaler.means)
            stds = np.asarray(scaler.stds)
            key.append(
                (means.dtype.str, means.shape, means.tobytes(), stds.dtype.str, stds.shape, stds.tobytes())
            )
    return tuple(key)

    
def predict_and_save_lgbm(
    args: PredictArgs,
    train_args: TrainArgs,
    test_data: MoleculeDataset,
    task_names: List[str],
    num_tasks: int,
    test_data_loader: MoleculeDataLoader,
    full_data: MoleculeDataset,
    full_to_valid_indices: dict,
    models: List[LightGBMModelBundle],
    scalers: List[Union[StandardScaler, AtomBondScaler]],
    num_models: int,
    calibrator: UncertaintyCalibrator = None,
    return_invalid_smiles: bool = False,
    save_results: bool = True,
):
    """Ensembles LightGBM bundles and optionally writes predictions."""
    # Keep LightGBM optional for standard FFN prediction imports.
    from chemprop.train.run_training_lgbm import encode_lgbm_features

    models = list(models)
    scalers = list(scalers)
    if not models:
        raise ValueError("At least one LightGBM bundle is required for prediction.")
    if len(models) != len(scalers):
        raise ValueError("LightGBM model and scaler counts do not match.")
    if num_models != len(models):
        raise ValueError(
            f"num_models={num_models} does not match the {len(models)} "
            "supplied LightGBM bundles."
        )

    encoded_feature_cache = {}
    test_preds = []
    for model, scaler in zip(models, scalers):
        cache_key = (id(model.encoder), _lgbm_input_scaler_key(model.scalers))
        if cache_key not in encoded_feature_cache:
            _apply_lgbm_feature_scalers(test_data, model.scalers)
            encoded_feature_cache[cache_key] = encode_lgbm_features(
                encoder=model.encoder,
                data=test_data,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        test_preds.append(
            predict_lgbm(
                args,
                model,
                scaler,
                test_data,
                encoded_feature_cache[cache_key],
            )
        )
    individual_preds = np.stack(test_preds, axis=2)
    preds = np.mean(individual_preds, axis=2)
    if preds.shape != (len(test_data), num_tasks):
        raise ValueError(
            f'LightGBM prediction shape {preds.shape} does not match expected '
            f'shape {(len(test_data), num_tasks)}.'
        )
    if not np.all(np.isfinite(preds)):
        raise ValueError('LightGBM prediction output contains non-finite values.')
    
    # Save results
    if save_results:
        print(f"Saving predictions to {args.preds_path}")

        makedirs(args.preds_path, isfile=True)

        # Copy predictions over to full_data
        for full_index, datapoint in enumerate(full_data):
            valid_index = full_to_valid_indices.get(full_index, None)
            if valid_index is not None:
                d_preds = preds[valid_index]
                if args.individual_ensemble_predictions:
                    ind_preds = individual_preds[valid_index]
            else:
                d_preds = ["Invalid SMILES"] * num_tasks
                if args.individual_ensemble_predictions:
                    ind_preds = [["Invalid SMILES"] * len(models) for _ in range(num_tasks)]

            # If extra columns have been dropped, add back in SMILES columns
            if args.drop_extra_columns:
                datapoint.row = OrderedDict()

                smiles_columns = args.smiles_columns

                for column, smiles in zip(smiles_columns, datapoint.smiles):
                    datapoint.row[column] = smiles

            # Add predictions columns
            for pred_name, pred in zip(
                task_names,  d_preds
            ):
                datapoint.row[pred_name] = pred
                
            if args.individual_ensemble_predictions:
                for pred_name, model_preds in zip(task_names, ind_preds):
                    for idx, pred in enumerate(model_preds):
                        datapoint.row[pred_name + f"_model_{idx}"] = pred

        fieldnames = list(args.smiles_columns) + list(task_names)
        if args.individual_ensemble_predictions:
            fieldnames.extend(
                f"{task_name}_model_{model_index}"
                for task_name in task_names
                for model_index in range(len(models))
            )
        if len(full_data) > 0:
            fieldnames = list(full_data[0].row.keys())

        # Save, including a header-only file for a truly empty input dataset.
        with open(args.preds_path, 'w', newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for datapoint in full_data:
                writer.writerow(datapoint.row)

    if return_invalid_smiles:
        full_preds = []
        for full_index in range(len(full_data)):
            valid_index = full_to_valid_indices.get(full_index, None)
            if valid_index is not None:
                pred = preds[valid_index].tolist()
            else:
                pred = ["Invalid SMILES"] * num_tasks
            full_preds.append(pred)
        return full_preds
    else:
        return preds.tolist()


@timeit()
def make_predictions(
    args: PredictArgs,
    smiles: List[List[str]] = None,
    model_objects: Tuple[
        PredictArgs,
        TrainArgs,
        List[MoleculeModel],
        List[Union[StandardScaler, AtomBondScaler]],
        int,
        List[str],
    ] = None,
    calibrator: UncertaintyCalibrator = None,
    return_invalid_smiles: bool = True,
    return_index_dict: bool = False,
    return_uncertainty: bool = False,
) -> List[List[Optional[float]]]:
    """
    Loads data and a trained model and uses the model to make predictions on the data.

    If SMILES are provided, then makes predictions on smiles.
    Otherwise makes predictions on :code:`args.test_data`.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                loading data and a model and making predictions.
    :param smiles: List of list of SMILES to make predictions on.
    :param model_objects: Tuple of output of load_model function which can be called separately outside this function. Preloaded model objects should have
                used the non-generator option for load_model if the objects are to be used multiple times or are intended to be used for calibration as well.
    :param calibrator: A :class: `~chemprop.uncertainty.UncertaintyCalibrator` object, for use in calibrating uncertainty predictions.
                Can be preloaded and provided as a function input or constructed within the function from arguments. The models and scalers used
                to initiate the calibrator must be lists instead of generators if the same calibrator is to be used multiple times or
                if the same models and scalers objects are also part of the provided model_objects input.
    :param return_invalid_smiles: Whether to return predictions of "Invalid SMILES" for invalid SMILES, otherwise will skip them in returned predictions.
    :param return_index_dict: Whether to return the prediction results as a dictionary keyed from the initial data indexes.
    :param return_uncertainty: Whether to return uncertainty predictions alongside the model value predictions.
    :return: A list of lists of target predictions. If returning uncertainty, a tuple containing first prediction values then uncertainty estimates.
    """
    if model_objects:
        (args, train_args, models, scalers, num_tasks, task_names) = model_objects
    else:
        (args, train_args, models, scalers, num_tasks, task_names) = load_model(
            args, generator=True
        )

    num_models = len(args.checkpoint_paths)

    set_features(args, train_args)

    # Note: to get the invalid SMILES for your data, use the get_invalid_smiles_from_file or get_invalid_smiles_from_list functions from data/utils.py
    full_data, test_data, test_data_loader, full_to_valid_indices = load_data(
        args, smiles, train_args=train_args
    )

    if args.uncertainty_method is not None and args.calibration_method in [
        "conformal_regression",
        "conformal_quantile_regression",
    ]:
        raise ValueError("Conformal regression is not compatible with an uncertainty method")

    if args.uncertainty_method is None and (
        args.calibration_method is not None or args.evaluation_methods is not None
    ):
        if args.dataset_type in ["classification", "multiclass"]:
            args.uncertainty_method = "classification"
        elif args.calibration_method == "conformal_regression":
            if args.loss_function == "quantile_interval":
                raise ValueError(
                    "For a model trained on the `quantile_interval` loss function, the calibration method should be assigned as `conformal_quantile_regression` instead of `conformal_regression`."
                    )
            args.uncertainty_method = "conformal_regression"
        elif args.calibration_method == "conformal_quantile_regression":
            if args.loss_function != "quantile_interval":
                raise ValueError(
                    "The calibration method `conformal_quantile_regression` only supports regression models trained on the `quantile_interval` loss function."
                    )
            args.uncertainty_method = "conformal_quantile_regression"
        else:
            raise ValueError(
                "Cannot calibrate or evaluate uncertainty without selection of an uncertainty method."
            )

    if args.calibration_method is None and args.loss_function == "quantile_interval":
        args.uncertainty_method = "conformal_quantile_regression"

    if calibrator is None and args.calibration_path is not None:

        calibration_data = get_data(
            path=args.calibration_path,
            smiles_columns=args.smiles_columns,
            target_columns=task_names,
            args=args,
            features_path=args.calibration_features_path,
            features_generator=args.features_generator,
            phase_features_path=args.calibration_phase_features_path,
            atom_descriptors_path=args.calibration_atom_descriptors_path,
            bond_descriptors_path=args.calibration_bond_descriptors_path,
            max_data_size=args.max_data_size,
            loss_function=args.loss_function,
        )

        validate_prediction_feature_schema(
            args=args,
            train_args=train_args,
            full_data=calibration_data,
            valid_data=calibration_data,
            input_label="Calibration",
        )

        if len(calibration_data) == 0:
            raise ValueError(
                'Calibration data must contain at least one valid molecule.'
            )
        calibration_mask = calibration_data.mask()
        if len(calibration_mask) != len(task_names):
            raise ValueError(
                'Calibration target shape does not match the checkpoint task count.'
            )
        missing_tasks = [
            task_names[index] if index < len(task_names) else str(index)
            for index, task_mask in enumerate(calibration_mask)
            if not any(task_mask)
        ]
        if missing_tasks:
            raise ValueError(
                'Calibration data must contain at least one observed target for '
                f'every task; missing: {", ".join(missing_tasks)}.'
            )

        calibration_data_loader = MoleculeDataLoader(
            dataset=calibration_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        if isinstance(models, List) and isinstance(scalers, List):
            calibration_models = models
            calibration_scalers = scalers
        else:
            calibration_model_objects = load_model(args, generator=True)
            calibration_models = calibration_model_objects[2]
            calibration_scalers = calibration_model_objects[3]

        calibrator = build_uncertainty_calibrator(
            calibration_method=args.calibration_method,
            uncertainty_method=args.uncertainty_method,
            interval_percentile=args.calibration_interval_percentile,
            regression_calibrator_metric=args.regression_calibrator_metric,
            calibration_data=calibration_data,
            calibration_data_loader=calibration_data_loader,
            models=calibration_models,
            scalers=calibration_scalers,
            num_models=num_models,
            dataset_type=args.dataset_type,
            loss_function=args.loss_function,
            uncertainty_dropout_p=args.uncertainty_dropout_p,
            conformal_alpha=args.conformal_alpha,
            dropout_sampling_size=args.dropout_sampling_size,
            spectra_phase_mask=getattr(train_args, "spectra_phase_mask", None),
        )

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        preds, unc = _save_no_valid_ffn_predictions(
            args=args,
            full_data=full_data,
            task_names=task_names,
            calibrator=calibrator,
            return_invalid_smiles=return_invalid_smiles,
        )
    else:
        preds, unc = predict_and_save(
            args=args,
            train_args=train_args,
            test_data=test_data,
            task_names=task_names,
            num_tasks=num_tasks,
            test_data_loader=test_data_loader,
            full_data=full_data,
            full_to_valid_indices=full_to_valid_indices,
            models=models,
            scalers=scalers,
            num_models=num_models,
            calibrator=calibrator,
            return_invalid_smiles=return_invalid_smiles,
        )

    if return_index_dict:
        preds_dict = {}
        unc_dict = {}
        for i in range(len(full_data)):
            if return_invalid_smiles:
                preds_dict[i] = preds[i]
                unc_dict[i] = unc[i]
            else:
                valid_index = full_to_valid_indices.get(i, None)
                if valid_index is not None:
                    preds_dict[i] = preds[valid_index]
                    unc_dict[i] = unc[valid_index]
        if return_uncertainty:
            return preds_dict, unc_dict
        else:
            return preds_dict
    else:
        if return_uncertainty:
            return preds, unc
        else:
            return preds
        
        
@timeit()
def make_predictions_lgbm(
    args: PredictArgs,
    smiles: List[List[str]] = None,
    model_objects: Tuple[
        PredictArgs,
        TrainArgs,
        List[LightGBMModelBundle],
        List[Union[StandardScaler, AtomBondScaler]],
        int,
        List[str],
    ] = None,
    calibrator: UncertaintyCalibrator = None,
    return_invalid_smiles: bool = True,
    return_index_dict: bool = False,
    return_uncertainty: bool = False,
) -> List[List[Optional[float]]]:
    """Makes predictions with one or more versioned LightGBM bundles."""
    if return_uncertainty:
        raise ValueError("LightGBM prediction does not provide uncertainty estimates.")
    if calibrator is not None or getattr(args, "calibration_path", None) is not None:
        raise ValueError("LightGBM prediction does not support uncertainty calibration.")
    if getattr(args, "uncertainty_method", None) is not None:
        raise ValueError("LightGBM prediction does not support uncertainty methods.")

    if model_objects:
        (
            args,
            train_args,
            models,
            scalers,
            num_tasks,
            task_names,
        ) = model_objects
    else:
        (
            args,
            train_args,
            models,
            scalers,
            num_tasks,
            task_names,
        ) = load_model_lgbm(args, generator=True)

    models = list(models)
    scalers = list(scalers)
    num_models = len(models)

    set_features(args, train_args)

    # Note: to get the invalid SMILES for your data, use the get_invalid_smiles_from_file or get_invalid_smiles_from_list functions from data/utils.py
    full_data, test_data, test_data_loader, full_to_valid_indices = load_data(
        args, smiles, train_args=train_args
    )

    # This also handles all-invalid and truly empty inputs, including CSV output.
    preds = predict_and_save_lgbm(
        args=args,
        train_args=train_args,
        test_data=test_data,
        task_names=task_names,
        num_tasks=num_tasks,
        test_data_loader=test_data_loader,
        full_data=full_data,
        full_to_valid_indices=full_to_valid_indices,
        models=models,
        scalers=scalers,
        num_models=num_models,
        calibrator=calibrator,
        return_invalid_smiles=return_invalid_smiles,
    )

    if return_index_dict:
        if return_invalid_smiles:
            return {index: prediction for index, prediction in enumerate(preds)}
        return {
            full_index: preds[valid_index]
            for full_index, valid_index in full_to_valid_indices.items()
        }
    return preds


def chemprop_predict() -> None:
    """Parses Chemprop predicting arguments and runs prediction using a trained Chemprop model.

    This is the entry point for the command line command :code:`chemprop_predict`.
    """
    args = PredictArgs().parse_args()
    
    if args.model_type == 'FFN':
        make_predictions(args=args)
    elif args.model_type == 'lgbm':
        make_predictions_lgbm(args=args)
