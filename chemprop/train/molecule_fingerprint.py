import csv
from typing import List, Optional, Union

import torch
import numpy as np
from tqdm import tqdm

from chemprop.args import FingerprintArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, is_valid_datapoint, load_selected_feature_columns, MoleculeDataLoader, MoleculeDataset
from chemprop.utils import load_args, load_checkpoint, makedirs, timeit, load_scalers, update_prediction_args
from chemprop.features import get_features_generators_metadata, set_reaction, set_explicit_h, set_adding_hs, set_keeping_atom_map, reset_featurization_parameters, set_extra_atom_fdim, set_extra_bond_fdim
from chemprop.models import MoleculeModel
from chemprop.train.make_predictions import (
    _FFN_ENSEMBLE_COMPATIBILITY_FIELDS,
    _feature_source_metadata_matches,
    _validate_ensemble_train_args,
)


def restore_checkpoint_featurization(train_args: TrainArgs) -> None:
    """Restores every process-global graph setting recorded by a checkpoint."""
    reset_featurization_parameters()
    set_explicit_h(train_args.explicit_h)
    set_adding_hs(getattr(train_args, 'adding_h', False))
    set_keeping_atom_map(getattr(train_args, 'keeping_atom_map', False))
    if train_args.reaction:
        set_reaction(True, train_args.reaction_mode)
    elif train_args.reaction_solvent:
        set_reaction(True, train_args.reaction_mode)
    if train_args.atom_descriptors == 'feature':
        set_extra_atom_fdim(train_args.atom_features_size)
    if train_args.bond_descriptors == 'feature':
        set_extra_bond_fdim(train_args.bond_features_size)


def validate_checkpoint_feature_schema(
    args: Union[FingerprintArgs, object],
    train_args: TrainArgs,
    data: MoleculeDataset,
    valid_data: MoleculeDataset = None,
) -> None:
    """Validates generator identity, realized width, and feature-source layout."""
    expected_generators = list(train_args.features_generator or [])
    actual_generators = list(args.features_generator or [])
    if actual_generators != expected_generators:
        raise ValueError(
            'Feature generators do not match the checkpoint: '
            f'expected {expected_generators}, received {actual_generators}.'
        )

    # Invalid molecules may carry a zero-width placeholder even when the
    # generator schema has a known width, so only validate a realized valid row.
    width_data = data if valid_data is None else valid_data
    actual_features_size = width_data.features_size() if len(width_data) > 0 else None
    expected_features_size = getattr(train_args, 'features_size', None)
    if (
        actual_features_size is not None
        and expected_features_size is not None
        and actual_features_size != expected_features_size
    ):
        raise ValueError(
            'Feature width does not match the checkpoint: '
            f'generated {actual_features_size}, expected {expected_features_size}.'
        )

    expected_generator_metadata = getattr(
        train_args, 'features_generator_metadata', None
    )
    if expected_generator_metadata is not None:
        selected_feature_columns = (
            load_selected_feature_columns(args.selected_features_path)
            if args.selected_features_path is not None
            else {}
        )
        actual_generator_metadata = get_features_generators_metadata(
            actual_generators,
            selected_feature_columns=selected_feature_columns,
            total_dimension=(
                actual_features_size
                if actual_features_size is not None
                else expected_generator_metadata.get('total_dimension')
            ),
        )
        if actual_generator_metadata != expected_generator_metadata:
            raise ValueError(
                'Feature generator schema does not match the checkpoint. Use the '
                'same ordered generators, selected columns, and dependency versions.'
            )

    expected_source_metadata = getattr(train_args, 'features_source_metadata', None)
    actual_source_metadata = getattr(data, '_features_source_metadata', None)
    if (
        expected_source_metadata is not None
        and not _feature_source_metadata_matches(
            expected_source_metadata, actual_source_metadata
        )
    ):
        raise ValueError(
            'Feature sources do not match the checkpoint. Use the same ordered '
            'external, phase, and generated feature layout.'
        )


def validate_checkpoint_ensemble(train_args_list: List[TrainArgs]) -> None:
    """Rejects FFN checkpoints with incompatible architectures or feature schemas."""
    _validate_ensemble_train_args(
        train_args_list,
        _FFN_ENSEMBLE_COMPATIBILITY_FIELDS,
        'FFN',
    )

@timeit()
def molecule_fingerprint(args: FingerprintArgs,
                         smiles: List[List[str]] = None,
                         return_invalid_smiles: bool = True) -> List[List[Optional[float]]]:
    """
    Loads data and a trained model and uses the model to encode fingerprint vectors for the data.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: List of list of SMILES to make predictions on.
    :param return_invalid_smiles: Whether to return predictions of "Invalid SMILES" for invalid SMILES, otherwise will skip them in returned predictions.
    :return: A list of fingerprint vectors (list of floats)
    """

    print('Loading training args')
    checkpoint_train_args = [load_args(path) for path in args.checkpoint_paths]
    validate_checkpoint_ensemble(checkpoint_train_args)
    train_args = checkpoint_train_args[0]
    if getattr(train_args, "is_atom_bond_targets", False):
        raise ValueError(
            'Latent fingerprint export is not supported for atom/bond target '
            'models because their MPN output is not a molecule-level tensor.'
        )

    # Update args with training arguments
    # MPN fingerprints are truncated before being returned, but the current v1
    # encoder still consumes and concatenates input features internally. Their
    # exact schema is therefore required for both MPN and last_FFN outputs.
    update_prediction_args(
        predict_args=args,
        train_args=train_args,
        validate_feature_sources=True,
    )
    args: Union[FingerprintArgs, TrainArgs]

    restore_checkpoint_featurization(train_args)

    print('Loading data')
    if smiles is not None:
        full_data = get_data_from_smiles(
            smiles=smiles,
            skip_invalid_smiles=False,
            features_generator=args.features_generator,
            selected_features_path=args.selected_features_path,
        )
    else:
        full_data = get_data(path=args.test_path, smiles_columns=args.smiles_columns, target_columns=[], ignore_columns=[], skip_invalid_smiles=False,
                             args=args, use_args_data_weights=False, store_row=True)

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if is_valid_datapoint(full_data[full_index]):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])
    validate_checkpoint_feature_schema(args, train_args, full_data, test_data)

    print(f'Test size = {len(test_data):,}')

    # Create data loader
    test_data_loader = (
        MoleculeDataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if len(test_data) > 0
        else None
    )

    # Set fingerprint size
    if args.fingerprint_type == 'MPN':
        if args.atom_descriptors == "descriptor": # special case when we have 'descriptor' extra dimensions need to be added
            total_fp_size = (
                args.hidden_size + train_args.atom_descriptors_size
            ) * args.number_of_molecules
        else:
            if args.reaction_solvent:
                total_fp_size = args.hidden_size + args.hidden_size_solvent
            else:
                total_fp_size = args.hidden_size * args.number_of_molecules
        if args.features_only:
            raise ValueError('With features_only models, there is no latent MPN representation. Use last_FFN fingerprint type instead.')
    elif args.fingerprint_type == 'last_FFN':
        if args.ffn_num_layers != 1:
            total_fp_size = args.ffn_hidden_size
        else:
            raise ValueError('With a ffn_num_layers of 1, there is no latent FFN representation. Use MPN fingerprint type instead.')
    else:
        raise ValueError(f'Fingerprint type {args.fingerprint_type} not supported')
    all_fingerprints = np.zeros((len(test_data), total_fp_size, len(args.checkpoint_paths)))

    # Load model
    print(f'Encoding smiles into a fingerprint vector from {len(args.checkpoint_paths)} models.')

    for index, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths))):
        if len(test_data) == 0:
            break
        model = load_checkpoint(checkpoint_path, device=args.device)
        scaler, features_scaler, atom_descriptor_scaler, bond_descriptor_scaler, atom_bond_scaler = load_scalers(args.checkpoint_paths[index])

        # Normalize features
        if args.features_scaling or train_args.atom_descriptor_scaling or train_args.bond_descriptor_scaling:
            test_data.reset_features_and_targets()
            if args.features_scaling:
                test_data.normalize_features(features_scaler)
            if train_args.atom_descriptor_scaling and args.atom_descriptors is not None:
                test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
            if train_args.bond_descriptor_scaling and args.bond_descriptors is not None:
                test_data.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)

        # Make fingerprints
        model_fp = model_fingerprint(
            model=model,
            data_loader=test_data_loader,
            fingerprint_type=args.fingerprint_type
        )
        if args.fingerprint_type == 'MPN' and (
            args.features_path is not None
            or getattr(args, 'phase_features_path', None) is not None
            or args.features_generator
        ):
            # v1's MPN fingerprint path concatenates all input features. Keep
            # only the graph representation requested by fingerprint_type=MPN.
            model_fp = np.asarray(model_fp)[:, :total_fp_size]
        model_fp = np.asarray(model_fp, dtype=float)
        expected_shape = (len(test_data), total_fp_size)
        if model_fp.shape != expected_shape:
            raise ValueError(
                f'Checkpoint {checkpoint_path!r} returned fingerprint shape '
                f'{model_fp.shape}; expected {expected_shape}.'
            )
        if not np.all(np.isfinite(model_fp)):
            raise ValueError(
                f'Checkpoint {checkpoint_path!r} produced a non-finite '
                'fingerprint for a valid molecule.'
            )
        all_fingerprints[:, :, index] = model_fp

    # Save predictions
    print(f'Saving predictions to {args.preds_path}')
    makedirs(args.preds_path, isfile=True)

    # Set column names
    fingerprint_columns = []
    if args.fingerprint_type == 'MPN':
        component_sizes = (
            [args.hidden_size, args.hidden_size_solvent]
            if args.reaction_solvent
            else [total_fp_size // args.number_of_molecules] * args.number_of_molecules
        )
        if sum(component_sizes) != total_fp_size:
            raise ValueError(
                'MPN fingerprint component widths do not sum to the model output width.'
            )
        for k, component_size in enumerate(component_sizes):
            for j in range(component_size):
                if len(args.checkpoint_paths) == 1:
                    fingerprint_columns.append(f'fp_{j}_mol_{k}')
                else:
                    for i in range(len(args.checkpoint_paths)):
                        fingerprint_columns.append(f'fp_{j}_mol_{k}_model_{i}')

    else: # args == 'last_FNN'
        if len(args.checkpoint_paths) == 1:
            for j in range(total_fp_size):
                fingerprint_columns.append(f'fp_{j}')
        else:
            for j in range(total_fp_size):
                for i in range(len(args.checkpoint_paths)):
                    fingerprint_columns.append(f'fp_{j}_model_{i}')

    expected_column_count = total_fp_size * len(args.checkpoint_paths)
    if len(fingerprint_columns) != expected_column_count:
        raise ValueError(
            f'Generated {len(fingerprint_columns)} fingerprint columns for '
            f'{expected_column_count} values.'
        )

    # Copy predictions over to full_data
    for full_index, datapoint in enumerate(full_data):
        valid_index = full_to_valid_indices.get(full_index, None)
        preds = all_fingerprints[valid_index].reshape((len(args.checkpoint_paths) * total_fp_size)) if valid_index is not None else ['Invalid SMILES'] * len(args.checkpoint_paths) * total_fp_size

        for i in range(len(fingerprint_columns)):
            datapoint.row[fingerprint_columns[i]] = preds[i]

    # Write predictions
    with open(args.preds_path, 'w', newline="") as f:
        writer = csv.DictWriter(f, fieldnames=args.smiles_columns+fingerprint_columns,extrasaction='ignore')
        writer.writeheader()
        for datapoint in full_data:
            writer.writerow(datapoint.row)

    if return_invalid_smiles:
        full_fingerprints = np.zeros((len(full_data), total_fp_size, len(args.checkpoint_paths)), dtype='object')
        for full_index in range(len(full_data)):
            valid_index = full_to_valid_indices.get(full_index, None)
            preds = all_fingerprints[valid_index] if valid_index is not None else np.full((total_fp_size, len(args.checkpoint_paths)), 'Invalid SMILES')
            full_fingerprints[full_index] = preds
        return full_fingerprints
    else:
        return all_fingerprints

def model_fingerprint(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            fingerprint_type: str = 'MPN',
            disable_progress_bar: bool = False) -> List[List[float]]:
    """
    Encodes the provided molecules into the latent fingerprint vectors, according to the provided model.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :return: A list of fingerprint vector lists.
    """
    model.eval()

    fingerprints = []

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_descriptors_batch, bond_features_batch = \
            batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_descriptors(), batch.bond_features()

        # Make predictions
        with torch.no_grad():
            batch_fp = model.fingerprint(mol_batch, features_batch, atom_descriptors_batch,
                                         atom_features_batch, bond_descriptors_batch,
                                         bond_features_batch, fingerprint_type)

        # Collect vectors
        batch_fp = batch_fp.data.cpu().tolist()

        fingerprints.extend(batch_fp)

    return fingerprints

def chemprop_fingerprint() -> None:
    """
    Parses Chemprop predicting arguments and returns the latent representation vectors for
    provided molecules, according to a previously trained model.
    """
    molecule_fingerprint(args=FingerprintArgs().parse_args())
