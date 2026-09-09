from argparse import Namespace
import csv
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
import logging
from math import ceil
import os
import pickle
import re
import tempfile
from time import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import collections

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import PredictArgs, TrainArgs, FingerprintArgs
from chemprop.data import StandardScaler, AtomBondScaler, MoleculeDataset, preprocess_smiles_columns, get_task_names
from chemprop.features import (
    reset_featurization_parameters,
    set_adding_hs,
    set_explicit_h,
    set_extra_atom_fdim,
    set_extra_bond_fdim,
    set_keeping_atom_map,
    set_reaction,
)
from chemprop.models import MoleculeModel, MoleculeModelEncoder
from chemprop.nn_utils import NoamLR
from chemprop.models.ffn import MultiReadout

from packaging.version import parse as parse_version


LIGHTGBM_BUNDLE_FORMAT = "chemprop-lightgbm-bundle"
LIGHTGBM_BUNDLE_VERSION = 1


class LightGBMCheckpointError(ValueError):
    """Raised when a LightGBM checkpoint cannot be loaded safely or compatibly."""


@dataclass
class LightGBMModelBundle:
    """A loaded LightGBM model and the exact MPN/scalers used to create its inputs."""

    encoder: MoleculeModelEncoder
    task_boosters: List[Any]
    train_args: TrainArgs
    scalers: Tuple[
        Optional[StandardScaler],
        Optional[StandardScaler],
        Optional[StandardScaler],
        Optional[StandardScaler],
        Optional[AtomBondScaler],
    ]
    task_names: List[str]
    dataset_type: str
    model_index: int
    seed: int
    checkpoint_path: str


def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. :code:`isfile == True`),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)


def save_checkpoint(
    path: str,
    model: MoleculeModel,
    scaler: StandardScaler = None,
    features_scaler: StandardScaler = None,
    atom_descriptor_scaler: StandardScaler = None,
    bond_descriptor_scaler: StandardScaler = None,
    atom_bond_scaler: AtomBondScaler = None,
    args: TrainArgs = None,
) -> None:
    """
    Saves a model checkpoint.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the data.
    :param features_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the features.
    :param atom_descriptor_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the atom descriptors.
    :param bond_descriptor_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the bond descriptors.
    :param atom_bond_scaler: A :class:`~chemprop.data.scaler.AtomBondScaler` fitted on the atomic/bond targets.
    :param args: The :class:`~chemprop.args.TrainArgs` object containing the arguments the model was trained with.
    :param path: Path where checkpoint will be saved.
    """
    # Convert args to namespace for backwards compatibility
    if args is not None:
        args = Namespace(**args.as_dict())

    data_scaler = {"means": scaler.means, "stds": scaler.stds} if scaler is not None else None
    if atom_bond_scaler is not None:
        atom_bond_scaler = {"means": atom_bond_scaler.means, "stds": atom_bond_scaler.stds}
    if features_scaler is not None:
        features_scaler = {"means": features_scaler.means, "stds": features_scaler.stds}
    if atom_descriptor_scaler is not None:
        atom_descriptor_scaler = {
            "means": atom_descriptor_scaler.means,
            "stds": atom_descriptor_scaler.stds,
        }
    if bond_descriptor_scaler is not None:
        bond_descriptor_scaler = {"means": bond_descriptor_scaler.means, "stds": bond_descriptor_scaler.stds}

    state = {
        "args": args,
        "state_dict": model.state_dict(),
        "data_scaler": data_scaler,
        "features_scaler": features_scaler,
        "atom_descriptor_scaler": atom_descriptor_scaler,
        "bond_descriptor_scaler": bond_descriptor_scaler,
        "atom_bond_scaler": atom_bond_scaler,
    }
    absolute_path = os.path.abspath(path)
    makedirs(absolute_path, isfile=True)
    checkpoint_dir = os.path.dirname(absolute_path) or "."
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=checkpoint_dir, prefix=".checkpoint-", suffix=".tmp"
    )
    os.close(file_descriptor)
    try:
        torch.save(state, temporary_path)
        with open(temporary_path, "rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, absolute_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    

def _standard_scaler_state(scaler: StandardScaler) -> Dict[str, np.ndarray]:
    """Converts a scaler into plain checkpoint data."""
    if scaler is None:
        return None

    return {
        "means": np.asarray(scaler.means),
        "stds": np.asarray(scaler.stds),
    }


def _is_lightgbm_booster(model: Any) -> bool:
    """Checks the concrete LightGBM type without importing the optional package."""
    model_type = type(model)
    return model_type.__name__ == "Booster" and model_type.__module__.startswith(
        "lightgbm."
    )


def _validate_lgbm_training_args(
    args: TrainArgs,
    checkpoint_label: str,
    error_type: type = ValueError,
) -> int:
    """Validates arguments that define a reproducible LightGBM input schema.

    LightGBM prediction bypasses the FFN and, in the supported configuration,
    passes molecule-level features through the MPN unchanged.  Consequently,
    ``features_size`` is both the encoder output width and every Booster's
    input width.  Validate that contract before restoring any model objects.
    """

    def fail(message: str) -> None:
        raise error_type(f'{checkpoint_label} {message}')

    if not isinstance(args, TrainArgs):
        fail("contains invalid training arguments.")
    if getattr(args, "model_type", None) != "lgbm":
        fail('was not saved with model_type="lgbm".')
    dataset_type = getattr(args, "dataset_type", None)
    if not isinstance(dataset_type, str) or dataset_type not in {
        "classification", "regression"
    }:
        fail("has an unsupported dataset type.")
    expected_loss = (
        "binary_cross_entropy"
        if args.dataset_type == "classification"
        else "mse"
    )
    if getattr(args, "loss_function", None) != expected_loss:
        fail(
            f'has loss_function={getattr(args, "loss_function", None)!r}; '
            f'{args.dataset_type} LightGBM bundles require {expected_loss!r}.'
        )
    boolean_fields = (
        "features_only",
        "is_atom_bond_targets",
        "reaction",
        "reaction_solvent",
        "no_features_scaling",
    )
    if any(
        not isinstance(getattr(args, field, None), bool)
        for field in boolean_fields
    ):
        fail("contains a non-boolean representation setting.")
    if args.features_only is not True:
        fail("does not use the required deterministic features-only representation.")
    if all(
        getattr(args, source, None) is None
        for source in ("features_generator", "features_path", "phase_features_path")
    ):
        fail("does not identify a molecule-level feature source.")
    if args.is_atom_bond_targets:
        fail("uses unsupported atom/bond targets.")
    if (
        getattr(args, "atom_descriptors", None) is not None
        or getattr(args, "bond_descriptors", None) is not None
    ):
        fail("uses unsupported atom or bond descriptors/features.")
    if (args.reaction or args.reaction_solvent) and getattr(
        args, "features_generator", None
    ):
        fail("uses a molecule feature generator that omits reaction products.")

    task_names = getattr(args, "task_names", None)
    if (
        not isinstance(task_names, (list, tuple))
        or not task_names
        or any(not isinstance(name, str) for name in task_names)
    ):
        fail("has invalid or empty task names.")

    feature_width = getattr(args, "features_size", None)
    if (
        not isinstance(feature_width, (int, np.integer))
        or isinstance(feature_width, (bool, np.bool_))
        or feature_width < 1
    ):
        fail("has an invalid encoded feature width.")
    feature_width = int(feature_width)
    for metadata_name in (
        "features_generator_metadata",
        "features_source_metadata",
    ):
        metadata = getattr(args, metadata_name, None)
        if metadata is None:
            continue
        metadata_width = (
            metadata.get("total_dimension") if isinstance(metadata, dict) else None
        )
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata_width, (int, np.integer))
            or isinstance(metadata_width, (bool, np.bool_))
            or metadata_width != feature_width
        ):
            fail(
                f"has {metadata_name} whose total dimension does not match "
                f"the encoded feature width {feature_width}."
            )
    return feature_width


def _validate_lgbm_boosters(
    task_boosters: Sequence[Any],
    args: TrainArgs,
    feature_width: int,
    checkpoint_label: str,
    error_type: type = ValueError,
) -> List[Any]:
    """Validates per-task Booster types, objectives, counts, and input widths."""

    def fail(message: str, cause: Exception = None) -> None:
        error = error_type(f'{checkpoint_label} {message}')
        if cause is None:
            raise error
        raise error from cause

    try:
        boosters = list(task_boosters)
    except TypeError as error:
        fail("has invalid task Boosters.", error)
    if len(boosters) != args.num_tasks or any(
        not _is_lightgbm_booster(booster) for booster in boosters
    ):
        fail(f"must contain one LightGBM Booster per task ({args.num_tasks}).")

    expected_objective = (
        "binary" if args.dataset_type == "classification" else "regression"
    )
    for task_index, booster in enumerate(boosters):
        try:
            objective = booster.params.get("objective")
            booster_width = booster.num_feature()
            models_per_iteration = booster.num_model_per_iteration()
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            fail(f"contains an invalid Booster for task {task_index}.", error)
        if objective != expected_objective:
            fail(
                f"contains objective {objective!r} for task {task_index}; "
                f"expected {expected_objective!r}."
            )
        if booster_width != feature_width:
            fail(
                f"contains Booster feature width {booster_width!r} for task "
                f"{task_index}; expected encoded width {feature_width}."
            )
        if models_per_iteration != 1:
            fail(
                f"contains {models_per_iteration!r} models per iteration for task "
                f"{task_index}; expected one scalar-output Booster."
            )
    return boosters


def _load_validated_standard_scaler(
    state: Any,
    *,
    label: str,
    expected_width: int,
    required: bool,
    checkpoint_label: str,
    replace_nan_token: Any = None,
    error_type: type = ValueError,
) -> StandardScaler:
    """Restores one scaler only after validating its presence and parameters."""

    def fail(message: str, cause: Exception = None) -> None:
        error = error_type(
            f'{checkpoint_label} has invalid {label} scaler state: {message}'
        )
        if cause is None:
            raise error
        raise error from cause

    if state is None:
        if required:
            fail("the required scaler is missing.")
        return None
    if not required:
        fail("a scaler is present although training arguments disable it.")
    if not isinstance(state, dict) or set(state) != {"means", "stds"}:
        fail('expected exactly the keys "means" and "stds".')
    try:
        means = np.asarray(state["means"], dtype=float)
        stds = np.asarray(state["stds"], dtype=float)
    except (TypeError, ValueError) as error:
        fail("means and standard deviations must be numeric arrays.", error)
    expected_shape = (expected_width,)
    if means.shape != expected_shape or stds.shape != expected_shape:
        fail(
            f"parameter widths are {means.shape} and {stds.shape}; "
            f"expected {expected_shape}."
        )
    if not np.all(np.isfinite(means)):
        fail("means must contain only finite values.")
    if not np.all(np.isfinite(stds)) or np.any(stds <= 0):
        fail("standard deviations must be finite and positive.")
    return StandardScaler(
        means=means,
        stds=stds,
        replace_nan_token=replace_nan_token,
    )


def _load_validated_lgbm_scalers(
    scaler_states: Any,
    args: TrainArgs,
    feature_width: int,
    checkpoint_label: str,
    error_type: type = ValueError,
) -> Tuple[
    Optional[StandardScaler],
    Optional[StandardScaler],
    None,
    None,
    None,
]:
    """Validates the exact five-slot scaler schema used by LightGBM bundles."""

    required_keys = {
        "data", "features", "atom_descriptor", "bond_descriptor", "atom_bond"
    }
    if not isinstance(scaler_states, dict) or set(scaler_states) != required_keys:
        raise error_type(
            f'{checkpoint_label} has invalid or incomplete scaler state.'
        )

    data_scaler = _load_validated_standard_scaler(
        scaler_states["data"],
        label="target data",
        expected_width=args.num_tasks,
        required=args.dataset_type == "regression",
        checkpoint_label=checkpoint_label,
        error_type=error_type,
    )
    features_scaler = _load_validated_standard_scaler(
        scaler_states["features"],
        label="molecular feature",
        expected_width=feature_width,
        required=bool(args.features_scaling),
        checkpoint_label=checkpoint_label,
        replace_nan_token=0,
        error_type=error_type,
    )
    for key, label in (
        ("atom_descriptor", "atom descriptor"),
        ("bond_descriptor", "bond descriptor"),
        ("atom_bond", "atom/bond target"),
    ):
        if scaler_states[key] is not None:
            raise error_type(
                f'{checkpoint_label} has an unexpected {label} scaler; '
                "that channel is unsupported by LightGBM features-only models."
            )
    return data_scaler, features_scaler, None, None, None


def save_checkpoint_lgbm(
    path: str,
    encoder: MoleculeModelEncoder,
    task_boosters: Sequence[Any],
    scaler: StandardScaler = None,
    features_scaler: StandardScaler = None,
    atom_descriptor_scaler: StandardScaler = None,
    bond_descriptor_scaler: StandardScaler = None,
    atom_bond_scaler: AtomBondScaler = None,
    args: TrainArgs = None,
    model_index: int = 0,
    seed: int = 0,
) -> str:
    """Saves a versioned LightGBM bundle.

    Unlike the legacy LightGBM path, this bundle contains the exact MPN state
    used to generate LightGBM inputs. The file is a trusted pickle: callers
    must never load a checkpoint obtained from an untrusted source.

    :return: The absolute checkpoint path.
    """
    if args is None:
        raise ValueError("LightGBM checkpoints require the training arguments.")
    if not isinstance(encoder, MoleculeModelEncoder):
        raise ValueError("LightGBM checkpoints require a MoleculeModelEncoder.")

    absolute_path = os.path.abspath(path)
    checkpoint_label = f'LightGBM checkpoint "{absolute_path}"'
    feature_width = _validate_lgbm_training_args(args, checkpoint_label)
    if (
        getattr(encoder.encoder, "features_only", None) is not True
        or getattr(encoder.encoder, "use_input_features", None) is not True
    ):
        raise ValueError(
            f"{checkpoint_label} encoder does not match the required "
            "features-only input representation."
        )
    if (
        getattr(encoder.encoder, "number_of_molecules", None)
        != args.number_of_molecules
    ):
        raise ValueError(
            f"{checkpoint_label} encoder molecule count does not match its "
            "training arguments."
        )
    for name, value in (("model_index", model_index), ("seed", seed)):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or (name == "model_index" and value < 0)
        ):
            raise ValueError(f"{checkpoint_label} {name} must be a valid integer.")
    task_boosters = _validate_lgbm_boosters(
        task_boosters,
        args,
        feature_width,
        checkpoint_label,
    )

    if atom_descriptor_scaler is not None:
        raise ValueError(
            f"{checkpoint_label} cannot contain an atom descriptor scaler."
        )
    if bond_descriptor_scaler is not None:
        raise ValueError(
            f"{checkpoint_label} cannot contain a bond descriptor scaler."
        )
    if atom_bond_scaler is not None:
        raise ValueError(
            f"{checkpoint_label} cannot contain an atom/bond target scaler."
        )
    scaler_states = {
        "data": _standard_scaler_state(scaler),
        "features": _standard_scaler_state(features_scaler),
        "atom_descriptor": None,
        "bond_descriptor": None,
        "atom_bond": None,
    }
    # Validate before creating the file so a caller cannot persist a model
    # whose prediction-time normalization differs from its saved arguments.
    _load_validated_lgbm_scalers(
        scaler_states,
        args,
        feature_width,
        checkpoint_label,
    )

    makedirs(absolute_path, isfile=True)
    args_namespace = Namespace(**args.as_dict())
    encoder_state_dict = {
        name: value.detach().cpu().clone()
        for name, value in encoder.encoder.state_dict().items()
    }

    bundle = {
        "format": LIGHTGBM_BUNDLE_FORMAT,
        "version": LIGHTGBM_BUNDLE_VERSION,
        "args": args_namespace,
        "encoder_state_dict": encoder_state_dict,
        "task_boosters": task_boosters,
        "scalers": scaler_states,
        "metadata": {
            "task_names": list(args.task_names),
            "dataset_type": args.dataset_type,
            "encoded_feature_width": feature_width,
            "model_index": int(model_index),
            "seed": int(seed),
        },
    }

    checkpoint_dir = os.path.dirname(absolute_path) or "."
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=checkpoint_dir, prefix=".lgbm-bundle-", suffix=".tmp"
    )
    os.close(file_descriptor)
    try:
        with open(temporary_path, "wb") as checkpoint_file:
            pickle.dump(bundle, checkpoint_file, protocol=pickle.HIGHEST_PROTOCOL)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary_path, absolute_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

    return absolute_path


def load_checkpoint_lgbm(
    path: str, device: torch.device = None
) -> LightGBMModelBundle:
    """Loads a trusted versioned LightGBM bundle and restores its frozen MPN.

    Legacy checkpoints stored a raw LightGBM ``Booster`` separately from a
    scaler-only ``.pt`` file and did not save the randomly initialized encoder.
    Reconstructing such an encoder would silently change predictions, so those
    checkpoints are intentionally rejected with a migration error.
    """
    absolute_path = os.path.abspath(path)
    try:
        with open(absolute_path, "rb") as checkpoint_file:
            bundle = pickle.load(checkpoint_file)
    except (OSError, pickle.PickleError, EOFError, ValueError) as error:
        raise LightGBMCheckpointError(
            f'Could not read LightGBM checkpoint "{absolute_path}". '
            "Only trusted Chemprop LightGBM bundle files may be loaded."
        ) from error

    if (
        not isinstance(bundle, dict)
        or not isinstance(bundle.get("format"), str)
        or bundle.get("format") != LIGHTGBM_BUNDLE_FORMAT
    ):
        raise LightGBMCheckpointError(
            f'LightGBM checkpoint "{absolute_path}" uses the legacy raw-Booster format. '
            "It has no saved MPN encoder state and cannot reproduce its training features. "
            "Retrain the model to create a versioned Chemprop LightGBM bundle."
        )

    version = bundle.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != LIGHTGBM_BUNDLE_VERSION
    ):
        raise LightGBMCheckpointError(
            f'Unsupported LightGBM bundle version {version!r} in "{absolute_path}"; '
            f"this Chemprop build supports version {LIGHTGBM_BUNDLE_VERSION}."
        )

    required_keys = {"args", "encoder_state_dict", "task_boosters", "scalers", "metadata"}
    missing_keys = sorted(required_keys.difference(bundle))
    if missing_keys:
        raise LightGBMCheckpointError(
            f'LightGBM bundle "{absolute_path}" is incomplete; missing keys: '
            f'{", ".join(missing_keys)}.'
        )

    train_args = TrainArgs()
    try:
        train_args.from_dict(vars(bundle["args"]), skip_unsettable=True)
    except (TypeError, AttributeError, ValueError) as error:
        raise LightGBMCheckpointError(
            f'LightGBM bundle "{absolute_path}" contains invalid training arguments.'
        ) from error
    if device is not None:
        train_args.device = device

    checkpoint_label = f'LightGBM bundle "{absolute_path}"'
    feature_width = _validate_lgbm_training_args(
        train_args,
        checkpoint_label,
        LightGBMCheckpointError,
    )
    task_boosters = _validate_lgbm_boosters(
        bundle["task_boosters"],
        train_args,
        feature_width,
        checkpoint_label,
        LightGBMCheckpointError,
    )
    scalers = _load_validated_lgbm_scalers(
        bundle["scalers"],
        train_args,
        feature_width,
        checkpoint_label,
        LightGBMCheckpointError,
    )

    metadata = bundle["metadata"]
    required_metadata_keys = {
        "task_names", "dataset_type", "model_index", "seed"
    }
    if (
        not isinstance(metadata, dict)
        or not required_metadata_keys.issubset(metadata)
    ):
        raise LightGBMCheckpointError(
            f'{checkpoint_label} has invalid or incomplete metadata.'
        )
    task_names_value = metadata["task_names"]
    if (
        not isinstance(task_names_value, (list, tuple))
        or any(not isinstance(name, str) for name in task_names_value)
    ):
        raise LightGBMCheckpointError(
            f'{checkpoint_label} metadata contains invalid task names.'
        )
    task_names = list(task_names_value)
    dataset_type = metadata["dataset_type"]
    if (
        not isinstance(dataset_type, str)
        or task_names != list(train_args.task_names)
        or dataset_type != train_args.dataset_type
    ):
        raise LightGBMCheckpointError(
            f'{checkpoint_label} metadata does not match its training arguments.'
        )
    encoded_feature_width = metadata.get("encoded_feature_width", feature_width)
    if (
        not isinstance(encoded_feature_width, (int, np.integer))
        or isinstance(encoded_feature_width, (bool, np.bool_))
        or encoded_feature_width != feature_width
    ):
        raise LightGBMCheckpointError(
            f'{checkpoint_label} metadata encoded feature width does not match '
            "its training arguments."
        )
    model_index = metadata["model_index"]
    seed = metadata["seed"]
    if (
        not isinstance(model_index, (int, np.integer))
        or isinstance(model_index, (bool, np.bool_))
        or model_index < 0
        or not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
    ):
        raise LightGBMCheckpointError(
            f'{checkpoint_label} metadata contains an invalid model index or seed.'
        )

    # Model input dimensions depend on process-global featurization settings.
    # Restore them before constructing the MPN so fresh-process loads work for
    # reactions and custom atom/bond features as well as ordinary molecules.
    reset_featurization_parameters(logger=logging.getLogger(__name__))
    set_explicit_h(train_args.explicit_h)
    set_adding_hs(getattr(train_args, "adding_h", False))
    set_keeping_atom_map(getattr(train_args, "keeping_atom_map", False))
    if train_args.reaction:
        set_reaction(True, train_args.reaction_mode)
    elif train_args.reaction_solvent:
        set_reaction(True, train_args.reaction_mode)
    if train_args.atom_descriptors == "feature":
        set_extra_atom_fdim(train_args.atom_features_size)
    if train_args.bond_descriptors == "feature":
        set_extra_bond_fdim(train_args.bond_features_size)

    try:
        encoder = MoleculeModelEncoder(train_args).to(train_args.device)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise LightGBMCheckpointError(
            f'{checkpoint_label} cannot reconstruct its saved encoder architecture.'
        ) from error
    encoder_state_dict = bundle["encoder_state_dict"]
    if not isinstance(encoder_state_dict, dict) or any(
        not isinstance(name, str) or not torch.is_tensor(value)
        for name, value in encoder_state_dict.items()
    ):
        raise LightGBMCheckpointError(
            f'{checkpoint_label} contains an invalid MPN state.'
        )
    try:
        encoder.encoder.load_state_dict(encoder_state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise LightGBMCheckpointError(
            f'{checkpoint_label} contains an incompatible MPN state.'
        ) from error
    if (
        getattr(encoder.encoder, "features_only", None) is not True
        or getattr(encoder.encoder, "use_input_features", None) is not True
    ):
        raise LightGBMCheckpointError(
            f'{checkpoint_label} reconstructed an incompatible encoder mode.'
        )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return LightGBMModelBundle(
        encoder=encoder,
        task_boosters=task_boosters,
        train_args=train_args,
        scalers=scalers,
        task_names=task_names,
        dataset_type=dataset_type,
        model_index=int(model_index),
        seed=int(seed),
        checkpoint_path=absolute_path,
    )


def load_checkpoint(
    path: str,
    device: torch.device = None,
    logger: logging.Logger = None,
    strict: bool = True,
) -> MoleculeModel:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param device: Device where the model will be moved.
    :param logger: A logger for recording output.
    :param strict: Whether every model parameter must be present with the
                   expected shape. Disable only for deliberate legacy partial
                   loading; transfer learning should normally use
                   :func:`load_frzn_model` instead.
    :return: The loaded :class:`~chemprop.models.model.MoleculeModel`.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Load model and args
    if parse_version(torch.__version__) >= parse_version("2.6"):
        state = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
    else:
        state = torch.load(path, map_location=lambda storage, loc: storage)
    args = TrainArgs()
    args.from_dict(vars(state["args"]), skip_unsettable=True)
    loaded_state_dict = state["state_dict"]

    if device is not None:
        args.device = device

    # Build model
    model = MoleculeModel(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    unexpected_parameters = []
    mismatched_parameters = []
    for loaded_param_name in loaded_state_dict.keys():
        # Backward compatibility for parameter names
        if re.match(r"(encoder\.encoder\.)([Wc])", loaded_param_name) and not args.reaction_solvent:
            param_name = loaded_param_name.replace("encoder.encoder", "encoder.encoder.0")
        elif re.match(r"(^ffn)", loaded_param_name):
            param_name = loaded_param_name.replace("ffn", "readout")
        else:
            param_name = loaded_param_name

        # Load pretrained parameter, skipping unmatched parameters
        if param_name not in model_state_dict:
            unexpected_parameters.append(loaded_param_name)
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.'
            )
        elif model_state_dict[param_name].shape != loaded_state_dict[loaded_param_name].shape:
            mismatched_parameters.append(
                (
                    loaded_param_name,
                    tuple(loaded_state_dict[loaded_param_name].shape),
                    tuple(model_state_dict[param_name].shape),
                )
            )
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" '
                f"of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding "
                f"model parameter of shape {model_state_dict[param_name].shape}."
            )
        else:
            debug(f'Loading pretrained parameter "{loaded_param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    missing_parameters = sorted(set(model_state_dict) - set(pretrained_state_dict))
    if strict and (unexpected_parameters or mismatched_parameters or missing_parameters):
        issue_parts = []
        if missing_parameters:
            issue_parts.append(f'missing parameters={missing_parameters}')
        if unexpected_parameters:
            issue_parts.append(f'unexpected parameters={sorted(unexpected_parameters)}')
        if mismatched_parameters:
            issue_parts.append(f'shape mismatches={mismatched_parameters}')
        raise ValueError(
            f'Checkpoint {path!r} is incompatible with its saved model '
            f'configuration: {"; ".join(issue_parts)}. Refusing to predict '
            'with randomly initialized or mismatched weights.'
        )

    # Strict loading prevents a corrupt checkpoint from silently leaving
    # random parameters in an otherwise plausible-looking prediction model.
    if strict:
        model.load_state_dict(pretrained_state_dict, strict=True)
    else:
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)

    if args.cuda:
        debug("Moving model to cuda")
    model = model.to(args.device)

    return model


def overwrite_state_dict(
    loaded_param_name: str,
    model_param_name: str,
    loaded_state_dict: collections.OrderedDict,
    model_state_dict: collections.OrderedDict,
    logger: logging.Logger = None,
) -> collections.OrderedDict:
    """
    Overwrites a given parameter in the current model with the loaded model.
    :param loaded_param_name: name of parameter in checkpoint model.
    :param model_param_name: name of parameter in current model.
    :param loaded_state_dict: state_dict for checkpoint model.
    :param model_state_dict: state_dict for current model.
    :param logger: A logger.
    :return: The updated state_dict for the current model.
    """
    debug = logger.debug if logger is not None else print

    if model_param_name not in model_state_dict:
        debug(f'Pretrained parameter "{model_param_name}" cannot be found in model parameters.')

    elif model_state_dict[model_param_name].shape != loaded_state_dict[loaded_param_name].shape:
        debug(
            f'Pretrained parameter "{loaded_param_name}" '
            f"of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding "
            f"model parameter of shape {model_state_dict[model_param_name].shape}."
        )

    else:
        debug(f'Loading pretrained parameter "{model_param_name}".')
        model_state_dict[model_param_name] = loaded_state_dict[loaded_param_name]

    return model_state_dict


def _normalize_frozen_checkpoint_state_dict(
    state_dict: collections.OrderedDict,
    loaded_args: Namespace,
) -> collections.OrderedDict:
    """Normalizes legacy encoder/readout names before strict frozen loading."""
    normalized = collections.OrderedDict()
    reaction_solvent = bool(getattr(loaded_args, "reaction_solvent", False))
    for loaded_name, value in state_dict.items():
        name = loaded_name
        encoder_prefix = "encoder.encoder."
        if not reaction_solvent and name.startswith(encoder_prefix):
            suffix = name[len(encoder_prefix):]
            first_component = suffix.split(".", 1)[0]
            if not first_component.isdigit():
                name = f"{encoder_prefix}0.{suffix}"
        if name.startswith("ffn"):
            name = f"readout{name[3:]}"
        if name in normalized:
            raise ValueError(
                f'Frozen checkpoint contains duplicate normalized parameter "{name}".'
            )
        normalized[name] = value
    return normalized


def _encoder_state_prefixes(args: Namespace) -> Tuple[str, ...]:
    """Returns state-dict prefixes for each independently named MPN encoder."""
    if bool(getattr(args, "features_only", False)):
        return ()
    if bool(getattr(args, "reaction_solvent", False)):
        return ("encoder.encoder.", "encoder.encoder_solvent.")
    number_of_molecules = int(getattr(args, "number_of_molecules", 1))
    return tuple(
        f"encoder.encoder.{index}." for index in range(number_of_molecules)
    )


def _model_encoder_state_prefixes(model: MoleculeModel) -> Tuple[str, ...]:
    """Returns encoder prefixes from the live model rather than CLI metadata."""
    mpn = model.encoder
    if getattr(mpn, 'features_only', False):
        return ()
    if hasattr(mpn, 'encoder_solvent'):
        return ("encoder.encoder.", "encoder.encoder_solvent.")
    encoders = getattr(mpn, 'encoder', None)
    if not isinstance(encoders, nn.ModuleList):
        raise ValueError('Current model has an unsupported MPN encoder layout.')
    return tuple(
        f"encoder.encoder.{index}." for index in range(len(encoders))
    )


def _validate_encoder_mode_compatibility(
    loaded_args: Namespace,
    current_args: Namespace,
) -> None:
    """Rejects transfers between chemically different encoder semantics."""
    loaded_features_only = bool(getattr(loaded_args, "features_only", False))
    current_features_only = bool(getattr(current_args, "features_only", False))
    if loaded_features_only != current_features_only:
        raise ValueError(
            'Checkpoint and current model must either both use features_only '
            'or both use an MPN encoder.'
        )
    loaded_reaction = bool(getattr(loaded_args, "reaction", False))
    current_reaction = bool(getattr(current_args, "reaction", False))
    loaded_reaction_solvent = bool(
        getattr(loaded_args, "reaction_solvent", False)
    )
    current_reaction_solvent = bool(
        getattr(current_args, "reaction_solvent", False)
    )
    if (
        loaded_reaction != current_reaction
        or loaded_reaction_solvent != current_reaction_solvent
    ):
        raise ValueError(
            'Checkpoint and current model must use the same molecule, reaction, '
            'or reaction_solvent encoder mode.'
        )
    if (loaded_reaction or loaded_reaction_solvent) and (
        getattr(loaded_args, "reaction_mode", None)
        != getattr(current_args, "reaction_mode", None)
    ):
        raise ValueError(
            'Checkpoint and current model must use the same reaction_mode.'
        )


def _copy_frozen_prefix(
    loaded_state_dict: collections.OrderedDict,
    model_state_dict: collections.OrderedDict,
    loaded_prefix: str,
    model_prefix: str,
    debug: Callable[[str], None],
) -> List[str]:
    """Strictly copies one complete encoder prefix and returns target keys."""
    loaded_by_suffix = {
        name[len(loaded_prefix):]: name
        for name in loaded_state_dict
        if name.startswith(loaded_prefix)
    }
    model_by_suffix = {
        name[len(model_prefix):]: name
        for name in model_state_dict
        if name.startswith(model_prefix)
    }
    if not loaded_by_suffix:
        raise ValueError(
            f'Frozen checkpoint has no encoder state under "{loaded_prefix}".'
        )
    if not model_by_suffix:
        raise ValueError(
            f'Current model has no encoder state under "{model_prefix}".'
        )
    if set(loaded_by_suffix) != set(model_by_suffix):
        missing = sorted(set(model_by_suffix) - set(loaded_by_suffix))
        extra = sorted(set(loaded_by_suffix) - set(model_by_suffix))
        raise ValueError(
            'Frozen encoder architectures do not match for '
            f'{loaded_prefix!r} -> {model_prefix!r}: missing suffixes={missing}; '
            f'extra suffixes={extra}.'
        )

    mismatches = []
    for suffix in sorted(model_by_suffix):
        loaded_name = loaded_by_suffix[suffix]
        model_name = model_by_suffix[suffix]
        loaded_value = loaded_state_dict[loaded_name]
        model_value = model_state_dict[model_name]
        if not hasattr(loaded_value, "shape") or loaded_value.shape != model_value.shape:
            mismatches.append(
                (
                    loaded_name,
                    getattr(loaded_value, "shape", None),
                    model_name,
                    model_value.shape,
                )
            )
    if mismatches:
        raise ValueError(f'Frozen encoder parameter shapes do not match: {mismatches}.')

    copied_names = []
    for suffix in sorted(model_by_suffix):
        loaded_name = loaded_by_suffix[suffix]
        model_name = model_by_suffix[suffix]
        debug(f'Loading frozen parameter "{loaded_name}" as "{model_name}".')
        model_state_dict[model_name] = loaded_state_dict[loaded_name]
        copied_names.append(model_name)
    return copied_names


def load_checkpoint_for_training(
    path: str,
    current_args: TrainArgs,
    device: torch.device = None,
    logger: logging.Logger = None,
) -> MoleculeModel:
    """Initializes a model built from current args with compatible checkpoint state.

    Prediction loading must reconstruct the exact saved architecture and is
    handled by :func:`load_checkpoint`.  Warm-start training instead needs the
    current architecture (for example a new task count or FFN) and transfers
    only compatible non-encoder state after strictly validating complete MPN
    encoders.
    """
    debug = logger.debug if logger is not None else print
    info = logger.info if logger is not None else print
    if current_args is None:
        raise ValueError('current_args are required for warm-start training.')

    if parse_version(torch.__version__) >= parse_version("2.6"):
        checkpoint = torch.load(
            path,
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
    else:
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage,
        )
    if 'state_dict' not in checkpoint or 'args' not in checkpoint:
        raise ValueError(f'Checkpoint {path!r} is missing state_dict or args.')

    loaded_args = checkpoint['args']
    _validate_encoder_mode_compatibility(loaded_args, current_args)
    loaded_state_dict = _normalize_frozen_checkpoint_state_dict(
        checkpoint['state_dict'], loaded_args,
    )

    model = MoleculeModel(current_args)
    model_state_dict = model.state_dict()
    current_prefixes = _model_encoder_state_prefixes(model)
    expected_current_prefixes = _encoder_state_prefixes(current_args)
    if current_prefixes != expected_current_prefixes:
        raise ValueError(
            'Warm-start model encoder layout does not match current_args: '
            f'model={current_prefixes}, args={expected_current_prefixes}.'
        )
    loaded_prefixes = _encoder_state_prefixes(loaded_args)

    if (
        bool(getattr(current_args, 'mpn_shared', False))
        and len(loaded_prefixes) > 1
        and not bool(getattr(loaded_args, 'mpn_shared', False))
    ):
        raise ValueError(
            'An independent-encoder checkpoint cannot initialize a shared-MPN '
            'model because distinct source values target the same parameters.'
        )
    if bool(getattr(current_args, 'reaction_solvent', False)):
        if len(loaded_prefixes) != 2 or len(current_prefixes) != 2:
            raise ValueError('reaction_solvent warm starts require both encoders.')
        prefix_mappings = tuple(zip(loaded_prefixes, current_prefixes))
    elif len(loaded_prefixes) == 1:
        prefix_mappings = tuple(
            (loaded_prefixes[0], target_prefix)
            for target_prefix in current_prefixes
        )
    elif len(loaded_prefixes) == len(current_prefixes):
        prefix_mappings = tuple(zip(loaded_prefixes, current_prefixes))
    else:
        raise ValueError(
            f'Warm-start checkpoint has {len(loaded_prefixes)} encoders but the '
            f'current model has {len(current_prefixes)}; only 1-to-N or '
            'equal-count transfer is supported.'
        )

    copied_names = []
    for loaded_prefix, current_prefix in prefix_mappings:
        copied_names.extend(_copy_frozen_prefix(
            loaded_state_dict,
            model_state_dict,
            loaded_prefix,
            current_prefix,
            debug,
        ))

    source_encoder_names = {
        name
        for name in loaded_state_dict
        if any(name.startswith(prefix) for prefix in loaded_prefixes)
    }
    skipped_unexpected = []
    skipped_shapes = []
    for name, value in loaded_state_dict.items():
        if name in source_encoder_names:
            continue
        if name not in model_state_dict:
            skipped_unexpected.append(name)
            continue
        if (
            not hasattr(value, 'shape')
            or value.shape != model_state_dict[name].shape
        ):
            skipped_shapes.append(name)
            continue
        model_state_dict[name] = value
        copied_names.append(name)

    model.load_state_dict(model_state_dict, strict=True)
    target_device = current_args.device if device is None else device
    model = model.to(target_device)
    debug(
        f'Warm-started {len(copied_names)} state entries from {path!r}; '
        f'skipped {len(skipped_unexpected)} unexpected and '
        f'{len(skipped_shapes)} shape-mismatched entries.'
    )
    if skipped_unexpected:
        info(
            'Warm-start checkpoint entries absent from the current model were '
            f'skipped: {sorted(skipped_unexpected)}'
        )
    if skipped_shapes:
        info(
            'Warm-start checkpoint entries with changed shapes were skipped: '
            f'{sorted(skipped_shapes)}'
        )
    return model


def _first_frozen_block_parameter_ids(
    module: nn.Module,
    linear_layer_count: int,
    label: str,
) -> Set[int]:
    """Selects complete leading Linear/PReLU blocks by module topology."""
    selected = set()
    linear_layers_seen = 0
    for _, child in module.named_modules(remove_duplicate=False):
        if child is module:
            continue
        if isinstance(child, nn.Linear):
            if linear_layers_seen >= linear_layer_count:
                break
            linear_layers_seen += 1
            selected.update(id(parameter) for parameter in child.parameters(recurse=False))
        elif 0 < linear_layers_seen <= linear_layer_count:
            # A trainable activation (notably PReLU) between selected Linear
            # layers is part of the frozen computation and must be reproduced.
            selected.update(id(parameter) for parameter in child.parameters(recurse=False))
    if linear_layers_seen != linear_layer_count:
        raise ValueError(
            f'{label} has only {linear_layers_seen} transferable Linear layers; '
            f'{linear_layer_count} were requested.'
        )
    return selected


def _frozen_ffn_parameter_names(
    model: MoleculeModel,
    linear_layer_count: int,
) -> List[str]:
    """Returns every state alias belonging to the requested leading FFN layers."""
    if linear_layer_count <= 0:
        return []

    selected_ids: Set[int] = set()
    if isinstance(model.readout, nn.Sequential):
        selected_ids.update(_first_frozen_block_parameter_ids(
            model.readout, linear_layer_count, "Molecular readout",
        ))
    elif isinstance(model.readout, MultiReadout):
        if (
            model.readout.atom_ffn_base is not None
            or model.readout.bond_ffn_base is not None
        ):
            if (
                model.readout.atom_ffn_base is None
                or model.readout.bond_ffn_base is None
            ):
                raise ValueError('Atom/bond shared FFN bases are incomplete.')
            groups = (
                (model.readout.atom_ffn_base, "Shared atom readout"),
                (model.readout.bond_ffn_base, "Shared bond readout"),
            )
        else:
            groups = tuple(
                (ffn.ffn, f"Atom/bond task readout {index}")
                for index, ffn in enumerate(model.readout.ffn_list)
            )
        if not groups:
            raise ValueError('Atom/bond model has no FFN groups to freeze.')
        for group, label in groups:
            selected_ids.update(_first_frozen_block_parameter_ids(
                group, linear_layer_count, label,
            ))
    else:
        raise TypeError(
            f'Unsupported readout type for frozen transfer: {type(model.readout)!r}.'
        )

    return [
        name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if id(parameter) in selected_ids
    ]


def _frozen_ffn_source_name(
    model: MoleculeModel,
    target_name: str,
    loaded_state_dict: collections.OrderedDict,
) -> str:
    """Finds the checkpoint alias supplying one target FFN state entry.

    Shared atom/bond FFN parameters appear both under their canonical base and
    once under every task head.  If the target has additional task aliases,
    load all of them from the canonical source so a later alias cannot
    overwrite the transferred value during ``load_state_dict``.
    """
    match = re.fullmatch(r'readout\.ffn_list\.(\d+)\.ffn\.(.+)', target_name)
    if match is not None and isinstance(model.readout, MultiReadout):
        task_index = int(match.group(1))
        task_kind = 'atom' if task_index < len(model.atom_targets) else 'bond'
        canonical_name = f'readout.{task_kind}_ffn_base.{match.group(2)}'
        if canonical_name in loaded_state_dict:
            # Prefer the canonical shared base.  A same-numbered source alias
            # can refer to the opposite task kind when the atom/bond task
            # boundary changed between checkpoint and current model.
            return canonical_name
    return target_name if target_name in loaded_state_dict else None


def load_frzn_model(
    model: torch.nn,
    path: str,
    current_args: Namespace = None,
    cuda: bool = None,
    logger: logging.Logger = None,
) -> MoleculeModel:
    """
    Loads a model checkpoint.
    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MoleculeModel.
    """
    debug = logger.debug if logger is not None else print

    if parse_version(torch.__version__) >= parse_version("2.6"):
        loaded_mpnn_model = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
    else:
        loaded_mpnn_model = torch.load(path, map_location=lambda storage, loc: storage)
    loaded_state_dict = loaded_mpnn_model["state_dict"]
    loaded_args = loaded_mpnn_model["args"]

    if current_args is None:
        raise ValueError('current_args are required when loading frozen parameters.')
    loaded_state_dict = _normalize_frozen_checkpoint_state_dict(
        loaded_state_dict, loaded_args,
    )
    model_state_dict = model.state_dict()

    _validate_encoder_mode_compatibility(loaded_args, current_args)

    loaded_prefixes = _encoder_state_prefixes(loaded_args)
    expected_current_prefixes = _encoder_state_prefixes(current_args)
    current_prefixes = _model_encoder_state_prefixes(model)
    if current_prefixes != expected_current_prefixes:
        raise ValueError(
            'Current model encoder layout does not match current_args: '
            f'model={current_prefixes}, args={expected_current_prefixes}.'
        )
    if (
        bool(getattr(current_args, "mpn_shared", False))
        and len(loaded_prefixes) > 1
        and not bool(getattr(loaded_args, "mpn_shared", False))
    ):
        raise ValueError(
            'A checkpoint with independent encoders cannot be frozen into a '
            'shared-MPN model because distinct values would target one parameter.'
        )
    if len(loaded_prefixes) == 1:
        if len(current_prefixes) > 1 and current_args.frzn_ffn_layers > 0:
            raise ValueError(
                'Frozen FFN layers require the checkpoint and current model to '
                'have the same number of molecule encoders.'
            )
        target_prefixes = (
            current_prefixes[:1]
            if current_args.freeze_first_only
            else current_prefixes
        )
        prefix_mappings = tuple(
            (loaded_prefixes[0], target_prefix)
            for target_prefix in target_prefixes
        )
    elif len(loaded_prefixes) == len(current_prefixes):
        if current_args.freeze_first_only and bool(
            getattr(current_args, 'reaction_solvent', False)
        ):
            prefix_mappings = ((loaded_prefixes[0], current_prefixes[0]),)
        elif current_args.freeze_first_only:
            raise ValueError(
                'freeze_first_only requires a frozen checkpoint with exactly '
                'one molecule encoder.'
            )
        else:
            prefix_mappings = tuple(zip(loaded_prefixes, current_prefixes))
    else:
        raise ValueError(
            f'Frozen checkpoint has {len(loaded_prefixes)} molecule encoders but '
            f'the current model has {len(current_prefixes)}; only 1-to-N or '
            'equal-count transfer is supported.'
        )

    copied_encoder_names = []
    for loaded_prefix, current_prefix in prefix_mappings:
        copied_encoder_names.extend(_copy_frozen_prefix(
            loaded_state_dict,
            model_state_dict,
            loaded_prefix,
            current_prefix,
            debug,
        ))

    copied_ffn_names = _frozen_ffn_parameter_names(
        model, current_args.frzn_ffn_layers,
    )
    if copied_ffn_names:
        loaded_atom_bond = bool(
            getattr(loaded_args, 'is_atom_bond_targets', False)
        )
        current_atom_bond = bool(
            getattr(current_args, 'is_atom_bond_targets', False)
        )
        if loaded_atom_bond != current_atom_bond:
            raise ValueError(
                'Frozen FFN transfer cannot switch between molecule-level and '
                'atom/bond target readouts.'
            )
        if current_atom_bond:
            loaded_shared = bool(
                getattr(loaded_args, 'shared_atom_bond_ffn', True)
            )
            current_shared = bool(
                getattr(current_args, 'shared_atom_bond_ffn', True)
            )
            if current_shared and not loaded_shared:
                raise ValueError(
                    'Independent atom/bond FFNs cannot initialize one shared '
                    'frozen FFN because their source values may differ.'
                )
            if not loaded_shared and (
                list(getattr(loaded_args, 'atom_targets', []))
                != list(getattr(current_args, 'atom_targets', []))
                or list(getattr(loaded_args, 'bond_targets', []))
                != list(getattr(current_args, 'bond_targets', []))
            ):
                raise ValueError(
                    'Frozen independent atom/bond FFNs require identical ordered '
                    'atom and bond target names.'
                )

    ffn_source_names = {
        name: _frozen_ffn_source_name(model, name, loaded_state_dict)
        for name in copied_ffn_names
    }
    missing_ffn_names = sorted(
        name for name, source_name in ffn_source_names.items()
        if source_name is None
    )
    if missing_ffn_names:
        raise ValueError(
            f'Frozen checkpoint is missing requested FFN state: {missing_ffn_names}.'
        )
    mismatched_ffn = [
        (
            name,
            getattr(loaded_state_dict[ffn_source_names[name]], "shape", None),
            model_state_dict[name].shape,
        )
        for name in copied_ffn_names
        if (
            not hasattr(loaded_state_dict[ffn_source_names[name]], "shape")
            or loaded_state_dict[ffn_source_names[name]].shape
            != model_state_dict[name].shape
        )
    ]
    if mismatched_ffn:
        raise ValueError(f'Frozen FFN parameter shapes do not match: {mismatched_ffn}.')
    for name in copied_ffn_names:
        source_name = ffn_source_names[name]
        debug(f'Loading frozen FFN parameter "{source_name}" as "{name}".')
        model_state_dict[name] = loaded_state_dict[source_name]

    # Validate every mapping before mutating the live model, then apply all
    # values atomically from the caller's perspective.
    model.load_state_dict(model_state_dict, strict=True)

    frozen_parameter_names = set(copied_encoder_names + copied_ffn_names)
    frozen_parameter_ids = {
        id(parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if name in frozen_parameter_names
    }
    for parameter in model.parameters():
        if id(parameter) in frozen_parameter_ids:
            parameter.requires_grad_(False)

    return model


def load_scalers(
    path: str,
) -> Tuple[StandardScaler, StandardScaler, StandardScaler, StandardScaler, List[StandardScaler]]:
    """
    Loads the scalers a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the data :class:`~chemprop.data.scaler.StandardScaler`
             and features :class:`~chemprop.data.scaler.StandardScaler`.
    """
    if parse_version(torch.__version__) >= parse_version("2.6"):
        state = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
    else:
        state = torch.load(path, map_location=lambda storage, loc: storage)

    if state["data_scaler"] is not None:
        scaler = StandardScaler(state["data_scaler"]["means"], state["data_scaler"]["stds"])
    else:
        scaler = None

    if state["features_scaler"] is not None:
        features_scaler = StandardScaler(
            state["features_scaler"]["means"], state["features_scaler"]["stds"], replace_nan_token=0
        )
    else:
        features_scaler = None

    if "atom_descriptor_scaler" in state.keys() and state["atom_descriptor_scaler"] is not None:
        atom_descriptor_scaler = StandardScaler(
            state["atom_descriptor_scaler"]["means"],
            state["atom_descriptor_scaler"]["stds"],
            replace_nan_token=0,
        )
    else:
        atom_descriptor_scaler = None

    if "bond_descriptor_scaler" in state.keys() and state["bond_descriptor_scaler"] is not None:
        bond_descriptor_scaler = StandardScaler(
            state["bond_descriptor_scaler"]["means"],
            state["bond_descriptor_scaler"]["stds"],
            replace_nan_token=0,
        )
    else:
        bond_descriptor_scaler = None

    if "atom_bond_scaler" in state.keys() and state["atom_bond_scaler"] is not None:
        atom_bond_scaler =AtomBondScaler(
            state["atom_bond_scaler"]["means"],
            state["atom_bond_scaler"]["stds"],
            replace_nan_token=0,
            n_atom_targets=len(state["args"].atom_targets),
            n_bond_targets=len(state["args"].bond_targets),
        )
    else:
        atom_bond_scaler = None

    return scaler, features_scaler, atom_descriptor_scaler, bond_descriptor_scaler, atom_bond_scaler


def load_args(path: str) -> TrainArgs:
    """
    Loads the arguments a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: The :class:`~chemprop.args.TrainArgs` object that the model was trained with.
    """
    args = TrainArgs()
    if parse_version(torch.__version__) >= parse_version("2.6"):
        args.from_dict(
            vars(torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)["args"]),
            skip_unsettable=True,
        )
    else:
        args.from_dict(
            vars(torch.load(path, map_location=lambda storage, loc: storage)["args"]),
            skip_unsettable=True,
        )

    return args


def load_task_names(path: str) -> List[str]:
    """
    Loads the task names a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A list of the task names that the model was trained with.
    """
    return load_args(path).task_names


def build_optimizer(model: nn.Module, args: TrainArgs) -> Optimizer:
    """
    Builds a PyTorch Optimizer.

    :param model: The model to optimize.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing optimizer arguments.
    :return: An initialized Optimizer.
    """
    params = [{"params": model.parameters(), "lr": args.init_lr, "weight_decay": 0}]

    return Adam(params)


def build_lr_scheduler(
    optimizer: Optimizer, args: TrainArgs, total_epochs: List[int] = None
) -> _LRScheduler:
    """
    Builds a PyTorch learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing learning rate arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=total_epochs or [args.epochs] * args.num_lrs,
        steps_per_epoch=max(1, ceil(args.train_data_size / args.batch_size)),
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr],
    )


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    Existing handlers created by this function are replaced so repeated jobs
    can safely reuse a logger name with a different output directory.

    The stream handler prints to the screen depending on the value of :code:`quiet`.
    One file handler (:code:`verbose.log`) saves all logs, the other (:code:`quiet.log`) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e., print only important info).
    :return: The logger.
    """

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Long-lived callers such as the Web application reuse ``train`` for many
    # jobs.  Retaining the first job's handlers writes to a deleted temporary
    # directory, leaks file descriptors, and prevents the next progress bar
    # from seeing its log.  Preserve handlers owned by embedding applications,
    # but close and replace every handler created here.
    for handler in list(logger.handlers):
        if getattr(handler, "_chemprop_handler", False):
            logger.removeHandler(handler)
            handler.close()

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    ch._chemprop_handler = True
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, "verbose.log"))
        fh_v._chemprop_handler = True
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, "quiet.log"))
        fh_q._chemprop_handler = True
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def timeit(logger_name: str = None) -> Callable[[Callable], Callable]:
    """
    Creates a decorator which wraps a function with a timer that prints the elapsed time.

    :param logger_name: The name of the logger used to record output. If None, uses :code:`print` instead.
    :return: A decorator which wraps a function with a timer that prints the elapsed time.
    """

    def timeit_decorator(func: Callable) -> Callable:
        """
        A decorator which wraps a function with a timer that prints the elapsed time.

        :param func: The function to wrap with the timer.
        :return: The function wrapped with the timer.
        """

        @wraps(func)
        def wrap(*args, **kwargs) -> Any:
            start_time = time()
            result = func(*args, **kwargs)
            delta = timedelta(seconds=round(time() - start_time))
            info = logging.getLogger(logger_name).info if logger_name is not None else print
            info(f"Elapsed time = {delta}")

            return result

        return wrap

    return timeit_decorator


def save_smiles_splits(
    data_path: str,
    save_dir: str,
    task_names: List[str] = None,
    features_path: List[str] = None,
    constraints_path: str = None,
    train_data: MoleculeDataset = None,
    val_data: MoleculeDataset = None,
    test_data: MoleculeDataset = None,
    smiles_columns: List[str] = None,
    loss_function: str = None,
    logger: logging.Logger = None,
) -> None:
    """
    Saves a csv file with train/val/test splits of target data and additional features.
    Also saves indices of train/val/test split as a pickle file. Pickle file does not support repeated entries
    with the same SMILES or entries entered from a path other than the main data path, such as a separate test path.

    :param data_path: Path to data CSV file.
    :param save_dir: Path where pickle files will be saved.
    :param task_names: List of target names for the model as from the function get_task_names().
        If not provided, will use datafile header entries.
    :param features_path: List of path(s) to files with additional molecule features.
    :param constraints_path: Path to constraints applied to atomic/bond properties prediction.
    :param train_data: Train :class:`~chemprop.data.data.MoleculeDataset`.
    :param val_data: Validation :class:`~chemprop.data.data.MoleculeDataset`.
    :param test_data: Test :class:`~chemprop.data.data.MoleculeDataset`.
    :param smiles_columns: The name of the column containing SMILES. By default, uses the first column.
    :param loss_function: The loss function to be used in training.
    :param logger: A logger for recording output.
    """
    makedirs(save_dir)

    info = logger.info if logger is not None else print
    save_split_indices = True

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(path=data_path, smiles_columns=smiles_columns)

    with open(data_path) as f:
        reader = csv.DictReader(f)

        indices_by_smiles = {}
        for i, row in enumerate(tqdm(reader)):
            smiles = tuple([row[column] for column in smiles_columns])
            if smiles in indices_by_smiles:
                save_split_indices = False
                info(
                    "Warning: Repeated SMILES found in data, pickle file of split indices cannot distinguish entries and will not be generated."
                )
                break
            indices_by_smiles[smiles] = i

    if task_names is None:
        task_names = get_task_names(
            path=data_path,
            smiles_columns=smiles_columns,
            loss_function=loss_function,
            )

    if loss_function == "quantile_interval":
        num_tasks = len(task_names) // 2
        task_names = task_names[:num_tasks]

    features_header = []
    if features_path is not None:
        extension_sets = set([os.path.splitext(feat_path)[1] for feat_path in features_path])
        if extension_sets == {'.csv'}:
            for feat_path in features_path:
                with open(feat_path, "r") as f:
                    reader = csv.reader(f)
                    feat_header = next(reader)
                    features_header.extend(feat_header)

    if constraints_path is not None:
        with open(constraints_path, "r") as f:
            reader = csv.reader(f)
            constraints_header = next(reader)

    all_split_indices = []
    for dataset, name in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        if dataset is None:
            continue

        with open(os.path.join(save_dir, f"{name}_smiles.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            if smiles_columns[0] == "":
                writer.writerow(["smiles"])
            else:
                writer.writerow(smiles_columns)
            for smiles in dataset.smiles():
                writer.writerow(smiles)

        with open(os.path.join(save_dir, f"{name}_full.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(smiles_columns + task_names)
            dataset_targets = dataset.targets()
            for i, smiles in enumerate(dataset.smiles()):
                targets = [x.tolist() if isinstance(x, np.ndarray) else x for x in dataset_targets[i]]
                # correct the number of targets when running quantile regression
                targets = targets[:len(task_names)]
                writer.writerow(smiles + targets)

        if features_path is not None:
            dataset_features = dataset.features()
            if extension_sets == {'.csv'}:
                with open(os.path.join(save_dir, f"{name}_features.csv"), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(features_header)
                    writer.writerows(dataset_features)
            else:
                np.save(os.path.join(save_dir, f"{name}_features.npy"), dataset_features)

        if constraints_path is not None:
            dataset_constraints = [d.raw_constraints for d in dataset._data]
            with open(os.path.join(save_dir, f"{name}_constraints.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(constraints_header)
                writer.writerows(dataset_constraints)

        if save_split_indices:
            split_indices = []
            for smiles in dataset.smiles():
                index = indices_by_smiles.get(tuple(smiles))
                if index is None:
                    save_split_indices = False
                    info(
                        f"Warning: SMILES string in {name} could not be found in data file, and "
                        "likely came from a secondary data file. The pickle file of split indices "
                        "can only indicate indices for a single file and will not be generated."
                    )
                    break
                split_indices.append(index)
            else:
                split_indices.sort()
                all_split_indices.append(split_indices)

        if name == "train":
            data_weights = dataset.data_weights()
            if any([w != 1 for w in data_weights]):
                with open(os.path.join(save_dir, f"{name}_weights.csv"), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["data weights"])
                    for weight in data_weights:
                        writer.writerow([weight])

    split_indices_path = os.path.join(save_dir, "split_indices.pckl")
    if save_split_indices:
        descriptor, temporary_path = tempfile.mkstemp(
            dir=save_dir, prefix='.split-indices-', suffix='.tmp',
        )
        try:
            with os.fdopen(descriptor, "wb") as split_file:
                pickle.dump(
                    all_split_indices,
                    split_file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                split_file.flush()
                os.fsync(split_file.fileno())
            os.replace(temporary_path, split_indices_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
    elif os.path.exists(split_indices_path):
        # Never leave a valid-looking index file from an earlier invocation
        # when the current splits contain duplicate or external SMILES that
        # cannot be represented unambiguously.
        os.unlink(split_indices_path)


def update_prediction_args(
    predict_args: PredictArgs,
    train_args: TrainArgs,
    missing_to_defaults: bool = True,
    validate_feature_sources: bool = True,
) -> None:
    """
    Updates prediction arguments with training arguments loaded from a checkpoint file.
    If an argument is present in both, the prediction argument will be used.

    Also raises errors for situations where the prediction arguments and training arguments
    are different but must match for proper function.

    :param predict_args: The :class:`~chemprop.args.PredictArgs` object containing the arguments to use for making predictions.
    :param train_args: The :class:`~chemprop.args.TrainArgs` object containing the arguments used to train the model previously.
    :param missing_to_defaults: Whether to replace missing training arguments with the current defaults for :class: `~chemprop.args.TrainArgs`.
        This is used for backwards compatibility.
    :param validate_feature_sources: Indicates whether the feature sources (from path or generator) are checked for consistency between
        the training and prediction arguments. This is not necessary for fingerprint generation, where molecule features are not used.
    """
    for key, value in vars(train_args).items():
        if not hasattr(predict_args, key):
            setattr(predict_args, key, value)

    if missing_to_defaults:
        # If a default argument would cause different behavior than occurred in legacy checkpoints before the argument existed,
        # then that argument must be included in the `override_defaults` dictionary to force the legacy behavior.
        override_defaults = {
            "bond_descriptors_scaling": False,
            "no_bond_descriptors_scaling": True,
            "atom_descriptors_scaling": False,
            "no_atom_descriptors_scaling": True,
        }
        default_train_args = TrainArgs().parse_args(
            ["--data_path", None, "--dataset_type", str(train_args.dataset_type)]
        )
        for key, value in vars(default_train_args).items():
            if not hasattr(predict_args, key):
                setattr(predict_args, key, override_defaults.get(key, value))

    # Same number of molecules must be used in training as in making predictions
    if train_args.number_of_molecules != predict_args.number_of_molecules and not (
        isinstance(predict_args, FingerprintArgs)
        and predict_args.fingerprint_type == "MPN"
        and predict_args.mpn_shared
        and predict_args.number_of_molecules == 1
    ):
        raise ValueError(
            "A different number of molecules was used in training "
            "model than is specified for prediction. This is only supported for models with shared MPN networks"
            f"and a fingerprint type of MPN. {train_args.number_of_molecules} smiles fields must be provided."
        )

    # if atom or bond features were scaled, the same must be done during prediction
    if train_args.features_scaling != predict_args.features_scaling:
        raise ValueError(
            "If scaling of the additional features was done during training, the "
            "same must be done during prediction."
        )

    # If atom descriptors were used during training, they must be used when predicting and vice-versa
    if train_args.atom_descriptors != predict_args.atom_descriptors:
        raise ValueError(
            "The use of atom descriptors is inconsistent between training and prediction. "
            "If atom descriptors were used during training, they must be specified again "
            "during prediction using the same type of descriptors as before. "
            "If they were not used during training, they cannot be specified during prediction."
        )

    # If bond features were used during training, they must be used when predicting and vice-versa
    if train_args.bond_descriptors != predict_args.bond_descriptors:
        raise ValueError(
            "The use of bond descriptors is inconsistent between training and prediction. "
            "If bond descriptors were used during training, they must be specified again "
            "during prediction using the same type of descriptors as before. "
            "If they were not used during training, they cannot be specified during prediction."
        )

    # If constraints were used during training, they must be used when predicting and vice-versa
    if (train_args.constraints_path is None) != (predict_args.constraints_path is None):
        raise ValueError(
            "The use of constraints is different between training and prediction. If you applied constraints "
            "for training, please specify a path to new constraints for prediction."
        )

    # If features were used during training, they must be used when predicting.
    # External feature files are intentionally compared by presence because the
    # prediction rows normally come from a different file. Generated features,
    # however, are part of the model input schema and their order is significant.
    if validate_feature_sources:
        train_features_path = getattr(train_args, "features_path", None)
        predict_features_path = getattr(predict_args, "features_path", None)
        train_features_generator = getattr(train_args, "features_generator", None)
        predict_features_generator = getattr(predict_args, "features_generator", None)
        if ((train_features_path is None) != (predict_features_path is None)) or (
            (train_features_generator is None)
            != (predict_features_generator is None)
        ):
            raise ValueError(
                "Features were used during training so they must be specified again during "
                "prediction using the same type of features as before "
                "(with either --features_generator or --features_path "
                "and using --no_features_scaling if applicable)."
            )
        if (
            train_features_generator is not None
            and predict_features_generator is not None
            and list(train_features_generator) != list(predict_features_generator)
        ):
            raise ValueError(
                "The ordered feature generators used for prediction must exactly "
                "match those used during training."
            )


def multitask_mean(
    scores: np.ndarray,
    metric: str,
    axis: int = None,
    ignore_nan_metrics: bool = False,
) -> float:
    """
    A function for combining the metric scores across different
    model tasks into a single score. When the metric being used
    is one that varies with the magnitude of the task (such as RMSE),
    a geometric mean is used, otherwise a more typical arithmetic mean
    is used. This prevents a task with a larger magnitude from dominating
    over one with a smaller magnitude (e.g., temperature and pressure).

    :param scores: The scores from different tasks for a single metric.
    :param metric: The metric used to generate the scores.
    :param axis: The axis along which to take the mean.
    :param ignore_nan_metrics: Ignore invalid task metrics (NaNs) when computing average metrics across tasks.
    :return: The combined score across the tasks.
    """
    scale_dependent_metrics = ["rmse", "mae", "mse", "bounded_rmse", "bounded_mae", "bounded_mse", "quantile"]
    nonscale_dependent_metrics = [
        "auc", "prc-auc", "r2", "accuracy", "cross_entropy",
        "binary_cross_entropy", "sid", "wasserstein", "f1", "mcc", "recall",
        "precision", "balanced_accuracy",
    ]

    mean_fn = np.nanmean if ignore_nan_metrics else np.mean

    if metric in scale_dependent_metrics:
        return np.exp(mean_fn(np.log(scores), axis=axis))
    elif metric in nonscale_dependent_metrics:
        return mean_fn(scores, axis=axis)
    else:
        raise NotImplementedError(
            f"The metric used, {metric}, has not been added to the list of\
                metrics that are scale-dependent or not scale-dependent.\
                This metric must be added to the appropriate list in the multitask_mean\
                function in `chemprop/utils.py` in order to be used."
        )
