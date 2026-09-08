from numbers import Integral
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np

from chemprop.data import MoleculeDataset


def _validate_task_counts(natom_targets: int, nbond_targets: int) -> int:
    """Validates atom/bond task counts and returns their sum."""
    for name, value in (
        ("natom_targets", natom_targets),
        ("nbond_targets", nbond_targets),
    ):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer; got {value!r}.")
    return int(natom_targets + nbond_targets)


def _single_molecule_counts(
    counts: Sequence[Sequence[int]], data_size: int, name: str
) -> np.ndarray:
    """Returns per-row atom/bond counts for the supported one-molecule schema."""
    count_array = np.asarray(counts)
    if count_array.ndim == 2 and count_array.shape[1] == 1:
        count_array = count_array[:, 0]
    elif count_array.ndim != 1:
        raise ValueError(
            f"{name} reshaping supports exactly one molecule per datapoint; "
            f"received count shape {count_array.shape}."
        )
    if len(count_array) != data_size:
        raise ValueError(
            f"{name} count length {len(count_array)} does not match data size "
            f"{data_size}."
        )
    try:
        numeric_counts = count_array.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} counts must be non-negative integers.") from exc
    if np.any(numeric_counts < 0) or np.any(count_array != numeric_counts):
        raise ValueError(f"{name} counts must be non-negative integers.")
    return numeric_counts


def _split_flat_values(values: np.ndarray, counts: np.ndarray, axis: int = 0):
    """Splits values by datapoint, including the zero-datapoint edge case."""
    if len(counts) == 0:
        return []
    return np.split(values, np.cumsum(counts)[:-1], axis=axis)


def validate_task_masks(
    masks: Sequence[Sequence[bool]],
    num_tasks: int = None,
    expected_lengths: Sequence[int] = None,
) -> List[np.ndarray]:
    """Validates task-major masks without coercing ragged tasks to an array.

    Atom and bond tasks normally have different flattened lengths. NumPy 2
    rejects a direct ``np.asarray`` conversion of such ragged input, so each
    task is validated independently.
    """
    try:
        task_masks = list(masks)
    except TypeError as error:
        raise ValueError('Task masks must be a task-major sequence.') from error
    if num_tasks is not None and len(task_masks) != num_tasks:
        raise ValueError(
            f'Expected {num_tasks} task masks but received {len(task_masks)}.'
        )
    if expected_lengths is not None and len(expected_lengths) != len(task_masks):
        raise ValueError('Expected mask lengths must contain one entry per task.')

    validated = []
    for task_index, task_mask in enumerate(task_masks):
        mask_array = np.asarray(task_mask)
        if mask_array.ndim != 1:
            raise ValueError(f'Task mask {task_index} must be one-dimensional.')
        if not (
            np.issubdtype(mask_array.dtype, np.bool_)
            or np.all(np.isin(mask_array, [0, 1]))
        ):
            raise ValueError(f'Task mask {task_index} must contain only booleans.')
        mask_array = mask_array.astype(bool, copy=False)
        if (
            expected_lengths is not None
            and len(mask_array) != expected_lengths[task_index]
        ):
            raise ValueError(
                f'Task mask {task_index} has length {len(mask_array)}; expected '
                f'{expected_lengths[task_index]}.'
            )
        validated.append(mask_array)
    return validated


def _coerce_atom_bond_values(
    values: Sequence[Sequence[Sequence[float]]],
    num_tasks: int,
    label: str,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Coerces ragged rows without first creating an object array."""
    if not isinstance(num_tasks, Integral) or isinstance(num_tasks, bool) or num_tasks < 0:
        raise ValueError('num_tasks must be a non-negative integer.')
    try:
        rows = list(values)
    except TypeError as error:
        raise ValueError(f'{label} must be a row-major sequence.') from error

    task_parts = [[] for _ in range(num_tasks)]
    row_lengths = []
    for row_index, row in enumerate(rows):
        try:
            row_values = list(row)
        except TypeError as error:
            raise ValueError(
                f'{label} row {row_index} must contain one value array per task.'
            ) from error
        if len(row_values) != num_tasks:
            raise ValueError(
                f'{label} row {row_index} contains {len(row_values)} tasks; '
                f'expected {num_tasks}.'
            )
        current_lengths = []
        for task_index, task_values in enumerate(row_values):
            try:
                task_array = np.asarray(task_values, dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f'{label} row {row_index}, task {task_index} must be numeric.'
                ) from error
            if task_array.ndim == 0:
                task_array = task_array.reshape(1)
            elif task_array.ndim == 1:
                pass
            elif task_array.ndim == 2 and 1 in task_array.shape:
                task_array = task_array.reshape(-1)
            else:
                raise ValueError(
                    f'{label} row {row_index}, task {task_index} must be a '
                    f'scalar vector; got shape {task_array.shape}.'
                )
            task_parts[task_index].append(task_array)
            current_lengths.append(task_array.size)
        row_lengths.append(current_lengths)

    flattened = [
        np.concatenate(parts) if parts else np.empty(0, dtype=float)
        for parts in task_parts
    ]
    return flattened, row_lengths


def flatten_atom_bond_values(
    values: Sequence[Sequence[Sequence[float]]],
    num_tasks: int,
    label: str,
    expected_lengths: Sequence[int] = None,
    expected_row_lengths: Sequence[Sequence[int]] = None,
) -> List[np.ndarray]:
    """Flattens row-major variable-length atom/bond values by task.

    Each datapoint contains one scalar vector per atom/bond task. The result is
    task-major and one-dimensional, matching :meth:`MoleculeDataset.mask`.
    Conversion happens per cell so heterogeneous molecule and task lengths do
    not create a ragged NumPy array.
    """
    flattened, row_lengths = _coerce_atom_bond_values(values, num_tasks, label)
    if expected_lengths is not None:
        if len(expected_lengths) != num_tasks:
            raise ValueError('Expected value lengths must contain one entry per task.')
        for task_index, (task_values, expected_length) in enumerate(
            zip(flattened, expected_lengths)
        ):
            if len(task_values) != expected_length:
                raise ValueError(
                    f'{label} task {task_index} has length {len(task_values)}; '
                    f'expected {expected_length}.'
                )
    if expected_row_lengths is not None:
        expected_row_lengths = [list(lengths) for lengths in expected_row_lengths]
        if row_lengths != expected_row_lengths:
            raise ValueError(
                f'{label} atom/bond lengths do not match the reference for each '
                'datapoint and task.'
            )
    return flattened


def flatten_atom_bond_value_sets(
    row_major_values: Mapping[str, Sequence[Sequence[Sequence[float]]]],
    num_tasks: int,
    expected_lengths: Sequence[int] = None,
) -> Dict[str, List[np.ndarray]]:
    """Flattens aligned atom/bond value sets and verifies row boundaries."""
    flattened_sets = {}
    reference_label = None
    reference_row_lengths = None
    for label, values in row_major_values.items():
        flattened, row_lengths = _coerce_atom_bond_values(values, num_tasks, label)
        if reference_row_lengths is None:
            reference_label = label
            reference_row_lengths = row_lengths
        elif row_lengths != reference_row_lengths:
            raise ValueError(
                f'{label} atom/bond lengths do not match {reference_label} for '
                'each datapoint and task.'
            )
        if expected_lengths is not None:
            if len(expected_lengths) != num_tasks:
                raise ValueError(
                    'Expected value lengths must contain one entry per task.'
                )
            for task_index, (task_values, expected_length) in enumerate(
                zip(flattened, expected_lengths)
            ):
                if len(task_values) != expected_length:
                    raise ValueError(
                        f'{label} task {task_index} has length {len(task_values)}; '
                        f'expected {expected_length}.'
                    )
        flattened_sets[label] = flattened
    return flattened_sets


def reshape_values(
    values: List[List[List[float]]],
    test_data: MoleculeDataset,
    natom_targets: int,
    nbond_targets: int,
) -> List[List[List[float]]]:
    """
    Reshape the input from shape (num_tasks, number of atomic/bond properties for each task, 1)
    to shape (data_size, num_tasks, number of atomic/bond properties for this data in each task).

    :param values: List of atomic/bond properties with shape
                   (num_tasks, number of atomic/bond properties for each task, 1).
    :param test_data: A :class:`~chemprop.data.MoleculeDataset` containing valid datapoints.
    :param natom_targets: The number of atomic targets.
    :param nbond_targets: The number of bond targets.
    :return: List of atomic/bond properties with shape
             (data_size, num_tasks, number of atomic/bond properties for this data in each task).
    """
    num_atom_bond_tasks = _validate_task_counts(natom_targets, nbond_targets)
    if len(values) != num_atom_bond_tasks:
        raise ValueError(
            f"Expected {num_atom_bond_tasks} atom/bond task arrays but received "
            f"{len(values)}."
        )

    n_atoms = _single_molecule_counts(
        test_data.number_of_atoms, len(test_data), "Atom"
    )
    n_bonds = _single_molecule_counts(
        test_data.number_of_bonds, len(test_data), "Bond"
    )
    reshaped_values = np.empty([len(test_data), num_atom_bond_tasks], dtype=object)

    for i in range(natom_targets):
        atom_targets = np.asarray(values[i]).reshape(-1)
        expected = int(n_atoms.sum())
        if atom_targets.size != expected:
            raise ValueError(
                f"Atom task {i} contains {atom_targets.size} values; expected "
                f"{expected} from the molecule atom counts."
            )
        atom_targets = _split_flat_values(atom_targets, n_atoms)
        reshaped_values[:, i] = atom_targets

    for i in range(nbond_targets):
        bond_targets = np.asarray(values[i + natom_targets]).reshape(-1)
        expected = int(n_bonds.sum())
        if bond_targets.size != expected:
            raise ValueError(
                f"Bond task {i} contains {bond_targets.size} values; expected "
                f"{expected} from the molecule bond counts."
            )
        bond_targets = _split_flat_values(bond_targets, n_bonds)
        reshaped_values[:, i + natom_targets] = bond_targets

    return reshaped_values


def reshape_individual_preds(
    individual_preds: List[List[List[List[float]]]],
    test_data: MoleculeDataset,
    natom_targets: int,
    nbond_targets: int,
    num_models: int,
) -> List[List[List[List[float]]]]:
    """
    Reshape the input from shape (num_tasks, number of atomic/bond properties for each task, 1, num_models)
    to shape (data_size, num_tasks, num_models, number of atomic/bond properties for this data in each task).

    :param individual_preds: List of atomic/bond properties with shape
                             (num_tasks, number of atomic/bond properties for each task, 1, num_models).
    :param test_data: A :class:`~chemprop.data.MoleculeDataset` containing valid datapoints.
    :param natom_targets: The number of atomic targets.
    :param nbond_targets: The number of bond targets.
    :param num_models: Number of models.
    :return: List of atomic/bond properties with shape
             (data_size, num_tasks, num_models, number of atomic/bond properties for this data in each task).
    """
    num_atom_bond_tasks = _validate_task_counts(natom_targets, nbond_targets)
    if not isinstance(num_models, Integral) or isinstance(num_models, bool) or num_models <= 0:
        raise ValueError(f"num_models must be a positive integer; got {num_models!r}.")
    if len(individual_preds) != num_atom_bond_tasks:
        raise ValueError(
            f"Expected {num_atom_bond_tasks} atom/bond task arrays but received "
            f"{len(individual_preds)}."
        )

    n_atoms = _single_molecule_counts(
        test_data.number_of_atoms, len(test_data), "Atom"
    )
    n_bonds = _single_molecule_counts(
        test_data.number_of_bonds, len(test_data), "Bond"
    )
    individual_values = np.empty([len(test_data), num_atom_bond_tasks], dtype=object)

    for i in range(natom_targets):
        atom_values = np.asarray(individual_preds[i])
        expected = int(n_atoms.sum())
        if atom_values.size != expected * num_models:
            raise ValueError(
                f"Atom task {i} contains {atom_values.size} individual values; "
                f"expected {expected * num_models}."
            )
        if atom_values.ndim == 0 or atom_values.shape[-1] != num_models:
            raise ValueError(
                f"Atom task {i} must have a final model axis of length "
                f"{num_models}; got shape {atom_values.shape}."
            )
        atom_targets = atom_values.reshape(-1, num_models).T
        atom_targets = _split_flat_values(atom_targets, n_atoms, axis=1)
        individual_values[:, i] = atom_targets

    for i in range(nbond_targets):
        bond_values = np.asarray(individual_preds[i + natom_targets])
        expected = int(n_bonds.sum())
        if bond_values.size != expected * num_models:
            raise ValueError(
                f"Bond task {i} contains {bond_values.size} individual values; "
                f"expected {expected * num_models}."
            )
        if bond_values.ndim == 0 or bond_values.shape[-1] != num_models:
            raise ValueError(
                f"Bond task {i} must have a final model axis of length "
                f"{num_models}; got shape {bond_values.shape}."
            )
        bond_targets = bond_values.reshape(-1, num_models).T
        bond_targets = _split_flat_values(bond_targets, n_bonds, axis=1)
        individual_values[:, i + natom_targets] = bond_targets

    return individual_values
