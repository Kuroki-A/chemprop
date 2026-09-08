import numpy as np
import pytest
import torch

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataset, scaffold_split
from chemprop.features import get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.models.ffn import FFNAtten, MultiReadout, build_ffn
from chemprop.models.mpn import MPN, MPNEncoder
from chemprop.multitask_utils import (
    flatten_atom_bond_value_sets,
    flatten_atom_bond_values,
    reshape_individual_preds,
    reshape_values,
    validate_task_masks,
)
from chemprop.train.evaluate import evaluate_predictions


class _CountDataset:
    number_of_atoms = [[2], [1]]
    number_of_bonds = [[1], [0]]

    def __len__(self):
        return 2


def test_flatten_atom_bond_values_preserves_ragged_task_order_and_empty_cells():
    values = [
        [np.array([1.0, 2.0]), np.array([10.0])],
        [np.array([3.0]), np.array([], dtype=float)],
    ]

    flattened = flatten_atom_bond_values(
        values,
        num_tasks=2,
        label='predictions',
        expected_lengths=[3, 1],
    )

    np.testing.assert_array_equal(flattened[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(flattened[1], [10.0])
    masks = validate_task_masks(
        [[True, False, True], [True]], num_tasks=2, expected_lengths=[3, 1]
    )
    assert [mask.tolist() for mask in masks] == [[True, False, True], [True]]


def test_flatten_atom_bond_value_sets_rejects_changed_molecule_boundaries():
    targets = [[np.array([1.0, 2.0])], [np.array([3.0])]]
    swapped_boundaries = [[np.array([1.0])], [np.array([2.0, 3.0])]]

    with pytest.raises(ValueError, match='each datapoint and task'):
        flatten_atom_bond_value_sets(
            {'targets': targets, 'predictions': swapped_boundaries},
            num_tasks=1,
            expected_lengths=[3],
        )


def test_reshape_values_preserves_atom_and_bond_row_boundaries():
    result = reshape_values(
        [np.array([[1.0], [2.0], [3.0]]), np.array([[4.0]])],
        _CountDataset(),
        natom_targets=1,
        nbond_targets=1,
    )

    assert result.shape == (2, 2)
    assert result[0, 0].tolist() == [1.0, 2.0]
    assert result[1, 0].tolist() == [3.0]
    assert result[0, 1].tolist() == [4.0]
    assert result[1, 1].tolist() == []


def test_reshape_values_rejects_silent_task_and_length_truncation():
    with pytest.raises(ValueError, match="task arrays"):
        reshape_values([], _CountDataset(), natom_targets=1, nbond_targets=0)
    with pytest.raises(ValueError, match="contains 2 values; expected 3"):
        reshape_values(
            [np.array([1.0, 2.0])],
            _CountDataset(),
            natom_targets=1,
            nbond_targets=0,
        )


def test_reshape_individual_preds_keeps_model_axis():
    values = np.array(
        [
            [[1.0, 10.0]],
            [[2.0, 20.0]],
            [[3.0, 30.0]],
        ]
    )
    result = reshape_individual_preds(
        [values], _CountDataset(), natom_targets=1, nbond_targets=0, num_models=2
    )

    assert result[0, 0].shape == (2, 2)
    assert result[0, 0].tolist() == [[1.0, 2.0], [10.0, 20.0]]
    assert result[1, 0].tolist() == [[3.0], [30.0]]


def test_evaluate_predictions_rejects_misaligned_or_nonfinite_rows():
    with pytest.raises(ValueError, match="row count"):
        evaluate_predictions(
            preds=[[1.0]],
            targets=[[1.0], [2.0]],
            num_tasks=1,
            metrics=["rmse"],
            dataset_type="regression",
        )
    with pytest.raises(ValueError, match="NaN/infinity"):
        evaluate_predictions(
            preds=[[np.nan]],
            targets=[[1.0]],
            num_tasks=1,
            metrics=["rmse"],
            dataset_type="regression",
        )


def test_spectra_evaluation_allows_nan_only_at_masked_target_bins():
    result = evaluate_predictions(
        preds=[[0.25, np.nan, 0.75]],
        targets=[[0.25, None, 0.75]],
        num_tasks=3,
        metrics=["sid"],
        dataset_type="spectra",
    )

    assert result["sid"] == pytest.approx([0.0])


def test_atom_bond_bounded_masks_are_transposed_with_targets():
    result = evaluate_predictions(
        preds=[np.array([[2.0], [5.0]])],
        targets=[[np.array([1.0])], [np.array([4.0])]],
        num_tasks=1,
        metrics=["bounded_mae"],
        dataset_type="regression",
        is_atom_bond_targets=True,
        gt_targets=[[np.array([True])], [np.array([False])]],
        lt_targets=[[np.array([False])], [np.array([True])]],
    )

    assert len(result["bounded_mae"]) == 1
    assert np.isfinite(result["bounded_mae"][0])


def test_scaffold_split_rejects_negative_sizes_and_bad_key():
    with pytest.raises(ValueError, match="Invalid train/val/test"):
        scaffold_split(MoleculeDataset([]), sizes=(1.1, -0.1, 0.0))
    with pytest.raises(ValueError, match="key_molecule_index"):
        scaffold_split(MoleculeDataset([]), key_molecule_index=-1)


def _train_args():
    return TrainArgs().parse_args(
        [
            "--data_path",
            "tests/data/regression.csv",
            "--dataset_type",
            "regression",
            "--no_cuda",
        ]
    )


def test_mpn_encoder_explicit_false_bias_overrides_primary_bias():
    args = _train_args()
    args.bias = True

    encoder = MPNEncoder(args, atom_fdim=5, bond_fdim=6, bias=False)

    assert encoder.bias is False
    assert encoder.W_i.bias is None


def test_mpn_encoder_places_bond_descriptors_without_numpy_tensor_indexing():
    args = _train_args()
    args.is_atom_bond_targets = True
    args.bond_descriptors = "descriptor"
    args.bond_descriptors_size = 2
    encoder = MPNEncoder(
        args,
        atom_fdim=get_atom_fdim(),
        bond_fdim=get_bond_fdim(),
    )
    graph = mol2graph(["CC", "C"])

    _, _, bond_hiddens, _, mapping = encoder(
        graph,
        bond_descriptors_batch=[
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        ],
    )

    assert bond_hiddens.shape[0] == 3  # padding plus two directed edges
    assert mapping.shape == (1, 2)
    assert torch.isfinite(bond_hiddens).all()


def test_mpn_rejects_ragged_molecule_counts_before_graph_creation():
    mpn = MPN(_train_args())

    with pytest.raises(ValueError, match="exactly 1 molecule"):
        mpn([["CC"], ["CC", "O"]])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"num_layers": 0}, "num_layers"),
        ({"dropout": float("nan")}, "dropout"),
        ({"first_linear_dim": 0}, "first_linear_dim"),
    ],
)
def test_build_ffn_validates_direct_call_parameters(kwargs, match):
    params = dict(
        first_linear_dim=2,
        hidden_size=2,
        num_layers=1,
        output_size=1,
        dropout=0.0,
        activation="ReLU",
    )
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        build_ffn(**params)


def test_constrained_readout_supports_an_all_bondless_batch():
    readout = FFNAtten(
        features_size=2,
        hidden_size=2,
        num_layers=1,
        output_size=1,
        dropout=0.0,
        activation="ReLU",
        ffn_type="bond",
    )

    result = readout.readout(
        torch.empty((0, 2)),
        scope=[(0, 0), (0, 0)],
        constraints=torch.tensor([0.0, 0.0]),
        bond_types=None,
    )

    assert result.shape == (0, 1)

    with pytest.raises(ValueError, match="cannot satisfy a non-zero constraint"):
        readout.readout(
            torch.empty((0, 2)),
            scope=[(0, 0)],
            constraints=torch.tensor([1.0]),
            bond_types=None,
        )


def test_constrained_atom_readout_preserves_each_molecule_total():
    readout = FFNAtten(
        features_size=2,
        hidden_size=2,
        num_layers=1,
        output_size=1,
        dropout=0.0,
        activation="ReLU",
        ffn_type="atom",
    )

    result = readout.readout(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        scope=[(1, 2), (3, 1)],
        constraints=torch.tensor([5.0, 7.0]),
        bond_types=None,
    )

    assert result[:2].sum().item() == pytest.approx(5.0)
    assert result[2:].sum().item() == pytest.approx(7.0)


def test_multi_readout_rejects_missing_task_side_inputs():
    readout = MultiReadout(
        atom_features_size=2,
        bond_features_size=2,
        atom_hidden_size=2,
        bond_hidden_size=2,
        num_layers=1,
        output_size=1,
        dropout=0.0,
        activation="ReLU",
        atom_constraints=[False],
        bond_constraints=[],
    )

    with pytest.raises(ValueError, match="constraints"):
        readout(None, [], [None])
