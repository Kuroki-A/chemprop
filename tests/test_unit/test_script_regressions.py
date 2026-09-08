import csv
import pickle
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from chemprop.data import get_data_from_smiles
from scripts.aggregate_results import aggregate_results
from scripts.avg_dups import average_duplicates
from scripts.create_crossval_index_files import create_crossval_indices
from scripts.create_crossval_splits import create_time_splits, split_indices
from scripts.find_similar_mols import find_similar_mols
from scripts.lsc_to_our_format import lsc_to_our_format
from scripts.overlap import overlap
from scripts.sanitize import sanitize
from scripts.tsne import Args as TsneArgs
from scripts.welchs import welchs
from scripts.wilcoxon_significance import compute_values


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.reader(file))


def test_overlap_writes_matching_multi_smiles_rows(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    intersection = tmp_path / "intersection.csv"
    difference = tmp_path / "difference.csv"
    _write_csv(
        first,
        [["left", "value", "right"], ["CC", "1", "O"], ["N", "2", "Cl"]],
    )
    _write_csv(
        second,
        [["a", "b"], ["CC", "O"], ["F", "Br"]],
    )

    overlap(
        SimpleNamespace(
            data_path_1=str(first),
            data_path_2=str(second),
            smiles_columns_1=["left", "right"],
            smiles_columns_2=["a", "b"],
            save_intersection_path=str(intersection),
            save_difference_path=str(difference),
        )
    )

    assert _read_csv(intersection) == [
        ["left", "value", "right"],
        ["CC", "1", "O"],
    ]
    assert _read_csv(difference) == [
        ["left", "value", "right"],
        ["N", "2", "Cl"],
    ]


def test_scaffold_fold_indices_remain_in_caller_index_space():
    data = get_data_from_smiles(
        smiles=[["CC"], ["CCC"], ["c1ccccc1"], ["c1ccncc1"]]
    )

    folds = split_indices(
        [10, 11, 12, 13],
        num_folds=2,
        scaffold=True,
        data=data,
        shuffle=False,
    )

    assert {index for fold in folds for index in fold} == {10, 11, 12, 13}


def test_time_split_writes_parent_schema_for_every_window(tmp_path):
    source = tmp_path / "data.csv"
    _write_csv(
        source,
        [["smiles", "y"]]
        + [
            [smiles, str(index)]
            for index, smiles in enumerate(
                [
                    "c1ccccc1",
                    "c1ccncc1",
                    "C1CCCCC1",
                    "C1CCNCC1",
                    "C1CCCC1",
                    "c1ccc2ccccc2c1",
                    "c1ccc2[nH]ccc2c1",
                    "C1CCOC1",
                ]
            )
        ],
    )
    output = tmp_path / "splits"

    create_time_splits(
        SimpleNamespace(
            data_path=str(source),
            smiles_columns=None,
            time_folds_per_train_set=1,
            num_folds=4,
            seed=0,
            split_key_molecule=0,
            save_dir=str(output),
        )
    )

    child_paths = sorted(output.glob("scaffold/fold_*/0/split_indices.pckl"))
    assert len(child_paths) == 2
    for child_path in child_paths:
        with child_path.open("rb") as file:
            child_split = pickle.load(file)
        with (child_path.parent.parent / "split_indices.pckl").open("rb") as file:
            parent_splits = pickle.load(file)
        assert len(parent_splits) == 1
        assert all(
            np.array_equal(parent, child)
            for parent, child in zip(parent_splits[0], child_split)
        )


def test_create_crossval_indices_creates_mayr_directory(tmp_path):
    create_crossval_indices(
        SimpleNamespace(
            save_dir=str(tmp_path),
            num_folds=3,
            test_folds_to_test=1,
            val_folds_per_test=1,
            seed=7,
        )
    )

    assert len(list((tmp_path / "mayr").glob("*_opt.pkl"))) == 1
    assert len(list((tmp_path / "mayr").glob("*_test.pkl"))) == 1


def test_find_similar_mols_does_not_depend_on_cli_global_args():
    neighbors = find_similar_mols(
        test_smiles=["CC"],
        train_smiles=["CCC", "c1ccccc1"],
        distance_measure="morgan",
        num_neighbors=1,
        batch_size=1,
        num_workers=0,
    )

    assert len(neighbors) == 1
    assert neighbors[0]["test_smiles"] == "CC"
    assert "train_1_smiles" in neighbors[0]


def test_compute_values_extracts_metric_values_from_result_dictionary():
    values = compute_values(
        "delaney",
        preds=[[[1.0], [3.0]]],
        targets=[[[1.0], [1.0]]],
    )

    assert values == pytest.approx([np.sqrt(2.0)])


def test_welch_one_sided_p_value_respects_effect_direction(capsys):
    welchs(
        mean1=[0.0],
        std1=[1.0],
        nobs1=[10],
        mean2=[5.0],
        std2=[1.0],
        nobs2=[10],
        alternative="greater",
    )

    first_pvalue = float(capsys.readouterr().out.splitlines()[0])
    assert first_pvalue > 0.5


def test_empty_aggregate_log_is_reported_without_unbound_local(tmp_path, capsys):
    log_dir = tmp_path / "unknown" / "random" / "0"
    log_dir.mkdir(parents=True)
    (log_dir / "verbose.log").touch()

    aggregate_results([str(tmp_path / "unknown")], "random")

    assert "unknown\tN/A\tN/A\t0" in capsys.readouterr().out


def test_average_duplicates_outputs_only_selected_schema(tmp_path):
    source = tmp_path / "data.csv"
    output = tmp_path / "averaged.csv"
    _write_csv(
        source,
        [
            ["mol", "ignored", "target"],
            ["CC", "x", "1"],
            ["CC", "y", "3"],
        ],
    )

    average_duplicates(
        SimpleNamespace(
            data_path=str(source),
            smiles_columns=["mol"],
            target_columns=["target"],
            save_path=str(output),
        )
    )

    assert _read_csv(output) == [["mol", "target"], ["CC", "2.0"]]


def test_sanitize_honors_named_smiles_column(tmp_path):
    source = tmp_path / "data.csv"
    output = tmp_path / "clean.csv"
    _write_csv(
        source,
        [["id", "mol"], ["1", "CC"], ["2", "not-a-smiles"]],
    )

    sanitize(str(source), str(output), ["mol"])

    assert _read_csv(output) == [["id", "mol"], ["1", "CC"]]


def test_lsc_conversion_rejects_prediction_target_shape_mismatch(tmp_path):
    lsc_dir = tmp_path / "lsc"
    ckpt_dir = tmp_path / "ckpt"
    h5_path = lsc_dir / "qm7" / "test" / "fold_0" / "semi" / "o0003.evalPredict.hdf5"
    target_path = ckpt_dir / "qm7" / "scaffold" / "0" / "targets.npy"
    h5_path.parent.mkdir(parents=True)
    target_path.parent.mkdir(parents=True)
    with h5py.File(h5_path, "w") as file:
        file.create_dataset("predictions", data=np.zeros((2, 1)))
    np.save(target_path, np.zeros((3, 1)))

    with pytest.raises(ValueError, match="does not match"):
        lsc_to_our_format(str(lsc_dir), str(ckpt_dir), str(tmp_path / "out"))


def test_tsne_args_do_not_share_mutable_plot_lists():
    first = TsneArgs()
    second = TsneArgs()
    first.process_args()
    second.process_args()

    first.colors.append("black")
    first.sizes.append(2)

    assert "black" not in second.colors
    assert len(second.sizes) == 5
