"""Regression tests for molecule-level feature generators."""

import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem

from chemprop.features import load_features
from chemprop.features import features_generators as generators
from scripts import save_features as save_features_script


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def _bit_vector_array(fingerprint, dtype=np.int64):
    output = np.empty(fingerprint.GetNumBits(), dtype=dtype)
    DataStructs.ConvertToNumpyArray(fingerprint, output)
    return output


class TestRDKitFeatureGenerators(unittest.TestCase):
    def setUp(self):
        generators.clear_features_generator_caches()
        self.mol = Chem.MolFromSmiles(ASPIRIN)

    def test_fingerprints_match_legacy_rdkit_apis(self):
        references = {
            "morgan": _bit_vector_array(
                AllChem.GetMorganFingerprintAsBitVect(self.mol, 2, nBits=2048), float,
            ),
            "rdkit": _bit_vector_array(Chem.RDKFingerprint(self.mol, fpSize=2048)),
            "avalon": _bit_vector_array(pyAvalonTools.GetAvalonFP(self.mol, nBits=512)),
            "atompair": _bit_vector_array(
                AllChem.GetHashedAtomPairFingerprintAsBitVect(self.mol, nBits=2048),
            ),
        }
        for name, expected in references.items():
            with self.subTest(generator=name):
                actual = generators.get_features_generator(name)(self.mol)
                np.testing.assert_array_equal(actual, expected)

        count_reference = np.zeros(2048, dtype=float)
        count_fingerprint = AllChem.GetHashedMorganFingerprint(self.mol, 2, nBits=2048)
        for index, value in count_fingerprint.GetNonzeroElements().items():
            count_reference[index] = value
        np.testing.assert_array_equal(
            generators.morgan_counts_features_generator(self.mol), count_reference,
        )

    def test_legacy_shapes_dtypes_and_golden_sums(self):
        expected = {
            "morgan": ((2048,), np.dtype("float64"), 24.0),
            "morgan_count": ((2048,), np.dtype("float64"), 35.0),
            "maccs": ((167,), np.dtype("int64"), 21.0),
            "rdkit": ((2048,), np.dtype("int64"), 354.0),
            "avalon": ((512,), np.dtype("int64"), 54.0),
            "atompair": ((2048,), np.dtype("int64"), 68.0),
            "erg": ((315,), np.dtype("int64"), 15.0),
            "erg_float": ((315,), np.dtype("float64"), 23.4),
        }
        for name, (shape, dtype, total) in expected.items():
            with self.subTest(generator=name):
                output = generators.get_features_generator(name)(self.mol)
                self.assertEqual(output.shape, shape)
                self.assertEqual(output.dtype, dtype)
                self.assertAlmostEqual(float(output.sum()), total)

        self.assertFalse(np.array_equal(
            generators.erg_legacy_features_generator(self.mol),
            generators.erg_float_features_generator(self.mol),
        ))

    def test_native_rdkit_fingerprint_batches_match_scalar(self):
        molecules = [
            ASPIRIN,
            "CCO",
            Chem.AddHs(Chem.MolFromSmiles("C")),
            Chem.MolFromSmiles("[CH3:7][OH:9]"),
        ]
        cases = {
            "morgan": "bit",
            "morgan_count": "count",
            "rdkit": "bit",
            "atompair": "bit",
        }
        for name, prefix in cases.items():
            with self.subTest(generator=name):
                generator = generators.get_features_generator(name)
                scalar = np.stack([generator(molecule) for molecule in molecules])
                batch = np.stack(generators.generate_features_batch(
                    name, molecules, batch_size=2,
                ))
                np.testing.assert_array_equal(batch, scalar)
                self.assertEqual(batch.dtype, scalar.dtype)

                selected = [f"{prefix}_17", f"{prefix}_3", f"{prefix}_17"]
                scalar_selected = np.stack([
                    generator(
                        molecule, selected_feature_columns=selected,
                    )
                    for molecule in molecules
                ])
                batch_selected = np.stack(generators.generate_features_batch(
                    name, molecules, selected, batch_size=3,
                ))
                np.testing.assert_array_equal(batch_selected, scalar_selected)
                self.assertEqual(batch_selected.dtype, scalar_selected.dtype)

                empty = generator.batch_transform(
                    [], selected_feature_columns=selected,
                )
                self.assertEqual(empty.shape, (0, len(selected)))
                self.assertEqual(empty.dtype, scalar.dtype)

    def test_native_rdkit_fingerprint_batches_preserve_dataset_policy(self):
        from chemprop.data.data import generate_features_for_smiles_batch

        smiles_rows = [
            ["[CH3:1][OH:2]>>[CH3:1][Cl:2]"],
            ["[H][H]"],
            ["OCC"],
        ]
        cases = {
            "morgan": "bit",
            "morgan_count": "count",
            "rdkit": "bit",
            "atompair": "bit",
        }
        for name, prefix in cases.items():
            with self.subTest(generator=name):
                generator = generators.get_features_generator(name)
                selected = [f"{prefix}_17", f"{prefix}_3", f"{prefix}_17"]

                def scalar_generator(molecule, selected_feature_columns=None):
                    return generator(
                        molecule,
                        selected_feature_columns=selected_feature_columns,
                    )

                expected = generate_features_for_smiles_batch(
                    smiles_rows,
                    [name],
                    selected_feature_columns={name: selected},
                    generator_overrides={name: scalar_generator},
                    auto_detect_reactions=True,
                )
                actual = generate_features_for_smiles_batch(
                    smiles_rows,
                    [name],
                    selected_feature_columns={name: selected},
                    auto_detect_reactions=True,
                )
                for expected_row, actual_row in zip(expected, actual):
                    np.testing.assert_array_equal(actual_row, expected_row)
                np.testing.assert_array_equal(actual[1], np.zeros(len(selected)))

    def test_rdkit_batch_thread_count_is_affinity_aware_and_bounded(self):
        with mock.patch.object(
            generators.os, "sched_getaffinity", return_value=set(range(64)),
        ):
            self.assertEqual(generators._rdkit_fingerprint_batch_num_threads(256), 4)
            self.assertEqual(generators._rdkit_fingerprint_batch_num_threads(3), 3)
            self.assertEqual(generators._rdkit_fingerprint_batch_num_threads(1), 1)

        with mock.patch.object(
            generators.os, "sched_getaffinity", side_effect=OSError,
        ), mock.patch.object(generators.os, "cpu_count", return_value=2):
            self.assertEqual(generators._rdkit_fingerprint_batch_num_threads(256), 2)

    def test_fixed_fingerprints_honor_selected_columns(self):
        cases = {
            "morgan": ("bit", [17, 3, 17]),
            "morgan_count": ("count", [17, 3, 17]),
            "maccs": ("bit", [17, 3, 17]),
            "rdkit": ("bit", [17, 3, 17]),
            "avalon": ("bit", [17, 3, 17]),
            "atompair": ("bit", [17, 3, 17]),
            "erg": ("erg", [17, 3, 17]),
            "erg_float": ("erg", [17, 3, 17]),
        }
        for name, (prefix, indices) in cases.items():
            with self.subTest(generator=name):
                generator = generators.get_features_generator(name)
                full = generator(self.mol)
                selected = [f"{prefix}_{index}" for index in indices]
                subset = generator(
                    self.mol, selected_feature_columns=selected,
                )
                np.testing.assert_array_equal(subset, full[indices])
                schema = generators.get_features_generator_schema(
                    name, subset, selected_feature_columns=selected,
                )
                self.assertEqual(schema["dimension"], len(selected))
                self.assertEqual(schema["feature_names"], selected)

                with self.assertRaisesRegex(KeyError, "not_a_feature"):
                    generator(
                        self.mol,
                        selected_feature_columns=["not_a_feature"],
                    )

    def test_checkpoint_metadata_preserves_generator_and_selected_column_order(self):
        first = generators.get_features_generators_metadata(
            ["rdkit_2d_208"],
            {"rdkit_2d_208": ["MolWt", "TPSA"]},
            total_dimension=2,
        )
        reordered = generators.get_features_generators_metadata(
            ["rdkit_2d_208"],
            {"rdkit_2d_208": ["TPSA", "MolWt"]},
            total_dimension=2,
        )
        self.assertNotEqual(first, reordered)
        self.assertEqual(first["generators"][0]["name"], "rdkit_2d_208")
        self.assertEqual(
            first["generators"][0]["config"]["selected_feature_columns"],
            ["MolWt", "TPSA"],
        )
        self.assertIn("rdkit", first["versions"])

    def test_checkpoint_metadata_supports_callable_instances_and_builtins(self):
        class CallableGenerator:
            def __call__(self, mol, selected_feature_columns=None):
                return np.asarray([1.0])

        registry = generators.FEATURES_GENERATOR_REGISTRY
        try:
            registry["_callable_instance"] = CallableGenerator()
            registry["_builtin_callable"] = len
            first = generators.get_features_generators_metadata(
                ["_callable_instance"], total_dimension=1,
            )
            second = generators.get_features_generators_metadata(
                ["_builtin_callable"], total_dimension=1,
            )
            self.assertEqual(len(first["generators"][0]["implementation_sha256"]), 64)
            self.assertEqual(len(second["generators"][0]["implementation_sha256"]), 64)
            self.assertNotIn("semantic_revision", first["generators"][0])
            self.assertNotIn("semantic_revision", second["generators"][0])
        finally:
            registry.pop("_callable_instance", None)
            registry.pop("_builtin_callable", None)

    def test_builtin_metadata_uses_only_targeted_semantic_revisions(self):
        with mock.patch.object(
            generators,
            "_features_generator_implementation_sha256",
            side_effect=AssertionError("built-ins must not use source hashing"),
        ):
            first_morgan = generators.get_features_generators_metadata(["morgan"])
            first_maccs = generators.get_features_generators_metadata(["maccs"])

        self.assertEqual(
            first_morgan["schema_version"],
            generators.FEATURE_GENERATOR_METADATA_SCHEMA_VERSION,
        )
        self.assertEqual(first_morgan["generators"][0]["semantic_revision"], 1)
        self.assertNotIn(
            "implementation_sha256", first_morgan["generators"][0],
        )

        with mock.patch.dict(
            generators._BUILTIN_FEATURES_GENERATOR_SEMANTIC_REVISIONS,
            {"morgan": 2},
        ):
            changed_morgan = generators.get_features_generators_metadata(["morgan"])
            unchanged_maccs = generators.get_features_generators_metadata(["maccs"])

        self.assertNotEqual(first_morgan, changed_morgan)
        self.assertEqual(first_maccs, unchanged_maccs)

    def test_custom_replacement_of_builtin_name_uses_source_hash(self):
        registry = generators.FEATURES_GENERATOR_REGISTRY
        original = registry["morgan"]
        self.assertTrue(generators.is_builtin_features_generator("morgan"))
        self.assertTrue(
            generators.is_builtin_features_generator("morgan", original)
        )
        self.assertFalse(generators.is_builtin_features_generator("not_registered"))

        def plugin_replacement(mol, selected_feature_columns=None):
            return np.asarray([1.0])

        try:
            generators.register_features_generator("morgan")(plugin_replacement)
            self.assertFalse(generators.is_builtin_features_generator("morgan"))
            self.assertFalse(
                generators.is_builtin_features_generator(
                    "morgan", plugin_replacement,
                )
            )
            entry = generators.get_features_generators_metadata(
                ["morgan"], total_dimension=1,
            )["generators"][0]
            self.assertEqual(len(entry["implementation_sha256"]), 64)
            self.assertNotIn("semantic_revision", entry)
        finally:
            registry["morgan"] = original

    @unittest.skipUnless(importlib.util.find_spec("descriptastorus"), "descriptastorus is optional")
    def test_descriptor_shapes_and_selected_order(self):
        expected_widths = {
            "rdkit_2d": 200,
            "rdkit_2d_normalized": 200,
            "rdkit_2d_wo_fr": 115,
            "rdkit_2d_normalized_wo_fr": 115,
            "rdkit_2d_208": 208,
            "rdkit_2d_400": 400,
            "rdkit_2d_autocorr": 192,
            "rdkit_2d_bcut": 8,
        }
        for name, width in expected_widths.items():
            with self.subTest(generator=name):
                output = generators.get_features_generator(name)(self.mol)
                self.assertEqual(output.shape, (width,))
                self.assertEqual(output.dtype, np.dtype("float64"))

        selected = ["TPSA", "MolWt", "TPSA"]
        output = generators.rdkit_2d_208_features_generator(self.mol, selected)
        np.testing.assert_allclose(output, [63.6, 180.159, 63.6], rtol=0, atol=1e-9)
        schema = generators.get_features_generator_schema(
            "rdkit_2d_208", output, selected_feature_columns=selected,
        )
        self.assertEqual(schema["feature_names"], selected)

        descriptor_columns = ["MolWt", "TPSA"]
        smiles = [ASPIRIN, "CCO", ASPIRIN]
        scalar = np.stack([
            generators.rdkit_2d_features_generator(value, descriptor_columns)
            for value in smiles
        ])
        batch = np.stack(generators.generate_features_batch(
            "rdkit_2d", smiles, descriptor_columns, batch_size=1,
        ))
        np.testing.assert_array_equal(batch, scalar)

        normalized_scalar = np.stack([
            generators.rdkit_2d_normalized_features_generator(
                value, descriptor_columns,
            )
            for value in smiles
        ])
        normalized_batch = np.stack(generators.generate_features_batch(
            "rdkit_2d_normalized", smiles, descriptor_columns,
        ))
        np.testing.assert_array_equal(normalized_batch, normalized_scalar)

    @unittest.skipUnless(importlib.util.find_spec("descriptastorus"), "descriptastorus is optional")
    def test_normalized_descriptor_batch_matches_scalar_for_explicit_hydrogens(self):
        explicit_h_mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        for name in (
            "rdkit_2d_normalized",
            "rdkit_2d_normalized_wo_fr",
        ):
            with self.subTest(generator=name):
                scalar = generators.get_features_generator(name)(explicit_h_mol)
                batch = generators.generate_features_batch(name, [explicit_h_mol])[0]
                np.testing.assert_array_equal(batch, scalar)

    def test_every_registered_function_is_pickleable(self):
        for name, generator in generators.FEATURES_GENERATOR_REGISTRY.items():
            with self.subTest(generator=name):
                pickle.dumps(generator)


class TestMap4FeatureGenerators(unittest.TestCase):
    """Locks down the intentionally different MAP4 v1.0 and v1.1 formats."""

    _legacy_dependencies_available = importlib.util.find_spec("mhfp") is not None
    _native_dependencies_available = (
        _legacy_dependencies_available and importlib.util.find_spec("map4") is not None
    )

    @classmethod
    def _available_names(cls):
        names = ["map4"]
        if cls._native_dependencies_available:
            names.append("map4_v1_1")
        return names

    def setUp(self):
        generators.clear_features_generator_caches()

    def tearDown(self):
        generators.clear_features_generator_caches()

    def test_both_map4_formats_are_registered(self):
        self.assertIn("map4", generators.FEATURES_GENERATOR_REGISTRY)
        self.assertIn("map4_v1_1", generators.FEATURES_GENERATOR_REGISTRY)

    @unittest.skipUnless(
        _legacy_dependencies_available, "The legacy MAP4 dependency is optional",
    )
    def test_legacy_shape_dtype_and_golden_vector(self):
        legacy = generators.get_features_generator("map4")("CCO")
        self.assertEqual(legacy.shape, (2048,))
        self.assertEqual(legacy.dtype, np.dtype("float64"))
        self.assertTrue(np.isfinite(legacy).all())

        # This compact golden vector guards the v1.0 SHA-1/modulo fold and
        # lexicographically sorted atom-environment shingles without requiring
        # the unmaintained v1.0 package.
        np.testing.assert_array_equal(
            np.flatnonzero(legacy),
            [333, 425, 515, 1625, 1827, 1922],
        )
        np.testing.assert_array_equal(legacy[np.flatnonzero(legacy)], np.ones(6))

    @unittest.skipUnless(
        _native_dependencies_available, "The native MAP4 dependencies are optional",
    )
    def test_native_shape_and_dtype(self):
        native = generators.get_features_generator("map4_v1_1")("CCO")
        self.assertEqual(native.shape, (2048,))
        self.assertEqual(native.dtype, np.dtype("float64"))
        self.assertTrue(np.isfinite(native).all())

    @unittest.skipUnless(
        _legacy_dependencies_available, "The legacy MAP4 dependency is optional",
    )
    def test_scalar_and_batch_results_are_identical(self):
        smiles = ["CCO", "c1ccccc1", ASPIRIN, "CCO"]
        for name in self._available_names():
            with self.subTest(generator=name):
                generator = generators.get_features_generator(name)
                scalar = np.stack([generator(value) for value in smiles])
                batch = np.stack(
                    generators.generate_features_batch(name, smiles, batch_size=2)
                )
                np.testing.assert_array_equal(batch, scalar)

    @unittest.skipUnless(
        _legacy_dependencies_available, "The legacy MAP4 dependency is optional",
    )
    def test_equivalent_smiles_and_atom_renumbering_are_invariant(self):
        equivalent_aspirin = "O=C(O)c1ccccc1OC(C)=O"
        mol = Chem.MolFromSmiles(ASPIRIN)
        renumbered = Chem.RenumberAtoms(
            mol, list(reversed(range(mol.GetNumAtoms()))),
        )
        self.assertEqual(
            Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN), canonical=True),
            Chem.MolToSmiles(Chem.MolFromSmiles(equivalent_aspirin), canonical=True),
        )

        for name in self._available_names():
            with self.subTest(generator=name):
                generator = generators.get_features_generator(name)
                expected = generator(ASPIRIN)
                np.testing.assert_array_equal(expected, generator(equivalent_aspirin))
                np.testing.assert_array_equal(expected, generator(renumbered))

    @unittest.skipUnless(
        _native_dependencies_available, "The native MAP4 dependencies are optional",
    )
    def test_formats_are_distinct_and_v1_1_matches_native_map4(self):
        from map4 import MAP4

        source = Chem.MolFromSmiles(ASPIRIN)
        canonical_smiles = Chem.MolToSmiles(
            source, canonical=True, isomericSmiles=False,
        )
        canonical_mol = Chem.MolFromSmiles(canonical_smiles)
        expected_native = np.asarray(
            MAP4(
                dimensions=2048,
                radius=2,
                include_duplicated_shingles=False,
            ).calculate(canonical_mol),
            dtype=float,
        )

        legacy = generators.get_features_generator("map4")(source)
        native = generators.get_features_generator("map4_v1_1")(source)
        self.assertFalse(np.array_equal(legacy, native))
        np.testing.assert_array_equal(native, expected_native)

    @unittest.skipUnless(
        _legacy_dependencies_available, "The legacy MAP4 dependency is optional",
    )
    def test_selected_columns_schema_and_provenance(self):
        selected = ["fp_17", "fp_3", "fp_17"]
        cases = [("map4", "MAP4-v1.0-folded", {"rdkit", "mhfp"})]
        if self._native_dependencies_available:
            cases.append(
                ("map4_v1_1", "MAP4-v1.1-folded", {"rdkit", "map4", "mhfp"}),
            )
        for name, algorithm, dependencies in cases:
            with self.subTest(generator=name):
                full = generators.get_features_generator(name)(ASPIRIN)
                subset = generators.get_features_generator(name)(ASPIRIN, selected)
                np.testing.assert_array_equal(subset, full[[17, 3, 17]])

                schema = generators.get_features_generator_schema(
                    name, subset, selected_feature_columns=selected,
                )
                self.assertEqual(schema["dimension"], 3)
                self.assertEqual(schema["feature_names"], selected)
                self.assertEqual(
                    schema["generator_config"]["algorithm"], algorithm,
                )
                self.assertEqual(
                    schema["generator_config"]["fragment_policy"], "retain-all",
                )
                self.assertEqual(
                    set(schema["versions"]), dependencies,
                )

    @unittest.skipUnless(
        _native_dependencies_available and importlib.util.find_spec("molfeat") is not None,
        "MAP4 and Molfeat feature dependencies are optional",
    )
    def test_fresh_process_dependency_boundary_and_pretrained_import_patch(self):
        code = """
import sys
from chemprop.features.features_generators import (
    _load_pretrained_transformer_class,
    map4_features_generator,
)
map4_features_generator('CCO')
assert 'map4' not in sys.modules
try:
    _load_pretrained_transformer_class('hf')
except ImportError:
    pass
import map4
assert hasattr(map4, 'MAP4Calculator')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.getcwd(),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


class TestBatchCacheAndSchema(unittest.TestCase):
    def test_batch_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            generators.generate_features_batch("morgan", ["CC"], batch_size=0)

    def tearDown(self):
        generators.clear_features_generator_caches()

    def test_pretrained_transformer_is_cached_and_batchable(self):
        constructions = []
        calls = []

        class FakeTransformer:
            def __init__(self, **kwargs):
                constructions.append(kwargs)

            def __call__(self, values):
                if isinstance(values, str):
                    values = [values]
                calls.append(list(values))
                return np.asarray([[len(value), len(value) + 1] for value in values], dtype=float)

        with mock.patch.object(
            generators, "_load_pretrained_transformer_class", return_value=FakeTransformer,
        ):
            generator = generators.get_features_generator("MolT5")
            np.testing.assert_array_equal(generator("CC"), [2.0, 3.0])
            np.testing.assert_array_equal(generator("CCC"), [3.0, 4.0])
            batch = generators.generate_features_batch("MolT5", ["CC", "CCC"], batch_size=1)
            np.testing.assert_array_equal(batch, [[2.0, 3.0], [3.0, 4.0]])
            self.assertEqual(len(constructions), 1)
            self.assertEqual(calls[-2:], [["CC"], ["CCC"]])

            selected = ["embedding_1", "embedding_0", "embedding_1"]
            np.testing.assert_array_equal(generator("CC", selected), [3.0, 2.0, 3.0])
            selected_batch = generators.generate_features_batch(
                "MolT5", ["CC", "CCC"], selected, batch_size=2,
            )
            np.testing.assert_array_equal(
                selected_batch,
                [[3.0, 2.0, 3.0], [4.0, 3.0, 4.0]],
            )
            selected_schema = generators.get_features_generator_schema(
                "MolT5", selected_feature_columns=selected,
            )
            self.assertEqual(selected_schema["feature_names"], selected)
            self.assertEqual(selected_schema["dimension"], 3)
            with self.assertRaisesRegex(KeyError, "embedding_2"):
                generator("CC", ["embedding_2"])

            generators.clear_pretrained_transformer_cache()
            generator("C")
            self.assertEqual(len(constructions), 2)

    def test_schema_has_order_dimension_dtype_and_versions(self):
        vector = generators.morgan_binary_features_generator("CCO")
        schema = generators.get_features_generator_schema("morgan", vector)
        self.assertEqual(schema["generator"], "morgan")
        self.assertEqual(schema["dimension"], 2048)
        self.assertEqual(schema["dtype"], "float64")
        self.assertEqual(schema["feature_names"][:3], ["bit_0", "bit_1", "bit_2"])
        self.assertEqual(schema["feature_names"][-1], "bit_2047")
        self.assertEqual(set(schema["versions"]), {"rdkit"})
        self.assertEqual(schema["generator_config"]["radius"], 2)

        optional_versions = generators._features_dependency_versions(
            ["rdkit_2d_normalized", "mordred", "padelpy", "map4", "secfp"],
        )
        self.assertEqual(
            set(optional_versions),
            {
                "rdkit", "descriptastorus", "scipy", "mordred",
                "mordredcommunity", "padelpy", "molfeat", "datamol",
                "mhfp",
            },
        )

        selected_vector = generators.morgan_binary_features_generator(
            "CCO", selected_feature_columns=["bit_0"],
        )
        selected_schema = generators.get_features_generator_schema(
            "morgan", selected_vector, selected_feature_columns=["bit_0"],
        )
        self.assertEqual(selected_schema["dimension"], 1)
        self.assertEqual(selected_schema["feature_names"], ["bit_0"])

    def test_batch_api_preserves_order_and_selected_columns(self):
        rows = generators.generate_features_batch(
            "rdkit_2d_208", ["CCO", "CCCC"], ["MolWt", "TPSA"],
        )
        self.assertEqual(len(rows), 2)
        np.testing.assert_allclose(rows[0], [46.069, 20.23], rtol=0, atol=1e-9)
        np.testing.assert_allclose(rows[1], [58.124, 0.0], rtol=0, atol=1e-9)

    def test_padel_missing_dependency_and_failure_are_explicit(self):
        with mock.patch.object(
            generators, "_load_padel_from_smiles", side_effect=ImportError("missing"),
        ):
            with self.assertRaisesRegex(ImportError, "missing"):
                generators.padelpy_features_generator("CC")

        row = OrderedDict((f"padel_{index}", str(index)) for index in range(1444))
        with mock.patch.object(generators, "_load_padel_from_smiles", return_value=lambda _: row):
            output = generators.padelpy_features_generator("CC")
        self.assertEqual(output.shape, (1444,))

        with mock.patch.object(
            generators, "_load_padel_from_smiles", return_value=mock.Mock(side_effect=RuntimeError("Java")),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "PaDEL failed.*SMILES 'CCC'.*Java",
            ):
                generators.padelpy_features_generator("CCC")

    def test_padel_batch_failure_reports_row_smiles_and_cause(self):
        row = OrderedDict((f"padel_{index}", str(index)) for index in range(1444))

        def fake_from_smiles(value):
            if isinstance(value, list):
                raise RuntimeError("batch Java failure")
            if value == "CCC":
                raise RuntimeError("molecule Java failure")
            return row

        with mock.patch.object(
            generators, "_load_padel_from_smiles", return_value=fake_from_smiles,
        ):
            with self.assertWarnsRegex(RuntimeWarning, "retrying 2 molecules"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "batch row 1.*SMILES 'CCC'.*molecule Java failure",
                ):
                    generators.padelpy_batch_features_generator(["CC", "CCC"])

    @unittest.skipUnless(importlib.util.find_spec("molfeat"), "molfeat is optional")
    def test_molfeat_011_generators_are_deterministic_and_batch_equivalent(self):
        expected_widths = {
            "fcfp": 2048,
            "fcfp_count": 2048,
            "topological": 2048,
            "topological_count": 2048,
            "layered": 2048,
            "avalon_count": 512,
            "rdkit_count": 2048,
            "atompair_count": 2048,
            "pattern": 2048,
            "estate": 79,
            "secfp": 2048,
            "cats2d": 189,
            "scaffoldkeys": 42,
            "pharm2d": 2048,
        }
        smiles = ["CCO", "c1ccccc1", ASPIRIN]
        for name, width in expected_widths.items():
            with self.subTest(generator=name):
                generator = generators.get_features_generator(name)
                scalar = np.stack([generator(value) for value in smiles])
                batch = np.stack(generators.generate_features_batch(name, smiles))
                self.assertEqual(scalar.shape, (3, width))
                self.assertEqual(scalar.dtype, np.dtype("float64"))
                np.testing.assert_array_equal(scalar, batch)
                np.testing.assert_array_equal(scalar[0], generator(smiles[0]))

        if importlib.util.find_spec("mhfp") is None:
            with self.assertRaisesRegex(ImportError, "mhfp"):
                generators.map4_features_generator("CCO")
        else:
            scalar = generators.map4_features_generator("CCO")
            batch = generators.generate_features_batch("map4", ["CCO", "CCC"])
            self.assertEqual(scalar.shape, (2048,))
            np.testing.assert_array_equal(scalar, batch[0])

        if importlib.util.find_spec("map4") is None:
            with self.assertRaisesRegex(ImportError, "map4>=1.1"):
                generators.map4_v1_1_features_generator("CCO")

        selected = ["Desc:0", "Desc:17"]
        selected_scalar = generators.pharmacophore_2d_features_generator(
            "CCO", selected,
        )
        selected_schema = generators.get_features_generator_schema(
            "pharm2d", selected_feature_columns=selected,
        )
        self.assertEqual(selected_schema["feature_names"], selected)
        self.assertEqual(selected_schema["dimension"], 2)
        np.testing.assert_array_equal(
            selected_scalar,
            generators.pharmacophore_2d_features_generator("CCO")[[0, 17]],
        )


class TestSaveFeaturesManifest(unittest.TestCase):
    def test_batched_iterator_is_bounded_and_validates_row_count(self):
        calls = []

        def batch(values):
            calls.append(list(values))
            return [np.asarray([len(value)]) for value in values]

        rows = list(save_features_script._iter_batched_features(["C", "CC", "CCC"], batch, 2))
        self.assertEqual(calls, [["C", "CC"], ["CCC"]])
        np.testing.assert_array_equal(np.stack(rows), [[1], [2], [3]])
        with self.assertRaisesRegex(RuntimeError, "returned 0 rows"):
            list(save_features_script._iter_batched_features(["CC"], lambda _: [], 1))

    def test_end_to_end_archive_and_atomic_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = os.path.join(directory, "molecules.csv")
            save_path = os.path.join(directory, "features.npz")
            with open(data_path, "w", encoding="utf-8") as file:
                file.write("smiles\nCCO\nCC\n")
            args = SimpleNamespace(
                data_path=data_path,
                smiles_column="smiles",
                features_generator="morgan",
                save_path=save_path,
                save_frequency=1,
                restart=False,
                sequential=False,
                num_workers=2,
                chunksize=1,
                batch_size=None,
            )
            save_features_script.generate_and_save_features(args)

            features = load_features(save_path)
            self.assertEqual(features.shape, (2, 2048))
            with open(save_path + ".manifest.json", encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual(manifest["num_molecules"], 2)
            self.assertEqual(manifest["dimension"], 2048)
            self.assertEqual(manifest["dtype"], "float64")
            self.assertEqual(manifest["storage"], "npz")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["num_molecules_completed"], 2)
            self.assertEqual(
                manifest["schema_version"],
                generators.FEATURE_GENERATOR_METADATA_SCHEMA_VERSION,
            )
            self.assertEqual(manifest["semantic_revision"], 1)
            self.assertNotIn("implementation_sha256", manifest)
            self.assertEqual(len(manifest["feature_names"]), 2048)
            self.assertEqual(len(manifest["input"]["data_sha256"]), 64)
            self.assertEqual(len(manifest["input"]["ordered_smiles_sha256"]), 64)
            self.assertFalse(os.path.exists(save_path + "_temp"))

    def test_resume_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = os.path.join(directory, "molecules.csv")
            save_path = os.path.join(directory, "features.npz")
            with open(data_path, "w", encoding="utf-8") as file:
                file.write("smiles\nCCO\nCC\n")
            args = SimpleNamespace(
                data_path=data_path,
                smiles_column="smiles",
                features_generator="morgan",
                save_path=save_path,
                save_frequency=10,
                restart=False,
                sequential=True,
                num_workers=None,
                chunksize=1,
                batch_size=None,
            )
            temp_dir = save_path + "_temp"
            os.mkdir(temp_dir)
            first_row = generators.morgan_binary_features_generator("CCO")
            save_features_script.save_features(os.path.join(temp_dir, "0.npz"), [first_row])
            identity = save_features_script._input_identity(data_path, ["CCO", "CC"])
            save_features_script._save_manifest(
                args, [first_row], storage="chunk_directory", temporary_file_count=1,
                input_identity=identity, total_molecules=2, status="in_progress",
            )

            with open(data_path, "a", encoding="utf-8") as file:
                file.write("CCC\n")
            with self.assertRaisesRegex(ValueError, "data_sha256.*ordered_smiles_sha256"):
                save_features_script.generate_and_save_features(args)

    def test_resume_rejects_changed_generator_semantic_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = os.path.join(directory, "molecules.csv")
            manifest_path = os.path.join(directory, "features.npz.manifest.json")
            with open(data_path, "w", encoding="utf-8") as file:
                file.write("smiles\nCCO\n")
            identity = save_features_script._input_identity(data_path, ["CCO"])
            expected = generators.get_features_generator_schema("morgan")
            manifest = dict(expected)
            manifest["semantic_revision"] += 1
            manifest["input"] = identity
            with open(manifest_path, "w", encoding="utf-8") as file:
                json.dump(manifest, file)

            with self.assertRaisesRegex(ValueError, "semantic_revision"):
                save_features_script._validate_resume_manifest(
                    manifest_path, expected, identity,
                )

    def test_resume_explicitly_rejects_legacy_identity_schema_once(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = os.path.join(directory, "molecules.csv")
            manifest_path = os.path.join(directory, "features.npz.manifest.json")
            with open(data_path, "w", encoding="utf-8") as file:
                file.write("smiles\nCCO\n")
            identity = save_features_script._input_identity(data_path, ["CCO"])
            expected = generators.get_features_generator_schema("morgan")
            legacy_manifest = dict(expected)
            legacy_manifest["schema_version"] = 1
            legacy_manifest.pop("semantic_revision")
            legacy_manifest["implementation_sha256"] = "0" * 64
            legacy_manifest["input"] = identity
            with open(manifest_path, "w", encoding="utf-8") as file:
                json.dump(legacy_manifest, file)

            with self.assertRaisesRegex(
                ValueError, "legacy whole-module.*Use --restart once",
            ):
                save_features_script._validate_resume_manifest(
                    manifest_path, expected, identity,
                )


if __name__ == "__main__":
    unittest.main()
