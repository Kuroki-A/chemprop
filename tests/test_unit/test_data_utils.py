"""Chemprop unit tests for chemprop/data/utils.py"""
import os
from unittest import TestCase
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np

from chemprop.data import get_header, preprocess_smiles_columns, get_task_names, get_mixed_task_names, \
    get_data_weights, get_smiles, filter_invalid_smiles, MoleculeDataset, MoleculeDatapoint, get_data, split_data, \
    validate_data, get_invalid_smiles_from_list


class TestGetHeader(TestCase):
    """
    Tests of the get_header function.
    """

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        with open(os.path.join(self.temp_dir.name, 'dummy_data.csv'), 'w') as f:
            data = 'column0,column1\nCC,10\nCCC,15'
            f.write(data)

    def test_correct_file(self):
        """ Test correct input """
        header = get_header(os.path.join(self.temp_dir.name, 'dummy_data.csv'))
        self.assertEqual(header, ['column0', 'column1'])

    def test_bad_path(self):
        """Test bad provided path """
        bad_path = os.path.join(self.temp_dir.name, 'bad_path.csv')
        with self.assertRaises(FileNotFoundError):
            get_header(bad_path)

    def test_empty_duplicate_and_blank_headers_are_rejected(self):
        for filename, contents, message in [
            ('empty.csv', '', 'header row'),
            ('empty_header.csv', '\nC,1\n', 'empty header row'),
            ('duplicate.csv', 'smiles,target,target\nC,1,2\n', 'duplicate'),
            ('blank.csv', 'smiles, ,target\nC,x,1\n', 'blank'),
        ]:
            path = os.path.join(self.temp_dir.name, filename)
            with open(path, 'w') as f:
                f.write(contents)
            with self.subTest(filename=filename), self.assertRaisesRegex(
                ValueError, message
            ):
                get_header(path)

    def test_utf8_bom_is_not_part_of_first_header_name(self):
        path = os.path.join(self.temp_dir.name, 'bom.csv')
        with open(path, 'w', encoding='utf-8-sig') as f:
            f.write('smiles,target\nC,1\n')

        self.assertEqual(get_header(path), ['smiles', 'target'])

    def test_validate_data_reports_empty_header_instead_of_crashing(self):
        path = os.path.join(self.temp_dir.name, 'empty_header.csv')
        with open(path, 'w') as f:
            f.write('\nC,1\n')

        errors = validate_data(path)

        self.assertEqual(len(errors), 1)
        self.assertIn('empty header row', next(iter(errors)))

    def tearDown(self):
        self.temp_dir.cleanup()


@patch(
    "chemprop.data.utils.get_header",
    lambda *args: [f'column{i}' for i in range(5)]
)
@patch(
    "os.path.isfile",
    lambda *args: True
)
class TestPreprocessSmiles(TestCase):
    """
    Tests of the preprocess_smiles_columns function
    """

    def test_basecase_1mol(self):
        """Test base case input with 1 molecule"""
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns=None,
            number_of_molecules=1,
        )
        self.assertEqual(smiles_columns, ['column0'])

    def test_2mol(self):
        """Test 2 molecule case"""
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns=None,
            number_of_molecules=2,
        )
        self.assertEqual(smiles_columns, ['column0', 'column1'])

    def test_specified_smiles(self):
        """Test specified smiles column"""
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns=['column3'],
            number_of_molecules=1,
        )
        self.assertEqual(smiles_columns, ['column3'])

    def test_out_of_order_smiles(self):
        """Test specified smiles columns provided in a different order"""
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns=['column3', 'column2'],
            number_of_molecules=2,
        )
        self.assertEqual(smiles_columns, ['column3', 'column2'])

    def test_input_not_list(self):
        """Test case where smiles_columns provided are not a list"""
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns='column3',
            number_of_molecules=1,
        )
        self.assertEqual(smiles_columns, ['column3'])

    def test_wrong_num_mol(self):
        """Test that error is raised when wrong number of molecules provided"""
        with self.assertRaises(ValueError):
            preprocess_smiles_columns(
                path='dummy_path.txt',
                smiles_columns=['column3'],
                number_of_molecules=2,
            )

    def test_smiles_not_in_file(self):
        """Test that error is raised whgen the provided smiles columns are not in the file"""
        with self.assertRaises(ValueError):
            preprocess_smiles_columns(
                path='dummy_path.txt',
                smiles_columns=['column3', 'not_in_file'],
                number_of_molecules=2,
            )


@patch(
    "chemprop.data.utils.get_header",
    lambda *args: [f'column{i}' for i in range(5)]
)
@patch(
    "chemprop.data.utils.preprocess_smiles_columns",
    lambda *args, **kwargs: ['column0', 'column1']  # default smiles columns if unspecified
)
class TestGetTaskNames(TestCase):
    """
    Tests of the get_task_names function.
    """

    def test_default_no_arguments(self):
        """Testing default behavior no arguments"""
        task_names = get_task_names(
            path='dummy_path.txt',
            smiles_columns=None,
            target_columns=None,
            ignore_columns=None,
        )
        self.assertEqual(task_names, ['column2', 'column3', 'column4'])

    def test_specified_smiles_columns(self):
        """Test behavior with smiles column specified"""
        task_names = get_task_names(
            path='dummy_path.txt',
            smiles_columns=['column0'],
            target_columns=None,
            ignore_columns=None,
        )
        self.assertEqual(task_names, ['column1', 'column2', 'column3', 'column4'])

    def test_specified_target_columns(self):
        """Test behavior with target columns specified"""
        task_names = get_task_names(
            path='dummy_path.txt',
            smiles_columns=None,
            target_columns=['column1', 'column3'],
            ignore_columns=None,
        )
        self.assertEqual(task_names, ['column1', 'column3'])

    def test_quantile_interval_duplicates_specified_target_columns(self):
        """Explicit targets still need lower and upper quantile heads."""
        task_names = get_task_names(
            path='dummy_path.txt',
            smiles_columns=None,
            target_columns=['column1', 'column3'],
            ignore_columns=None,
            loss_function='quantile_interval',
        )
        self.assertEqual(
            task_names,
            ['column1', 'column3', 'column1', 'column3'],
        )

    def test_ignore_columns(self):
        """Test behavior with ignore columns specified"""
        task_names = get_task_names(
            path='dummy_path.txt',
            smiles_columns=None,
            target_columns=None,
            ignore_columns=['column2', 'column3'],
        )
        self.assertEqual(task_names, ['column4'])


class TestGetMixedTaskNames(TestCase):
    """
    Tests of the get_mixed_task_names function.
    """

    def setUp(self):
        self.data_path = 'tests/data/atomic_bond_regression.csv'

    def test_default_no_arguments(self):
        """Testing default behavior no arguments"""
        atom_target_names, bond_target_names, molecule_target_names = get_mixed_task_names(
            path=self.data_path,
            smiles_columns=None,
            target_columns=None,
            ignore_columns=None,
            add_h=True,
        )
        self.assertEqual(atom_target_names, ['hirshfeld_charges', 'hirshfeld_charges_plus1', 'hirshfeld_charges_minus1',
                                             'hirshfeld_spin_density_plus1', 'hirshfeld_spin_density_minus1',
                                             'hirshfeld_charges_fukui_neu', 'hirshfeld_charges_fukui_elec', 'NMR'])
        self.assertEqual(bond_target_names, ['bond_length_matrix', 'bond_index_matrix'])
        self.assertEqual(molecule_target_names, ['homo', 'lumo'])

    def test_specified_target_columns(self):
        """Test behavior with target columns specified"""
        atom_target_names, bond_target_names, molecule_target_names = get_mixed_task_names(
            path=self.data_path,
            smiles_columns=None,
            target_columns=['hirshfeld_charges', 'bond_length_matrix'],
            ignore_columns=None,
            add_h=True,
        )
        self.assertEqual(atom_target_names, ['hirshfeld_charges'])
        self.assertEqual(bond_target_names, ['bond_length_matrix'])
        self.assertEqual(molecule_target_names, [])

    def test_ignore_columns(self):
        """Test behavior with ignore columns specified"""
        atom_target_names, bond_target_names, molecule_target_names = get_mixed_task_names(
            path=self.data_path,
            smiles_columns=None,
            target_columns=None,
            ignore_columns=['hirshfeld_charges', 'bond_length_matrix'],
            add_h=True,
        )
        self.assertEqual(atom_target_names,
                         ['hirshfeld_charges_plus1', 'hirshfeld_charges_minus1', 'hirshfeld_spin_density_plus1',
                          'hirshfeld_spin_density_minus1', 'hirshfeld_charges_fukui_neu',
                          'hirshfeld_charges_fukui_elec', 'NMR'])
        self.assertEqual(bond_target_names, ['bond_index_matrix'])
        self.assertEqual(molecule_target_names, ['homo', 'lumo'])


class TestGetDataWeights(TestCase):
    """
    Tests for the function get_data_weights.
    """

    def setUp(self):
        self.temp_dir = TemporaryDirectory()

    def test_base_case(self):
        """Testing already normalized inputs"""
        path = os.path.join(self.temp_dir.name, 'base_case.csv')
        with open(path, 'w') as f:
            f.write('weights\n1.5\n0.5\n1\n1\n1')
        weights = get_data_weights(path)
        self.assertEqual(weights, [1.5, 0.5, 1, 1, 1])

    def test_normalize_weights(self):
        """Testing weights that need to be normalized"""
        path = os.path.join(self.temp_dir.name, 'normalize.csv')
        with open(path, 'w') as f:
            f.write('weights\n3\n1\n2\n2\n2')
        weights = get_data_weights(path)
        self.assertEqual(weights, [1.5, 0.5, 1, 1, 1])

    def test_negative_weights(self):
        """Testing behavior when one of the weights is negative"""
        path = os.path.join(self.temp_dir.name, 'negativ.csv')
        with open(path, 'w') as f:
            f.write('weights\n3\n-1\n2\n2\n2')
        with self.assertRaises(ValueError):
            get_data_weights(path)

    def test_invalid_weight_sets(self):
        """Reject weights that cannot define a finite non-negative average."""
        for filename, contents in {
            'all_negative.csv': 'weights\n-1\n-2',
            'all_zero.csv': 'weights\n0\n0',
            'zero_sum.csv': 'weights\n1\n-1',
            'nan.csv': 'weights\nnan\n1',
            'inf.csv': 'weights\ninf\n1',
            'empty.csv': 'weights\n',
            'no_header.csv': '',
            'blank_row.csv': 'weights\n\n',
            'two_columns.csv': 'weights\n1,2\n',
        }.items():
            path = os.path.join(self.temp_dir.name, filename)
            with open(path, 'w') as f:
                f.write(contents)
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                get_data_weights(path)

    def tearDown(self):
        self.temp_dir.cleanup()


@patch(
    "chemprop.data.utils.preprocess_smiles_columns",
    lambda *args, **kwargs: ['column0', 'column1']  # default smiles columns if unspecified
)
class TestGetSmiles(TestCase):
    """
    Test for the get_smiles function.
    """

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.smiles_path = os.path.join(self.temp_dir.name, 'smiles.csv')
        with open(self.smiles_path, 'w') as f:
            f.write('column0,column1\nC,CC\nCC,CN\nO,CO')
        self.no_header_path = os.path.join(self.temp_dir.name, 'no_header.csv')
        with open(self.no_header_path, 'w') as f:
            f.write('C,CC\nCC,CN\nO,CO')

    def test_default_inputs(self):
        """Testing with no optional arguments."""
        smiles = get_smiles(
            path=self.smiles_path
        )
        self.assertEqual(smiles, [['C', 'CC'], ['CC', 'CN'], ['O', 'CO']])

    def test_specified_column_inputs(self):
        """Testing with a specified smiles column argument."""
        smiles = get_smiles(
            path=self.smiles_path,
            smiles_columns=['column1']
        )
        self.assertEqual(smiles, [['CC'], ['CN'], ['CO']])

    def test_specified_columns_changed_order(self):
        """Testing with no optional arguments."""
        smiles = get_smiles(
            path=self.smiles_path,
            smiles_columns=['column1', 'column0']
        )
        self.assertEqual(smiles, [['CC', 'C'], ['CN', 'CC'], ['CO', 'O']])

    def test_noheader_1mol(self):
        """Testing with no header"""
        smiles = get_smiles(
            path=self.no_header_path,
            header=False
        )
        self.assertEqual(smiles, [['C'], ['CC'], ['O']])

    def test_noheader_2mol(self):
        """Testing with no header and 2 molecules."""
        smiles = get_smiles(
            path=self.no_header_path,
            number_of_molecules=2,
            header=False
        )
        self.assertEqual(smiles, [['C', 'CC'], ['CC', 'CN'], ['O', 'CO']])

    def test_flatten(self):
        """Testing with flattened output"""
        smiles = get_smiles(
            path=self.smiles_path,
            flatten=True,
        )
        self.assertEqual(smiles, ['C', 'CC', 'CC', 'CN', 'O', 'CO'])

    def test_ragged_rows_are_rejected(self):
        for filename, contents in [
            ('too_short.csv', 'column0,column1\nC\n'),
            ('too_long.csv', 'column0,column1\nC,CC,CCC\n'),
        ]:
            path = os.path.join(self.temp_dir.name, filename)
            with open(path, 'w') as f:
                f.write(contents)
            with self.subTest(filename=filename), self.assertRaisesRegex(
                ValueError, 'row 2'
            ):
                get_smiles(path, smiles_columns=['column0', 'column1'])

    def tearDown(self):
        self.temp_dir.cleanup()


class TestFilterInvalidSmiles(TestCase):
    """
    Tests for the filter_invalid_smiles function.
    """

    def test_filter_good_smiles(self):
        """Test pass through for all good smiles"""
        smiles_list = [['C'], ['CC'], ['CN'], ['O']]
        dataset = MoleculeDataset([MoleculeDatapoint(s) for s in smiles_list])
        filtered_dataset = filter_invalid_smiles(dataset)
        self.assertEqual(filtered_dataset.smiles(), [['C'], ['CC'], ['CN'], ['O']])

    def test_filter_empty_smiles(self):
        """Test filter out empty smiles"""
        smiles_list = [['C'], ['CC'], [''], ['O']]
        dataset = MoleculeDataset([MoleculeDatapoint(s) for s in smiles_list])
        filtered_dataset = filter_invalid_smiles(dataset)
        self.assertEqual(filtered_dataset.smiles(), [['C'], ['CC'], ['O']])

    def test_no_heavy_smiles(self):
        """Test filter out smiles with no heavy atoms"""
        smiles_list = [['C'], ['CC'], ['[HH]'], ['O']]
        dataset = MoleculeDataset([MoleculeDatapoint(s) for s in smiles_list])
        filtered_dataset = filter_invalid_smiles(dataset)
        self.assertEqual(filtered_dataset.smiles(), [['C'], ['CC'], ['O']])

    def test_invalid_smiles(self):
        """Test filter out smiles with an invalid smiles"""
        smiles_list = [['C'], ['CC'], ['cccXc'], ['O']]
        dataset = MoleculeDataset([MoleculeDatapoint(s) for s in smiles_list])
        filtered_dataset = filter_invalid_smiles(dataset)
        self.assertEqual(filtered_dataset.smiles(), [['C'], ['CC'], ['O']])


class TestGetInvalidSmilesFromList(TestCase):
    def test_empty_input(self):
        self.assertEqual(get_invalid_smiles_from_list([]), [])

    def test_inconsistent_row_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'same number of molecules'):
            get_invalid_smiles_from_list([['C', 'CC'], ['CCC']])

    def test_empty_molecule_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least one molecule'):
            get_invalid_smiles_from_list([[]])


@patch(
    "chemprop.data.utils.get_data_weights",
    lambda *args, **kwargs: np.array([1, 1.5, 0.5])
)
@patch(
    "chemprop.data.utils.preprocess_smiles_columns",
    lambda *args, **kwargs: ['column0', 'column1']  # default smiles columns if unspecified
)
class TestGetData(TestCase):
    """
    Tests for the get_data function. Note, not including testing for args input because that may be removed.
    """

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.data_path = os.path.join(self.temp_dir.name, 'data.csv')
        with open(self.data_path, 'w') as f:
            f.write('column0,column1,column2,column3\nC,CC,0,1\nCC,CN,2,3\nO,CO,4,5')

    def test_return_dataset(self):
        """Testing the return type"""
        data = get_data(
            path=self.data_path
        )
        self.assertIsInstance(data, MoleculeDataset)

    def test_smiles(self):
        """Testing the base case smiles"""
        data = get_data(
            path=self.data_path
        )
        self.assertEqual(data.smiles(), [['C', 'CC'], ['CC', 'CN'], ['O', 'CO']])

    def test_targets(self):
        """Testing the base case targets"""
        data = get_data(
            path=self.data_path
        )
        self.assertEqual(data.targets(), [[0, 1], [2, 3], [4, 5]])

    def test_ragged_rows_are_rejected(self):
        for filename, row in [
            ('too_short.csv', 'C,CC,0'),
            ('too_long.csv', 'C,CC,0,1,extra'),
        ]:
            path = os.path.join(self.temp_dir.name, filename)
            with open(path, 'w') as f:
                f.write('column0,column1,column2,column3\n' + row + '\n')
            with self.subTest(filename=filename), self.assertRaisesRegex(
                ValueError, 'row 2'
            ):
                get_data(path=path)

    @patch(
        "chemprop.data.utils.load_features",
        lambda *args, **kwargs: np.array([[0, 1], [2, 3], [4, 5]])
    )
    def test_features(self):
        """Testing the features return"""
        data = get_data(
            path=self.data_path,
            features_path=['dummy_path.csv'],
        )
        print(data.features())
        self.assertTrue(np.array_equal(data.features(), [[0, 1], [2, 3], [4, 5]]))

    @patch(
        "chemprop.data.utils.load_features",
        lambda *args, **kwargs: np.array([[0, 1], [2, 3], [4, 5]])
    )
    def test_2features(self):
        """Testing the features return for two features paths"""
        data = get_data(
            path=self.data_path,
            features_path=['dummy_path.csv', 'also_dummy_path.csv'],
        )
        print(data.features())
        self.assertTrue(np.array_equal(data.features(), [[0, 1, 0, 1], [2, 3, 2, 3], [4, 5, 4, 5]]))

    def test_dataweights(self):
        """Testing the handling of data weights"""
        data = get_data(
            path=self.data_path,
            data_weights_path='dummy_path.csv'
        )
        self.assertEqual(data.data_weights(), [1, 1.5, 0.5])

    @patch(
        "chemprop.data.utils.load_features",
        lambda *args, **kwargs: np.array([[0, 1], [1, 0], [1, 0]])
    )
    def test_phase_features(self):
        """Testing the handling of phase features"""
        data = get_data(
            path=self.data_path,
            phase_features_path='dummy_path.csv'
        )
        self.assertTrue(np.array_equal(data.phase_features(), [[0, 1], [1, 0], [1, 0]]))

    @patch(
        "chemprop.data.utils.load_features",
        lambda *args, **kwargs: np.array([[0, 1], [1, 0], [1, 0]])
    )
    def test_features_and_phase_features(self):
        """Testing the handling of phase features"""
        data = get_data(
            path=self.data_path,
            features_path=['dummy_path.csv'],
            phase_features_path='dummy_path.csv'
        )
        self.assertTrue(np.array_equal(data.features(), [[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]]))

    @patch(
        "chemprop.data.utils.load_features",
        lambda *args, **kwargs: np.array([[0, 2], [1, 0], [1, 0]])
    )
    def test_invalid_phase_features(self):
        """Test that non-binary phase features are rejected."""
        with self.assertRaises(ValueError):
            get_data(
                path=self.data_path,
                features_path=['dummy_path.csv'],
                phase_features_path='dummy_path.csv'
            )

    def tearDown(self):
        self.temp_dir.cleanup()


class TestSplitData(TestCase):
    """
    Testing of the split_data function.
    """

    # Testing currently covers random and random_with_repeated_smiles
    def setUp(self):
        smiles_list = [['C', 'CC'], ['CC', 'CCC'], ['CCC', 'CN'], ['CN', 'CCN'], ['CCN', 'CCCN'], ['CCCN', 'CCCCN'],
                       ['CCCCN', 'CO'], ['CO', 'CCO'], ['CO', 'CCCO'], ['CN', 'CCC']]
        self.dataset = MoleculeDataset([MoleculeDatapoint(s) for s in smiles_list])

    def test_splits_sum1(self):
        with self.assertRaises(ValueError):
            train, val, test = split_data(
                data=self.dataset,
                sizes=(0.4, 0.8, 0.2)
            )

    def test_three_splits_provided(self):
        with self.assertRaises(ValueError):
            train, val, test = split_data(
                data=self.dataset,
                sizes=(0.8, 0.2)
            )

    def test_random_split(self):
        """Testing the random split with seed 0"""
        train, val, test = split_data(
            data=self.dataset
        )
        self.assertEqual(train.smiles(),
                         [['CO', 'CCO'], ['CO', 'CCCO'], ['CC', 'CCC'], ['CCCN', 'CCCCN'], ['CN', 'CCN'],
                          ['CCN', 'CCCN'], ['CCC', 'CN'], ['C', 'CC']])

    def test_seed1(self):
        """Testing the random split with seed 1"""
        train, val, test = split_data(
            data=self.dataset,
            seed=1,
        )
        self.assertEqual(train.smiles(),
                         [['CCCCN', 'CO'], ['CO', 'CCCO'], ['CN', 'CCC'], ['CO', 'CCO'], ['CCCN', 'CCCCN'],
                          ['CN', 'CCN'], ['C', 'CC'], ['CCN', 'CCCN']])

    def test_split_4_4_2(self):
        """Testing the random split with changed sizes"""
        train, val, test = split_data(
            data=self.dataset,
            sizes=(0.4, 0.4, 0.2)
        )
        self.assertEqual(train.smiles(), [['CO', 'CCO'], ['CO', 'CCCO'], ['CC', 'CCC'], ['CCCN', 'CCCCN']])

    def test_split_4_0_6(self):
        """Testing the random split with an empty set"""
        train, val, test = split_data(
            data=self.dataset,
            sizes=(0.4, 0, 0.6)
        )
        self.assertEqual(val.smiles(), [])

    def test_repeated_smiles(self):
        """Testing the random split with repeated smiles"""
        train, val, test = split_data(
            data=self.dataset,
            sizes=(0.4, 0.4, 0.2),
            split_type='random_with_repeated_smiles'
        )
        self.assertEqual(test.smiles(), [['CO', 'CCCO'], ['CO', 'CCO']])

    def test_molecular_weight_split(self):
        """Testing the molecular weight-based split."""
        train, val, test = split_data(
            data=self.dataset,
            sizes=(0.8, 0.1, 0.1),
            split_type='molecular_weight'
        )

        # sort the subsets by molecular weight
        sorted_train_mw = [dp.max_molwt for dp in train]
        sorted_val_mw = [dp.max_molwt for dp in val]
        sorted_test_mw = [dp.max_molwt for dp in test]

        # Assert that consecutive data are sorted in descending order
        assert np.all(np.diff(sorted_train_mw) >= 0)
        assert np.all(np.diff(sorted_val_mw) >= 0)
        assert np.all(np.diff(sorted_test_mw) >= 0)
