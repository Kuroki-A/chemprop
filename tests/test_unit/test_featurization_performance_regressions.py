"""Regression coverage for sparse graph construction optimizations."""

import numpy as np
from rdkit import Chem

from chemprop.features import (
    MolGraph,
    onek_encoding_unk,
    reset_featurization_parameters,
    set_reaction,
)
from chemprop.features.featurization import (
    PARAMS,
    atom_features,
    atom_features_zeros,
    bond_features,
    map_reac_to_prod,
)


def _historical_reaction_graph(mol_reac, mol_prod, mode):
    """Reference for the original all-atom-pairs reaction implementation."""
    ri2pi, pio, rio = map_reac_to_prod(mol_reac, mol_prod)
    if mode in {'reac_diff', 'prod_diff', 'reac_prod'}:
        f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms()] + [
            atom_features_zeros(mol_prod.GetAtomWithIdx(index)) for index in pio
        ]
        f_atoms_prod = [
            atom_features(mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()]))
            if atom.GetIdx() not in rio else atom_features_zeros(atom)
            for atom in mol_reac.GetAtoms()
        ] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]
    else:
        f_atoms_reac = [atom_features(atom) for atom in mol_reac.GetAtoms()] + [
            atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio
        ]
        f_atoms_prod = [
            atom_features(mol_prod.GetAtomWithIdx(ri2pi[atom.GetIdx()]))
            if atom.GetIdx() not in rio else atom_features(atom)
            for atom in mol_reac.GetAtoms()
        ] + [atom_features(mol_prod.GetAtomWithIdx(index)) for index in pio]

    f_atoms_diff = [
        [product - reactant for product, reactant in zip(prod, reac)]
        for prod, reac in zip(f_atoms_prod, f_atoms_reac)
    ]
    if mode in {'reac_prod', 'reac_prod_balance'}:
        f_atoms = [
            reac + prod[PARAMS.MAX_ATOMIC_NUM + 1:]
            for reac, prod in zip(f_atoms_reac, f_atoms_prod)
        ]
    elif mode in {'reac_diff', 'reac_diff_balance'}:
        f_atoms = [
            reac + diff[PARAMS.MAX_ATOMIC_NUM + 1:]
            for reac, diff in zip(f_atoms_reac, f_atoms_diff)
        ]
    else:
        f_atoms = [
            prod + diff[PARAMS.MAX_ATOMIC_NUM + 1:]
            for prod, diff in zip(f_atoms_prod, f_atoms_diff)
        ]

    n_atoms = len(f_atoms)
    n_atoms_reac = mol_reac.GetNumAtoms()
    n_bonds = 0
    f_bonds = []
    a2b = [[] for _ in range(n_atoms)]
    b2a = []
    b2revb = []
    balanced = mode in {
        'reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance',
    }
    for a1 in range(n_atoms):
        for a2 in range(a1 + 1, n_atoms):
            if a1 >= n_atoms_reac and a2 >= n_atoms_reac:
                bond_prod = mol_prod.GetBondBetweenAtoms(
                    pio[a1 - n_atoms_reac], pio[a2 - n_atoms_reac],
                )
                bond_reac = bond_prod if balanced else None
            elif a1 < n_atoms_reac and a2 >= n_atoms_reac:
                bond_reac = None
                bond_prod = (
                    mol_prod.GetBondBetweenAtoms(
                        ri2pi[a1], pio[a2 - n_atoms_reac],
                    )
                    if a1 in ri2pi else None
                )
            else:
                bond_reac = mol_reac.GetBondBetweenAtoms(a1, a2)
                if a1 in ri2pi and a2 in ri2pi:
                    bond_prod = mol_prod.GetBondBetweenAtoms(ri2pi[a1], ri2pi[a2])
                elif balanced:
                    bond_prod = (
                        None if a1 in ri2pi or a2 in ri2pi else bond_reac
                    )
                else:
                    bond_prod = None

            if bond_reac is None and bond_prod is None:
                continue
            f_bond_reac = bond_features(bond_reac)
            f_bond_prod = bond_features(bond_prod)
            if mode in {
                'reac_diff', 'prod_diff',
                'reac_diff_balance', 'prod_diff_balance',
            }:
                f_bond_diff = [
                    product - reactant
                    for reactant, product in zip(f_bond_reac, f_bond_prod)
                ]
            if mode in {'reac_prod', 'reac_prod_balance'}:
                f_bond = f_bond_reac + f_bond_prod
            elif mode in {'reac_diff', 'reac_diff_balance'}:
                f_bond = f_bond_reac + f_bond_diff
            else:
                f_bond = f_bond_prod + f_bond_diff

            f_bonds.extend([f_atoms[a1] + f_bond, f_atoms[a2] + f_bond])
            b1, b2 = n_bonds, n_bonds + 1
            a2b[a2].append(b1)
            b2a.append(a1)
            a2b[a1].append(b2)
            b2a.append(a2)
            b2revb.extend([b2, b1])
            n_bonds += 2

    return {
        'n_atoms': n_atoms,
        'n_bonds': n_bonds,
        'f_atoms': f_atoms,
        'f_bonds': f_bonds,
        'a2b': a2b,
        'b2a': b2a,
        'b2revb': b2revb,
    }


def test_sparse_bond_iteration_preserves_historical_atom_pair_order():
    molecule = Chem.RWMol()
    for _ in range(4):
        molecule.AddAtom(Chem.Atom(6))

    # Deliberately create RDKit bond indices in a different order from the
    # historical nested atom-pair traversal.
    molecule.AddBond(2, 3, Chem.BondType.SINGLE)  # RDKit bond 0
    molecule.AddBond(0, 1, Chem.BondType.SINGLE)  # RDKit bond 1
    molecule.AddBond(1, 2, Chem.BondType.SINGLE)  # RDKit bond 2
    molecule = molecule.GetMol()
    Chem.SanitizeMol(molecule)

    graph = MolGraph(molecule)

    # Directed bonds remain ordered by (0,1), (1,2), (2,3), while b2br still
    # indexes rows by RDKit's original bond index.
    np.testing.assert_array_equal(
        graph.b2br,
        [[4, 5], [0, 1], [2, 3]],
    )
    assert graph.b2a == [0, 1, 1, 2, 2, 3]
    assert graph.b2revb == [1, 0, 3, 2, 5, 4]


def test_one_hot_unknown_encoding_retains_public_semantics():
    assert onek_encoding_unk(20, (10, 20, 30)) == [0, 1, 0, 0]
    assert onek_encoding_unk(99, (10, 20, 30)) == [0, 0, 0, 1]


def test_sparse_reaction_bond_union_matches_historical_all_pairs_reference():
    modes = (
        'reac_prod', 'reac_diff', 'prod_diff',
        'reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance',
    )
    reaction_smiles = (
        # Bond-order change, deleted mapped atom, and product-only unmapped atom.
        '[CH3:1][CH2:2][Br:3]>>[CH3:1][CH:2]=[O]',
        # Product-only mapped atom plus a disconnected reactant-only atom.
        '[CH3:1].[Cl-:2]>>[CH3:1][OH:3]',
    )

    try:
        for mode in modes:
            reset_featurization_parameters()
            set_reaction(True, mode)
            for reaction in reaction_smiles:
                reactant, _, product = reaction.split('>')
                mols = (
                    Chem.MolFromSmiles(reactant),
                    Chem.MolFromSmiles(product),
                )
                expected = _historical_reaction_graph(*mols, mode)
                actual = MolGraph(mols)

                assert actual.n_atoms == expected['n_atoms']
                assert actual.n_bonds == expected['n_bonds']
                np.testing.assert_array_equal(actual.f_atoms, expected['f_atoms'])
                np.testing.assert_array_equal(actual.f_bonds, expected['f_bonds'])
                assert actual.a2b == expected['a2b']
                assert actual.b2a == expected['b2a']
                assert actual.b2revb == expected['b2revb']
    finally:
        reset_featurization_parameters()
