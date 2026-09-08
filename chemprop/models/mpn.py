from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function


class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int, hidden_size: int = None,
                 bias: bool = None, depth: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        :param hidden_size: Hidden layers dimension.
        :param bias: Whether to add bias to linear layers.
        :param depth: Number of message passing steps.
       """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size if hidden_size is None else hidden_size
        self.bias = args.bias if bias is None else bias
        self.depth = args.depth if depth is None else depth
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm
        self.is_atom_bond_targets = args.is_atom_bond_targets

        # Dropout
        self.dropout = nn.Dropout(args.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        if self.is_atom_bond_targets:
            self.W_o_b = nn.Linear(self.bond_fdim + self.hidden_size, self.hidden_size)

        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size,)

        if args.bond_descriptors == 'descriptor':
            self.bond_descriptors_size = args.bond_descriptors_size
            self.bond_descriptors_layer = nn.Linear(self.hidden_size + self.bond_descriptors_size,
                                                    self.hidden_size + self.bond_descriptors_size,)
    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None,
                bond_descriptors_batch: List[np.ndarray] = None) -> torch.Tensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atomic descriptors.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

        if atom_descriptors_batch is not None:
            if len(atom_descriptors_batch) != len(a_scope):
                raise ValueError(
                    'The number of atom descriptor arrays must match the number of molecules.'
                )
            atom_arrays = []
            for index, (descriptors, (_, atom_count)) in enumerate(
                zip(atom_descriptors_batch, a_scope)
            ):
                try:
                    descriptors = np.asarray(descriptors, dtype=np.float32)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f'Atom descriptors for molecule {index} must be numeric.'
                    ) from exc
                if descriptors.shape != (atom_count, self.atom_descriptors_size):
                    raise ValueError(
                        f'Atom descriptors for molecule {index} have shape '
                        f'{descriptors.shape}; expected '
                        f'{(atom_count, self.atom_descriptors_size)}.'
                    )
                if not np.isfinite(descriptors).all():
                    raise ValueError(
                        f'Atom descriptors for molecule {index} contain NaN or infinity.'
                    )
                atom_arrays.append(descriptors)
            padded_atom_descriptors = [
                np.zeros((1, self.atom_descriptors_size), dtype=np.float32),
                *atom_arrays,
            ]
            atom_descriptors_batch = torch.from_numpy(
                np.concatenate(padded_atom_descriptors, axis=0)
            ).to(self.device)

        if self.is_atom_bond_targets:
            b2br = mol_graph.get_b2br().to(self.device)
            if bond_descriptors_batch is not None:
                if len(bond_descriptors_batch) != len(b_scope):
                    raise ValueError(
                        'The number of bond descriptor arrays must match the number of molecules.'
                    )
                bond_arrays = []
                for index, (descriptors, (_, directed_bond_count)) in enumerate(
                    zip(bond_descriptors_batch, b_scope)
                ):
                    bond_count = directed_bond_count // 2
                    try:
                        descriptors = np.asarray(descriptors, dtype=np.float32)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f'Bond descriptors for molecule {index} must be numeric.'
                        ) from exc
                    if descriptors.shape != (bond_count, self.bond_descriptors_size):
                        raise ValueError(
                            f'Bond descriptors for molecule {index} have shape '
                            f'{descriptors.shape}; expected '
                            f'{(bond_count, self.bond_descriptors_size)}.'
                        )
                    if not np.isfinite(descriptors).all():
                        raise ValueError(
                            f'Bond descriptors for molecule {index} contain NaN or infinity.'
                        )
                    bond_arrays.append(descriptors)

                descriptors_tensor = torch.from_numpy(
                    np.concatenate(bond_arrays, axis=0)
                    if bond_arrays
                    else np.empty((0, self.bond_descriptors_size), dtype=np.float32)
                ).to(self.device)
                if len(descriptors_tensor) != len(b2br):
                    raise ValueError(
                        'The number of bond descriptor rows does not match the molecular graph.'
                    )
                # Keep both descriptor values and indices on the model device.
                # Indexing a NumPy array with CUDA scalar tensors fails on GPU.
                bond_descriptors_batch = descriptors_tensor.new_zeros(
                    (len(f_bonds), self.bond_descriptors_size)
                )
                if len(b2br) > 0:
                    bond_descriptors_batch[b2br[:, 0]] = descriptors_tensor
                    bond_descriptors_batch[b2br[:, 1]] = descriptors_tensor

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout(message)  # num_bonds x hidden

        # atom hidden
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout(atom_hiddens)  # num_atoms x hidden

        # bond hidden
        if self.is_atom_bond_targets:
            b_input = torch.cat([f_bonds, message], dim=1)  # num_bonds x (bond_fdim + hidden)
            bond_hiddens = self.act_func(self.W_o_b(b_input))  # num_bonds x hidden
            bond_hiddens = self.dropout(bond_hiddens)  # num_bonds x hidden

        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError('The number of atoms is different from the length of the extra atom features')

            atom_hiddens = torch.cat([atom_hiddens, atom_descriptors_batch], dim=1)     # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)                    # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.dropout(atom_hiddens)                             # num_atoms x (hidden + descriptor size)

        # concatenate the bond descriptors
        if self.is_atom_bond_targets and bond_descriptors_batch is not None:
            if len(bond_hiddens) != len(bond_descriptors_batch):
                raise ValueError('The number of bonds is different from the length of the extra bond features')

            bond_hiddens = torch.cat([bond_hiddens, bond_descriptors_batch], dim=1)     # num_bonds x (hidden + descriptor size)
            bond_hiddens = self.bond_descriptors_layer(bond_hiddens)                    # num_bonds x (hidden + descriptor size)
            bond_hiddens = self.dropout(bond_hiddens)                             # num_bonds x (hidden + descriptor size)

        # Readout
        if self.is_atom_bond_targets:
            return atom_hiddens, a_scope, bond_hiddens, b_scope, b2br  # num_atoms x hidden, remove the first one which is zero padding
        
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                else:
                    raise ValueError(
                        f'Aggregation "{self.aggregation}" is not supported. '
                        'Choose one of "mean", "sum", or "norm".'
                    )
                mol_vecs.append(mol_vec)
                '''
                if self.aggregation == 'norm':
                    mol_vec = self.agg_layer(mol_vec) / self.aggregation_norm
                else:
                    mol_vec = self.agg_layer(mol_vec)
                mol_vecs.append(mol_vec[0])
                '''

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN, self).__init__()
        self.reaction = args.reaction
        self.reaction_solvent = args.reaction_solvent
        self.atom_fdim = atom_fdim or get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    is_reaction=self.reaction if self.reaction is not False else self.reaction_solvent)
        self.bond_fdim = bond_fdim or get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    overwrite_default_bond=args.overwrite_default_bond_features,
                                                    atom_messages=args.atom_messages,
                                                    is_reaction=self.reaction if self.reaction is not False else self.reaction_solvent)
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.atom_descriptors = args.atom_descriptors
        self.bond_descriptors = args.bond_descriptors
        self.number_of_molecules = args.number_of_molecules
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features

        if self.features_only:
            return

        if not self.reaction_solvent:
            if args.mpn_shared:
                self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)] * args.number_of_molecules)
            else:
                self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                                             for _ in range(args.number_of_molecules)])
        else:
            self.encoder = MPNEncoder(args, self.atom_fdim, self.bond_fdim)
            # Set separate atom_fdim and bond_fdim for solvent molecules
            self.atom_fdim_solvent = get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                   is_reaction=False)
            self.bond_fdim_solvent = get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                   overwrite_default_bond=args.overwrite_default_bond_features,
                                                   atom_messages=args.atom_messages,
                                                   is_reaction=False)
            self.encoder_solvent = MPNEncoder(args, self.atom_fdim_solvent, self.bond_fdim_solvent,
                                              args.hidden_size_solvent, args.bias_solvent, args.depth_solvent)

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_descriptors_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.Tensor:
        """
        Encodes a batch of molecules.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if not isinstance(batch, (list, tuple)) or len(batch) == 0:
            raise ValueError('MPN input batch must be a non-empty list or tuple.')

        if not isinstance(batch[0], BatchMolGraph):
            if not all(isinstance(mols, (list, tuple)) for mols in batch):
                raise ValueError(
                    'Each MPN datapoint must contain a list or tuple of molecules.'
                )
            molecule_counts = {len(mols) for mols in batch}
            if molecule_counts != {self.number_of_molecules}:
                raise ValueError(
                    'Every MPN datapoint must contain exactly '
                    f'{self.number_of_molecules} molecule(s); got counts '
                    f'{sorted(molecule_counts)}.'
                )
            # Group first molecules, second molecules, etc for mol2graph
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]

            # TODO: handle atom_descriptors_batch with multiple molecules per input
            if self.atom_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        atom_features_batch=atom_features_batch,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            elif self.bond_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            else:
                batch = [mol2graph(b) for b in batch]
        elif not all(isinstance(graph, BatchMolGraph) for graph in batch):
            raise ValueError('MPN graph batches must contain only BatchMolGraph objects.')

        if self.use_input_features:
            if features_batch is None:
                raise ValueError('Input features are required when use_input_features is enabled.')
            try:
                feature_array = np.asarray(np.stack(features_batch), dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    'Input features must be a rectangular numeric matrix.'
                ) from exc
            if feature_array.ndim != 2 or feature_array.shape[0] == 0:
                raise ValueError(
                    f'Input features must be a non-empty 2D matrix; got shape '
                    f'{feature_array.shape}.'
                )
            if not np.isfinite(feature_array).all():
                raise ValueError('Input features contain NaN or infinity.')
            features_batch = torch.from_numpy(feature_array).to(self.device)

            if self.features_only:
                return features_batch

        if self.atom_descriptors == 'descriptor' or self.bond_descriptors == 'descriptor':
            if len(batch) > 1:
                raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                          'per input (i.e., number_of_molecules = 1).')

            if len(self.encoder) != len(batch):
                raise ValueError(
                    f'Expected {len(self.encoder)} molecular graph batches but '
                    f'received {len(batch)}.'
                )
            encodings = [enc(ba, atom_descriptors_batch, bond_descriptors_batch) for enc, ba in zip(self.encoder, batch)]
        else:
            if not self.reaction_solvent:
                if len(self.encoder) != len(batch):
                    raise ValueError(
                        f'Expected {len(self.encoder)} molecular graph batches but '
                        f'received {len(batch)}.'
                    )
                encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]
            else:
                encodings = []
                for ba in batch:
                    if ba.is_reaction:
                        encodings.append(self.encoder(ba))
                    else:
                        encodings.append(self.encoder_solvent(ba))

        output = encodings[0] if len(encodings) == 1 else torch.cat(encodings, dim=1)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)

            if output.shape[0] != features_batch.shape[0]:
                raise ValueError(
                    f'Molecular encodings contain {output.shape[0]} rows but '
                    f'input features contain {features_batch.shape[0]} rows.'
                )

            output = torch.cat([output, features_batch], dim=1)

        return output
