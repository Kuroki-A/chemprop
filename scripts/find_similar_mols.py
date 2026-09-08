"""Script to find the molecules in the training set which are most similar to each molecule in the test set."""

from collections import OrderedDict
import csv
import os
import sys
from typing import List
from typing_extensions import Literal

from rdkit import DataStructs
from rdkit import Chem

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import get_data_from_smiles, get_smiles, MoleculeDataLoader
from chemprop.features import morgan_binary_features_generator
from chemprop.models import MoleculeModel
from chemprop.utils import load_checkpoint, makedirs
from chemprop.train import model_fingerprint


class Args(Tap):
    test_path: str  # Path to CSV file with test set of molecules
    train_path: str  # Path to CSV file with train set of molecules
    save_path: str  # Path to CSV file where similar molecules will be saved
    distance_measure: Literal['embedding', 'morgan', 'tanimoto'] = 'embedding'  # Distance measure to use to find nearest neighbors in train set
    checkpoint_path: str = None  # Path to .pt file containing a model checkpoint (only needed for distance_measure == "embedding")
    num_neighbors: int = 5  # Number of neighbors to search for each molecule
    batch_size: int = 50  # Batch size when making predictions
    smiles_column: str = None # Columns in dataset CSV file containing SMILES
    num_workers: int = 8 # Number of workers used to build batches.


def find_similar_mols(test_smiles: List[str],
                      train_smiles: List[str],
                      distance_measure: str,
                      model: MoleculeModel = None,
                      num_neighbors: int = None,
                      batch_size: int = 50,
                      num_workers: int = 0) -> List[OrderedDict]:
    """
    For each test molecule, finds the N most similar training molecules according to some distance measure.

    :param test_smiles: A list of test SMILES strings.
    :param train_smiles: A list of train SMILES strings.
    :param model: A trained MoleculeModel (only needed for distance_measure == 'embedding').
    :param distance_measure: The distance measure to use to determine nearest neighbors.
    :param num_neighbors: The number of nearest training molecules to find for each test molecule.
    :param batch_size: Batch size.
    :return: A list of OrderedDicts containing the test smiles, the num_neighbors nearest training smiles,
    and other relevant distance info.
    """
    if not test_smiles or not train_smiles:
        raise ValueError('Test and training SMILES collections must both be non-empty.')
    if distance_measure not in {'embedding', 'morgan', 'tanimoto'}:
        raise ValueError(f'Distance measure {distance_measure!r} is not supported.')
    def invalid_smiles(smiles_values):
        invalid = []
        for smiles in smiles_values:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None or mol.GetNumHeavyAtoms() == 0:
                invalid.append(smiles)
        return invalid

    invalid_test = invalid_smiles(test_smiles)
    invalid_train = invalid_smiles(train_smiles)
    if invalid_test or invalid_train:
        raise ValueError(
            'All SMILES must be valid so neighbor rows remain aligned with the '
            f'input files (invalid test={len(invalid_test)}, train={len(invalid_train)}).'
        )
    if num_neighbors is None:
        num_neighbors = 5
    if not isinstance(num_neighbors, int) or isinstance(num_neighbors, bool) or num_neighbors <= 0:
        raise ValueError('num_neighbors must be a positive integer.')
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer.')
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError('num_workers must be a non-negative integer.')
    num_neighbors = min(num_neighbors, len(train_smiles))

    test_data = get_data_from_smiles(smiles=[[smiles] for smiles in test_smiles])
    train_data = get_data_from_smiles(smiles=[[smiles] for smiles in train_smiles])
    train_smiles_set = set(train_smiles)

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers
    )
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers
    )

    print(f'Computing {distance_measure} vectors')
    if distance_measure == 'embedding':
        if model is None:
            raise ValueError(
                'A trained model is required when distance_measure="embedding".'
            )
        test_vecs = np.array(model_fingerprint(model=model, data_loader=test_data_loader, fingerprint_type='last_FFN'))
        train_vecs = np.array(model_fingerprint(model=model, data_loader=train_data_loader, fingerprint_type='last_FFN'))
        metric = 'cosine'
    elif distance_measure == 'morgan':
        test_vecs = np.array([morgan_binary_features_generator(smiles) for smiles in tqdm(test_smiles, total=len(test_smiles))])
        train_vecs = np.array([morgan_binary_features_generator(smiles) for smiles in tqdm(train_smiles, total=len(train_smiles))])
        metric = 'jaccard'
    elif distance_measure == 'tanimoto':
        # Generate RDKit topological fingerprints
        test_fps = [Chem.RDKFingerprint(m.mol[0]) for m in tqdm(test_data)]
        train_fps = [Chem.RDKFingerprint(m.mol[0]) for m in tqdm(train_data)]

        # Compute pairwise similarity
        print('Computing distances')
        similarity = np.zeros([len(test_fps), len(train_fps)])
        for (x, y), _ in np.ndenumerate(similarity):
            similarity[x, y] = DataStructs.FingerprintSimilarity(test_fps[x], train_fps[y])

        # Convert the tanimoto similarity to a distance
        distances = 1 - similarity
        metric = 'tanimoto'
    else:
        raise ValueError(f'Distance measure "{distance_measure}" not supported.')

    if distance_measure in ('embedding', 'morgan'):
        print('Computing distances')
        distances = cdist(test_vecs, train_vecs, metric=metric)
    if distances.shape != (len(test_smiles), len(train_smiles)):
        raise ValueError(
            f'Distance matrix has shape {distances.shape}; expected '
            f'{(len(test_smiles), len(train_smiles))}.'
        )
    if not np.isfinite(distances).all():
        raise ValueError(
            'Distance computation produced NaN or infinity. For embedding '
            'distance, this can indicate an all-zero model embedding.'
        )

    print('Finding neighbors')
    neighbors = []
    for test_index, test_smile in enumerate(test_smiles):
        # Find the num_neighbors molecules in the training set which are most similar to the test molecule
        nearest_train_indices = np.argsort(distances[test_index])[:num_neighbors]

        # Build dictionary with distance info
        neighbor = OrderedDict()
        neighbor['test_smiles'] = test_smile
        neighbor['test_in_train'] = test_smile in train_smiles_set

        for i, train_index in enumerate(nearest_train_indices):
            neighbor[f'train_{i + 1}_smiles'] = train_smiles[train_index]
            neighbor[f'train_{i + 1}_{distance_measure}_{metric}_distance'] = distances[test_index][train_index]

        neighbors.append(neighbor)

    return neighbors


def find_similar_mols_from_file(test_path: str,
                                train_path: str,
                                distance_measure: str,
                                checkpoint_path: str = None,
                                num_neighbors: int = 5,
                                batch_size: int = 50,
                                smiles_column: str = None,
                                num_workers: int = 0) -> List[OrderedDict]:
    """
    For each test molecule, finds the N most similar training molecules according to some distance measure.
    Loads molecules and model from file.

    :param test_path: Path to a CSV file containing test SMILES.
    :param train_path: Path to a CSV file containing train SMILES.
    :param checkpoint_path: Path to a .pt model checkpoint file (only needed for distance_measure == 'embedding').
    :param distance_measure: The distance measure to use to determine nearest neighbors.
    :param num_neighbors: The number of nearest training molecules to find for each test molecule.
    :param batch_size: Batch size.
    :return: A list of OrderedDicts containing the test smiles, the num_neighbors nearest training smiles,
    and other relevant distance info.
    """
    print('Loading data')
    test_smiles, train_smiles = get_smiles(test_path, flatten=True, smiles_columns=smiles_column), get_smiles(train_path, flatten=True, smiles_columns=smiles_column)

    if distance_measure not in {'embedding', 'morgan', 'tanimoto'}:
        raise ValueError(f'Distance measure {distance_measure!r} is not supported.')
    if distance_measure == 'embedding' and checkpoint_path is None:
        raise ValueError(
            'checkpoint_path is required when distance_measure="embedding".'
        )
    if distance_measure != 'embedding' and checkpoint_path is not None:
        raise ValueError(
            'checkpoint_path is only used when distance_measure="embedding".'
        )

    if checkpoint_path is not None:
        print('Loading model')
        model = load_checkpoint(checkpoint_path)
    else:
        model = None

    return find_similar_mols(
        test_smiles=test_smiles,
        train_smiles=train_smiles,
        distance_measure=distance_measure,
        model=model,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def save_similar_mols(test_path: str,
                      train_path: str,
                      save_path: str,
                      distance_measure: str,
                      checkpoint_path: str = None,
                      num_neighbors: int = None,
                      batch_size: int = 50,
                      smiles_column: str = None,
                      num_workers: int = 0):
    """
    For each test molecule, finds the N most similar training molecules according to some distance measure.
    Loads molecules and model from file and saves results to file.

    :param test_path: Path to a CSV file containing test SMILES.
    :param train_path: Path to a CSV file containing train SMILES.
    :param checkpoint_path: Path to a .pt model checkpoint file (only needed for distance_measure == 'embedding').
    :param save_path: Path to a CSV file where the results will be saved.
    :param distance_measure: The distance measure to use to determine nearest neighbors.
    :param num_neighbors: The number of nearest training molecules to find for each test molecule.
    :param batch_size: Batch size.
    :return: A list of OrderedDicts containing the test smiles, the num_neighbors nearest training smiles,
    and other relevant distance info.
    """

    # Find similar molecules
    similar_mols = find_similar_mols_from_file(
        test_path=test_path,
        train_path=train_path,
        checkpoint_path=checkpoint_path,
        distance_measure=distance_measure,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        smiles_column=smiles_column,
        num_workers=num_workers,
    )

    # Save results
    makedirs(save_path, isfile=True)

    with open(save_path, 'w', newline="") as f:
        writer = csv.DictWriter(f, fieldnames=similar_mols[0].keys())
        writer.writeheader()
        for row in similar_mols:
            writer.writerow(row)


if __name__ == '__main__':
    args = Args().parse_args()

    save_similar_mols(
        test_path=args.test_path,
        train_path=args.train_path,
        save_path=args.save_path,
        distance_measure=args.distance_measure,
        checkpoint_path=args.checkpoint_path,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        smiles_column=args.smiles_column,
        num_workers=args.num_workers,
    )
