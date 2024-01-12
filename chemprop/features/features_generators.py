import inspect
from typing import Callable, List, Union

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Avalon import pyAvalonTools
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP, GetAvalonFP
from rdkit.ML.Descriptors import MoleculeDescriptors

from descriptastorus.descriptors import rdNormalizedDescriptors
from padelpy import from_smiles
#from unimol_tools import UniMolRepr
from mordred import Calculator, AcidBase, AdjacencyMatrix, Aromatic, AtomCount, BalabanJ, BaryszMatrix, BertzCT, BondCount, CarbonTypes, Constitutional, DistanceMatrix, EccentricConnectivityIndex, FragmentComplexity, Framework, HydrogenBond, InformationContent, KappaShapeIndex, LogS, McGowanVolume, MoeType, MolecularId, PathCount, Polarizability, RingCount, RotatableBond, SLogP, TopoPSA, TopologicalCharge, TopologicalIndex, VertexAdjacencyInformation, WalkCount, Weight, WienerIndex, ZagrebIndex

desc_list = [AcidBase, AdjacencyMatrix, Aromatic, AtomCount, BalabanJ, BaryszMatrix, BertzCT, BondCount, CarbonTypes, Constitutional, DistanceMatrix, EccentricConnectivityIndex, FragmentComplexity, Framework, HydrogenBond, InformationContent, KappaShapeIndex, LogS, McGowanVolume, MoeType, MolecularId, PathCount, Polarizability, RingCount, RotatableBond, SLogP, TopoPSA, TopologicalCharge, TopologicalIndex, VertexAdjacencyInformation, WalkCount, Weight, WienerIndex, ZagrebIndex]

#Generate columns for rdkit_2d w/o fragments
generator = rdNormalizedDescriptors.RDKit2DNormalized()
generator_columns = [list(i)[:1][0] for i in generator.columns]
c = pd.DataFrame(generator_columns)
c_ = c[~c[0].str.contains("fr_")]
wo_fr = [column[0] for column in c_.values]


Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]


FEATURES_GENERATOR_REGISTRY = {}


def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    """
    Creates a decorator which registers a features generator in a global dictionary to enable access by name.

    :param features_generator_name: The name to use to access the features generator.
    :return: A decorator which will add a features generator to the registry using the specified name.
    """
    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def get_features_generator(features_generator_name: str) -> FeaturesGenerator:
    """
    Gets a registered features generator by name.

    :param features_generator_name: The name of the features generator.
    :return: The desired features generator.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
                         f'If this generator relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
    """Returns a list of names of available features generators."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())


MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048


@register_features_generator('morgan')
def morgan_binary_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS,
                                     selected_feature_columns: list = None) -> np.ndarray:
    """
    Generates a binary Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the binary Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


@register_features_generator('morgan_count')
def morgan_counts_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS,
                                     selected_feature_columns: list = None) -> np.ndarray:
    """
    Generates a counts-based Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the counts-based Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol: Molecule,
                                    selected_feature_columns: list = None) -> np.ndarray:
        """
        Generates RDKit 2D features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdDescriptors.RDKit2D()
        features = generator.process(smiles)[1:]
        
        features = pd.DataFrame(features).T
        features.columns = generator_columns
        
        if selected_feature_columns is not None:
            features = features.loc[:, selected_feature_columns]

        return features.to_numpy()[0]

    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_normalized_features_generator(mol: Molecule,
                                               selected_feature_columns: list = None) -> np.ndarray:
        """
        Generates RDKit 2D normalized features for a molecule.

        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]
        
        features = pd.DataFrame(features).T
        features.columns = generator_columns
        
        if selected_feature_columns is not None:
            features = features.loc[:, selected_feature_columns]

        return features.to_numpy()[0]

except ImportError:
    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol: Molecule,
                                    selected_feature_columns: list = None) -> np.ndarray:
        """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
        raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                          '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D features.')

    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_normalized_features_generator(mol: Molecule,
                                               selected_feature_columns: list = None) -> np.ndarray:
        """Mock implementation raising an ImportError if descriptastorus cannot be imported."""
        raise ImportError('Failed to import descriptastorus. Please install descriptastorus '
                          '(https://github.com/bp-kelley/descriptastorus) to use RDKit 2D normalized features.')

        
@register_features_generator('rdkit_2d_wo_fr')
def custom_features_generator(mol: Molecule,
                              selected_feature_columns: list = None) -> np.ndarray:
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    generator = rdDescriptors.RDKit2D()
    features = generator.process(smiles)[1:]
    
    features = pd.DataFrame(features).T
    features.columns = generator_columns
    features = features.loc[:, wo_fr]
    
    if selected_feature_columns is not None:
        features = features.loc[:, selected_feature_columns]

    return features.to_numpy()[0]


@register_features_generator('rdkit_2d_normalized_wo_fr')
def custom_features_generator(mol: Molecule,
                              selected_feature_columns: list = None) -> np.ndarray:
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smiles)[1:]

    features = pd.DataFrame(features).T
    features.columns = generator_columns
    features = features.loc[:, wo_fr]
    
    if selected_feature_columns is not None:
        features = features.loc[:, selected_feature_columns]

    return features.to_numpy()[0]


@register_features_generator('rdkit_2d_208')
def custom_features_generator(mol: Molecule,
                              selected_feature_columns: list = None) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    
    desc_names = [x[0] for x in Descriptors._descList if x[0]]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    features = []
    for d in calc.CalcDescriptors(mol):
        features.append(d)
        
    features = pd.DataFrame(features).T
    features.columns = desc_names
    
    if selected_feature_columns is not None:
        features = features.loc[:, selected_feature_columns]

    return features.to_numpy()[0]


@register_features_generator('rdkit_2d_400')
def custom_features_generator(mol: Molecule,
                              selected_feature_columns: list = None) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    features = []
    desc_names = []
    for desc_name in inspect.getmembers(Descriptors, inspect.isfunction):
        desc_name = desc_name[0]
        if desc_name.startswith("_"):
            continue
        if desc_name == "setupAUTOCorrDescriptors":
            continue
        if desc_name == "CalcMolDescriptors":
            continue
        desc_names.append(desc_name)
        features.append(getattr(Descriptors, desc_name)(mol))
        
    features = pd.DataFrame(features).T
    features.columns = desc_names

    if selected_feature_columns is not None:
        features = features.loc[:, selected_feature_columns]

    return features.to_numpy()[0]


@register_features_generator('maccs')
def custom_features_generator(mol: Molecule,
                              selected_feature_columns: list = None) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    return np.array(AllChem.GetMACCSKeysFingerprint(mol), int)


@register_features_generator('rdkit')
def custom_features_generator(mol: Molecule,
                              selected_feature_columns: list = None) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    return np.array(Chem.RDKFingerprint(mol), int)


@register_features_generator('avalon')
def custom_features_generator(mol: Molecule,
                              selected_feature_columns: list = None) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    return np.array(pyAvalonTools.GetAvalonFP(mol), int)


@register_features_generator('atompair')
def custom_features_generator(mol: Molecule,
                              selected_feature_columns: list = None) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    return np.array(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol), int)


@register_features_generator('erg')
def custom_features_generator(mol: Molecule,
                              selected_feature_columns: list = None) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    return np.array(AllChem.GetErGFingerprint(mol), int)


@register_features_generator('mordred')
def custom_features_generator(mol: Molecule,
                              selected_feature_columns: list = None) -> np.ndarray:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    mol_list = []
    mol_list.append(mol)
    
    calc = Calculator()
    for desc in desc_list:
        calc.register(desc)
        
    df = calc.pandas(mol_list)
    
    if selected_feature_columns is not None:
        df = df.loc[:, selected_feature_columns]

    return df.values[0].astype('float')


@register_features_generator('padelpy')
def custom_features_generator(mol: Molecule,
                              selected_feature_columns: list = None) -> np.ndarray:
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    try:
        descriptors = from_smiles(smiles)
        features = pd.DataFrame.from_dict(descriptors, orient='index').T.iloc[:, :1444]
        
        if selected_feature_columns is not None:
            features = features.loc[:, selected_feature_columns]
        
        return features.to_numpy()[0].astype('float')
    except:
        print('Timeout may be occurred.')
        descriptors = from_smiles('c1ccccc1')
        features = pd.DataFrame.from_dict(descriptors, orient='index').T.iloc[:, :1444]
        
        if selected_feature_columns is not None:
            features = features.loc[:, selected_feature_columns]
        
        return np.zeros(len(features.columns))

"""
@register_features_generator('unimol')
def custom_features_generator(mol: Molecule,
                              selected_feature_columns: list = None) -> np.ndarray:
    clf = UniMolRepr(data_type='molecule')
    smiles = [Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol]

    reprs = clf.get_repr(smiles)
    features = np.array(reprs["cls_repr"][0])

    return features
"""

"""
Custom features generator template.

Note: The name you use to register the features generator is the name
you will specify on the command line when using the --features_generator <name> flag.
Ex. python train.py ... --features_generator custom ...

@register_features_generator('custom')
def custom_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    features = np.array([0, 0, 1])

    return features
"""