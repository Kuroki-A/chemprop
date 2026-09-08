import json
import os
from tempfile import TemporaryDirectory
import pickle
from typing import Dict, List, Optional, Union, get_args, get_origin
from typing_extensions import Literal
from packaging import version
from warnings import warn

import torch
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)
import numpy as np

import chemprop.data.utils
from chemprop.data import set_cache_mol, empty_cache
from chemprop.features import get_available_features_generators


Metric = Literal['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy', 'binary_cross_entropy', 'sid', 'wasserstein', 'f1', 'mcc', 'bounded_rmse', 'bounded_mae', 'bounded_mse',
                'recall', 'precision','balanced_accuracy']


def _is_finite_number(value) -> bool:
    """Returns whether ``value`` is a finite, non-boolean real scalar."""
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )


def _config_value_matches_annotation(value, annotation) -> bool:
    """Checks JSON values against the CLI annotation which Tap normally enforces.

    ``config_path`` values are assigned after command-line parsing, so they do
    not pass through argparse's type and ``choices`` checks.  This deliberately
    implements only the JSON-representable annotations used by these argument
    classes; an unknown annotation is left for the argument's semantic
    validation rather than being guessed here.
    """
    origin = get_origin(annotation)
    annotation_args = get_args(annotation)

    if origin is Literal:
        return value in annotation_args
    if origin in (list, List):
        return isinstance(value, list) and (
            not annotation_args
            or all(_config_value_matches_annotation(item, annotation_args[0]) for item in value)
        )
    if origin in (dict, Dict):
        if not isinstance(value, dict):
            return False
        if len(annotation_args) != 2:
            return True
        key_type, value_type = annotation_args
        return all(
            _config_value_matches_annotation(key, key_type)
            and _config_value_matches_annotation(item, value_type)
            for key, item in value.items()
        )
    if origin is Union:
        return any(_config_value_matches_annotation(value, option) for option in annotation_args)
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is str:
        return isinstance(value, str)
    return True


def get_checkpoint_paths(checkpoint_path: Optional[str] = None,
                         checkpoint_paths: Optional[List[str]] = None,
                         checkpoint_dir: Optional[str] = None,
                         ext: str = '.pt') -> Optional[List[str]]:
    """
    Gets a list of checkpoint paths either from a single checkpoint path or from a directory of checkpoints.

    If :code:`checkpoint_path` is provided, only collects that one checkpoint.
    If :code:`checkpoint_paths` is provided, collects all of the provided checkpoints.
    If :code:`checkpoint_dir` is provided, walks the directory and collects all checkpoints.
    A checkpoint is any file ending in the extension ext.

    :param checkpoint_path: Path to a checkpoint.
    :param checkpoint_paths: List of paths to checkpoints.
    :param checkpoint_dir: Path to a directory containing checkpoints.
    :param ext: The extension which defines a checkpoint file.
    :return: A list of paths to checkpoints or None if no checkpoint path(s)/dir are provided.
    """
    if sum(var is not None for var in [checkpoint_dir, checkpoint_path, checkpoint_paths]) > 1:
        raise ValueError('Can only specify one of checkpoint_dir, checkpoint_path, and checkpoint_paths')

    expected_extension = ext.lower()

    def validate_extensions(paths: List[str]) -> List[str]:
        invalid_paths = [
            path
            for path in paths
            if os.path.splitext(os.fspath(path))[1].lower() != expected_extension
        ]
        if invalid_paths:
            raise ValueError(
                f'Expected checkpoint files with extension "{ext}", but received: '
                + ', '.join(map(os.fspath, invalid_paths))
            )
        return paths

    if checkpoint_path is not None:
        return validate_extensions([checkpoint_path])

    if checkpoint_paths is not None:
        return validate_extensions(checkpoint_paths)

    if checkpoint_dir is not None:
        checkpoint_paths = []

        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if os.path.splitext(fname)[1].lower() == expected_extension:
                    checkpoint_paths.append(os.path.join(root, fname))

        if len(checkpoint_paths) == 0:
            raise ValueError(f'Failed to find any checkpoints with extension "{ext}" in directory "{checkpoint_dir}"')

        # ``os.walk`` does not guarantee traversal order. A stable checkpoint
        # order is important because ensemble member indices are exposed in
        # prediction outputs.
        return sorted(checkpoint_paths)

    return None


class CommonArgs(Tap):
    """:class:`CommonArgs` contains arguments that are used in both :class:`TrainArgs` and :class:`PredictArgs`."""

    smiles_columns: List[str] = None
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    number_of_molecules: int = 1
    """Number of molecules in each input to the model.
    This must equal the length of :code:`smiles_columns` (if not :code:`None`)."""
    checkpoint_dir: str = None
    """Directory from which to load model checkpoints (walks directory and ensembles all models that are found)."""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pt` file)."""
    checkpoint_paths: List[str] = None
    """List of paths to model checkpoints (:code:`.pt` files)."""
    selected_features_path: str = None
    """Path to selected features csv (:code:`.csv` file)."""    
    features_generator_metadata: Dict = None
    """Ordered feature schema metadata stored in new checkpoints for prediction validation."""
    features_source_metadata: Dict = None
    """Structure of generated, external, and phase feature inputs stored in new checkpoints."""
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""
    gpu: int = None
    """Which GPU to use."""
    early_stopping: int = 5
    """Number of early stopping counts"""
    data_type: Literal['validation', 'test'] = 'test'
    """Output scores of cross_validate.py."""
    use_cache: bool = False
    """Whether to cache loaded datasets in a content-addressed ``.chemprop_cache`` directory.
    Set ``CHEMPROP_CACHE_DIR`` to choose a private user-owned cache location.
    Cached datasets use trusted Python pickle data and must never be shared with
    untrusted users."""
    features_generator: List[str] = None
    """Method(s) of generating additional features."""
    features_path: List[str] = None
    """Path(s) to features to use in FNN (instead of features_generator)."""
    phase_features_path: str = None
    """Path to features used to indicate the phase of the data in one-hot vector form. Used in spectra datatype."""
    no_features_scaling: bool = False
    """Turn off scaling of features."""
    max_data_size: int = None
    """Maximum number of data points to load."""
    num_workers: int = 8
    """Number of workers for the parallel data loading (0 means sequential)."""
    batch_size: int = 50
    """Batch size."""
    atom_descriptors: Literal['feature', 'descriptor'] = None
    """
    Custom extra atom descriptors.
    :code:`feature`: used as atom features to featurize a given molecule.
    :code:`descriptor`: used as descriptor and concatenated to the machine learned atomic representation.
    """
    atom_descriptors_path: str = None
    """Path to the extra atom descriptors."""
    bond_descriptors: Literal['feature', 'descriptor'] = None
    """
    Custom extra bond descriptors.
    :code:`feature`: used as bond features to featurize a given molecule.
    :code:`descriptor`: used as descriptor and concatenated to the machine learned bond representation.
    """
    bond_descriptors_path: str = None
    """Path to the extra bond descriptors that will be used as bond features to featurize a given molecule."""
    no_cache_mol: bool = False
    """
    Whether to not cache the RDKit molecule for each SMILES string to reduce memory usage (cached by default).
    """
    empty_cache: bool = False
    """
    Whether to empty all caches before training or predicting. This is necessary if multiple jobs are run within a single script and the atom or bond features change.
    """
    constraints_path: str = None
    """
    Path to constraints applied to atomic/bond properties prediction.
    """

    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)
        self._atom_features_size = 0
        self._bond_features_size = 0
        self._atom_descriptors_size = 0
        self._bond_descriptors_size = 0
        self._atom_constraints = []
        self._bond_constraints = []

    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        """Whether to use CUDA (i.e., GPUs) or not."""
        return not self.no_cuda and torch.cuda.is_available()

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    @property
    def features_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional molecule-level features.
        """
        return not self.no_features_scaling

    @features_scaling.setter
    def features_scaling(self, features_scaling: bool) -> None:
        self.no_features_scaling = not features_scaling

    @property
    def atom_features_size(self) -> int:
        """The size of the atom features."""
        return self._atom_features_size

    @atom_features_size.setter
    def atom_features_size(self, atom_features_size: int) -> None:
        self._atom_features_size = atom_features_size

    @property
    def atom_descriptors_size(self) -> int:
        """The size of the atom descriptors."""
        return self._atom_descriptors_size

    @atom_descriptors_size.setter
    def atom_descriptors_size(self, atom_descriptors_size: int) -> None:
        self._atom_descriptors_size = atom_descriptors_size

    @property
    def bond_features_size(self) -> int:
        """The size of the atom features."""
        return self._bond_features_size

    @bond_features_size.setter
    def bond_features_size(self, bond_features_size: int) -> None:
        self._bond_features_size = bond_features_size

    @property
    def bond_descriptors_size(self) -> int:
        """The size of the bond descriptors."""
        return self._bond_descriptors_size

    @bond_descriptors_size.setter
    def bond_descriptors_size(self, bond_descriptors_size: int) -> None:
        self._bond_descriptors_size = bond_descriptors_size

    def configure(self) -> None:
        """Registers runtime-dependent command-line choices."""
        self.add_argument('--gpu', choices=list(range(torch.cuda.device_count())))
        self.add_argument('--features_generator', choices=get_available_features_generators())

    def _validate_common_numeric_args(self) -> None:
        """Validates sizes shared by data loading, training, and prediction."""
        for name, value, minimum in (
            ('batch_size', self.batch_size, 1),
            ('num_workers', self.num_workers, 0),
            ('number_of_molecules', self.number_of_molecules, 1),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f'{name} must be an integer of at least {minimum}.')
        if (
            self.max_data_size is not None
            and (
                not isinstance(self.max_data_size, int)
                or isinstance(self.max_data_size, bool)
                or self.max_data_size < 1
            )
        ):
            raise ValueError('max_data_size must be None or a positive integer.')
        if self.gpu is not None and (
            not isinstance(self.gpu, int)
            or isinstance(self.gpu, bool)
            or self.gpu not in range(torch.cuda.device_count())
        ):
            raise ValueError('gpu must identify an available CUDA device.')

    def process_args(self) -> None:
        self._validate_common_numeric_args()

        # LightGBM bundles use ``.pkl`` while neural-network checkpoints use
        # ``.pt``. Infer the model type when the supplied source is
        # unambiguous, so prediction from a LightGBM checkpoint directory does
        # not require a redundant ``--model_type lgbm`` flag.
        if hasattr(self, 'model_type') and self.model_type == 'FFN':
            checkpoint_extensions = set()
            if self.checkpoint_path is not None:
                checkpoint_extensions.add(os.path.splitext(self.checkpoint_path)[1].lower())
            elif self.checkpoint_paths is not None:
                checkpoint_extensions.update(
                    os.path.splitext(path)[1].lower() for path in self.checkpoint_paths
                )
            elif self.checkpoint_dir is not None:
                for _, _, files in os.walk(self.checkpoint_dir):
                    checkpoint_extensions.update(
                        os.path.splitext(filename)[1].lower()
                        for filename in files
                        if os.path.splitext(filename)[1].lower() in {'.pt', '.pkl'}
                    )
            if checkpoint_extensions == {'.pt', '.pkl'}:
                raise ValueError(
                    'Checkpoint sources mix FFN .pt files and LightGBM .pkl '
                    'bundles. Specify a directory or explicit path list '
                    'containing only one model type.'
                )
            if checkpoint_extensions == {'.pkl'}:
                self.model_type = 'lgbm'

        # Load checkpoint paths
        # Not every CommonArgs subclass has a model_type (for example,
        # InterpretArgs). Default those callers to the standard FFN path.
        if getattr(self, 'model_type', 'FFN') == 'lgbm':
            self.checkpoint_paths = get_checkpoint_paths(
                checkpoint_path=self.checkpoint_path,
                checkpoint_paths=self.checkpoint_paths,
                checkpoint_dir=self.checkpoint_dir,
                ext='.pkl',
            )
        else:
            self.checkpoint_paths = get_checkpoint_paths(
                checkpoint_path=self.checkpoint_path,
                checkpoint_paths=self.checkpoint_paths,
                checkpoint_dir=self.checkpoint_dir,
            )

        # Validate features
        if self.features_generator is not None and 'rdkit_2d_normalized' in self.features_generator and self.features_scaling:
            raise ValueError('When using rdkit_2d_normalized features, --no_features_scaling must be specified.')

        # Validate atom descriptors
        if (self.atom_descriptors is None) != (self.atom_descriptors_path is None):
            raise ValueError('If atom_descriptors is specified, then an atom_descriptors_path must be provided '
                             'and vice versa.')

        if self.atom_descriptors is not None and self.number_of_molecules > 1:
            raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                      'per input (i.e., number_of_molecules = 1).')

        # Validate bond descriptors
        if (self.bond_descriptors is None) != (self.bond_descriptors_path is None):
            raise ValueError('If bond_descriptors is specified, then an bond_descriptors_path must be provided '
                             'and vice versa.')

        if self.bond_descriptors is not None and self.number_of_molecules > 1:
            raise NotImplementedError('Bond descriptors are currently only supported with one molecule '
                                      'per input (i.e., number_of_molecules = 1).')

        set_cache_mol(not self.no_cache_mol)

        if self.empty_cache:
            empty_cache()


class TrainArgs(CommonArgs):
    """:class:`TrainArgs` includes :class:`CommonArgs` along with additional arguments used for training a Chemprop model."""

    # General arguments
    data_path: str
    """Path to data CSV file."""
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    By default, uses all columns except the SMILES column and the :code:`ignore_columns`.
    """
    ignore_columns: List[str] = None
    """Name of the columns to ignore when :code:`target_columns` is not provided."""
    dataset_type: Literal['regression', 'classification', 'multiclass', 'spectra']
    """Type of dataset. This determines the default loss function used during training."""
    model_type: Literal['FFN', 'lgbm'] = 'FFN'
    """Type of the prediction model."""
    loss_function: Literal['mse', 'bounded_mse', 'binary_cross_entropy', 'cross_entropy', 'mcc', 'sid', 'wasserstein', 'mve', 'evidential', 'dirichlet', 'quantile_interval'] = None
    """Choice of loss function. Loss functions are limited to compatible dataset types."""
    multiclass_num_classes: int = 3
    """Number of classes when running multiclass classification."""
    separate_val_path: str = None
    """Path to separate val set, optional."""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    spectra_phase_mask_path: str = None
    """Path to a file containing a phase mask array, used for excluding particular regions in spectra predictions."""
    data_weights_path: str = None
    """Path to weights for each molecule in the training data, affecting the relative weight of molecules in the loss function"""
    target_weights: List[float] = None
    """Weights associated with each target, affecting the relative weight of targets in the loss function. Must match the number of target columns."""
    split_type: Literal['random', 'scaffold_balanced', 'predetermined', 'crossval', 'cv', 'cv-no-test', 'index_predetermined', 'random_with_repeated_smiles', 'molecular_weight'] = 'random'
    """Method of splitting the data into train/val/test."""
    split_sizes: List[float] = None
    """Split proportions for train/validation/test sets."""
    split_key_molecule: int = 0
    """The index of the key molecule used for splitting when multiple molecules are present and constrained split_type is used, like scaffold_balanced or random_with_repeated_smiles.
       Note that this index begins with zero for the first molecule."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    folds_file: str = None
    """Optional file of fold labels."""
    val_fold_index: int = None
    """Which fold to use as val for leave-one-out cross val."""
    test_fold_index: int = None
    """Which fold to use as test for leave-one-out cross val."""
    crossval_index_dir: str = None
    """Directory in which to find cross validation index files."""
    crossval_index_file: str = None
    """Indices of files to use as train/val/test. Overrides :code:`--num_folds` and :code:`--seed`."""
    seed: int = 0
    """
    Random seed to use when splitting data into train/val/test sets.
    When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.
    """
    pytorch_seed: int = 0
    """Seed for PyTorch randomness (e.g., random initial weights)."""
    metric: Metric = None
    """
    Metric to use during evaluation. It is also used with the validation set for early stopping.
    Defaults to "auc" for classification, "rmse" for regression, and "sid" for spectra.
    """
    extra_metrics: List[Metric] = []
    """Additional metrics to use to evaluate the model. Not used for early stopping."""
    ignore_nan_metrics: bool = False
    """Ignore invalid task metrics (NaNs) when computing average metrics across tasks."""
    save_dir: str = None
    """Directory where model checkpoints will be saved."""
    checkpoint_frzn: str = None
    """Path to model checkpoint file to be loaded for overwriting and freezing weights."""
    save_smiles_splits: bool = False
    """Save smiles for each train/val/test splits for prediction convenience later."""
    test: bool = False
    """Whether to skip training and only test the model."""
    quiet: bool = False
    """Skip non-essential print statements."""
    log_frequency: int = 10
    """The number of batches between each logging of the training loss."""
    show_individual_scores: bool = False
    """Show all scores for individual targets, not just average, at the end."""
    cache_cutoff: float = 10000
    """
    Maximum number of molecules in dataset to allow caching.
    Below this number, caching is used and data loading is sequential.
    Above this number, caching is not used and data loading is parallel.
    Use "inf" to always cache.
    """
    save_preds: bool = False
    """Whether to save test split predictions during training."""
    resume_experiment: bool = False
    """
    Whether to resume the experiment.
    Loads test results from any folds that have already been completed and skips training those folds.
    """
    skip_test_evaluation: bool = False
    """Withhold test-set evaluation and artifacts. Enabled automatically for hyperparameter optimization."""

    # Model arguments
    bias: bool = False
    """Whether to add bias to linear layers."""
    hidden_size: int = 300
    """Dimensionality of hidden layers in MPN."""
    depth: int = 3
    """Number of message passing steps."""
    bias_solvent: bool = False
    """Whether to add bias to linear layers for solvent MPN if :code:`reaction_solvent` is True."""
    hidden_size_solvent: int = 300
    """Dimensionality of hidden layers in solvent MPN if :code:`reaction_solvent` is True."""
    depth_solvent: int = 3
    """Number of message passing steps for solvent if :code:`reaction_solvent` is True."""
    mpn_shared: bool = False
    """Whether to use the same message passing neural network for all input molecules
    Only relevant if :code:`number_of_molecules > 1`"""
    dropout: float = 0.0
    """Dropout probability."""
    activation: Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'
    """Activation function."""
    atom_messages: bool = False
    """Centers messages on atoms instead of on bonds."""
    undirected: bool = False
    """Undirected edges (always sum the two relevant bond vectors)."""
    ffn_hidden_size: int = None
    """Hidden dim for higher-capacity FFN (defaults to hidden_size)."""
    ffn_num_layers: int = 2
    """Number of layers in FFN after MPN encoding."""
    features_only: bool = False
    """Use only the additional features in an FFN, no graph network."""
    separate_val_features_path: List[str] = None
    """Path to file with features for separate val set."""
    separate_test_features_path: List[str] = None
    """Path to file with features for separate test set."""
    separate_val_phase_features_path: str = None
    """Path to file with phase features for separate val set."""
    separate_test_phase_features_path: str = None
    """Path to file with phase features for separate test set."""
    separate_val_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_atom_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate test set."""
    separate_val_bond_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate val set."""
    separate_test_bond_descriptors_path: str = None
    """Path to file with extra atom descriptors for separate test set."""
    separate_val_constraints_path: str = None
    """Path to file with constraints for separate val set."""
    separate_test_constraints_path: str = None
    """Path to file with constraints for separate test set."""
    config_path: str = None
    """
    Path to a :code:`.json` file containing arguments. Any arguments present in the config file
    will override arguments specified via the command line or by the defaults.
    """
    ensemble_size: int = 1
    """Number of models in ensemble."""
    lgbm_num_boost_round: int = 500
    """Maximum number of boosting rounds for each LightGBM task head."""
    lgbm_early_stopping_rounds: int = 30
    """Validation rounds without improvement before stopping; 0 disables early stopping."""
    lgbm_learning_rate: float = 0.05
    """LightGBM shrinkage rate."""
    lgbm_num_leaves: int = 31
    """Maximum leaves in each LightGBM tree."""
    lgbm_feature_fraction: float = 0.8
    """Fraction of encoded features sampled for each LightGBM model."""
    lgbm_bagging_fraction: float = 0.8
    """Fraction of training rows sampled by LightGBM bagging."""
    lgbm_bagging_freq: int = 1
    """Boosting-iteration frequency for LightGBM bagging; 0 disables it."""
    lgbm_min_data_in_leaf: int = 20
    """Minimum number of training rows in a LightGBM leaf."""
    lgbm_num_threads: int = None
    """LightGBM CPU threads; defaults to max(1, --num_workers)."""
    aggregation: Literal['sum', 'norm', 'mean'] = 'mean'
    """Aggregation scheme for atomic vectors into molecular vectors"""
    aggregation_norm: int = 100
    """For norm aggregation, number by which to divide summed up atomic features"""
    reaction: bool = False
    """
    Whether to adjust MPNN layer to take reactions as input instead of molecules.
    """
    reaction_mode: Literal['reac_prod', 'reac_diff', 'prod_diff', 'reac_prod_balance', 'reac_diff_balance', 'prod_diff_balance'] = 'reac_diff'
    """
    Choices for construction of atom and bond features for reactions
    :code:`reac_prod`: concatenates the reactants feature with the products feature.
    :code:`reac_diff`: concatenates the reactants feature with the difference in features between reactants and products.
    :code:`prod_diff`: concatenates the products feature with the difference in features between reactants and products.
    :code:`reac_prod_balance`: concatenates the reactants feature with the products feature, balances imbalanced reactions.
    :code:`reac_diff_balance`: concatenates the reactants feature with the difference in features between reactants and products, balances imbalanced reactions.
    :code:`prod_diff_balance`: concatenates the products feature with the difference in features between reactants and products, balances imbalanced reactions.
    """
    reaction_solvent: bool = False
    """
    Whether to adjust the MPNN layer to take as input a reaction and a molecule, and to encode them with separate MPNNs.
    """
    explicit_h: bool = False
    """
    Whether H are explicitly specified in input (and should be kept this way). This option is intended to be used
    with the :code:`reaction` or :code:`reaction_solvent` options, and applies only to the reaction part.
    """
    adding_h: bool = False
    """
    Whether RDKit molecules will be constructed with adding the Hs to them. This option is intended to be used
    with Chemprop's default molecule or multi-molecule encoders, or in :code:`reaction_solvent` mode where it applies to the solvent only.
    """
    is_atom_bond_targets: bool = False
    """
    whether this is atomic/bond properties prediction.
    """
    keeping_atom_map: bool = False
    """
    Whether RDKit molecules keep the original atom mapping. This option is intended to be used when providing atom-mapped SMILES with
    the :code:`is_atom_bond_targets`.
    """
    no_shared_atom_bond_ffn: bool = False
    """
    Whether the FFN weights for atom and bond targets should be independent between tasks.
    """
    weights_ffn_num_layers: int = 2
    """
    Number of layers in FFN for determining weights used in constrained targets.
    """
    no_adding_bond_types: bool = False
    """
    Whether the bond types determined by RDKit molecules added to the output of bond targets. This option is intended to be used
    with the :code:`is_atom_bond_targets`.
    """

    # Training arguments
    epochs: int = 30
    """Number of epochs to run."""
    warmup_epochs: float = 2.0
    """
    Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
    """
    init_lr: float = 1e-4
    """Initial learning rate."""
    max_lr: float = 1e-3
    """Maximum learning rate."""
    final_lr: float = 1e-4
    """Final learning rate."""
    grad_clip: float = None
    """Maximum magnitude of gradient during training."""
    class_balance: bool = False
    """Trains with an equal number of positives and negatives in each batch."""
    spectra_activation: Literal['exp', 'softplus'] = 'exp'
    """Indicates which function to use in dataset_type spectra training to constrain outputs to be positive."""
    spectra_target_floor: float = 1e-8
    """Values in targets for dataset type spectra are replaced with this value, intended to be a small positive number used to enforce positive values."""
    evidential_regularization: float = 0
    """Value used in regularization for evidential loss function. The default value recommended by Soleimany et al.(2021) is 0.2. 
    Optimal value is dataset-dependent; it is recommended that users test different values to find the best value for their model."""
    quantile_loss_alpha: float = 0.1
    """Target error bounds for quantile interval loss"""
    overwrite_default_atom_features: bool = False
    """
    Overwrites the default atom descriptors with the new ones instead of concatenating them.
    Can only be used if atom_descriptors are used as a feature.
    """
    no_atom_descriptor_scaling: bool = False
    """Turn off atom feature scaling."""
    overwrite_default_bond_features: bool = False
    """
    Overwrites the default bond descriptors with the new ones instead of concatenating them.
    Can only be used if bond_descriptors are used as a feature.
    """
    no_bond_descriptor_scaling: bool = False
    """Turn off atom feature scaling."""
    frzn_ffn_layers: int = 0
    """
    Overwrites weights for the first n layers of the ffn from checkpoint model (specified checkpoint_frzn),
    where n is specified in the input.
    Automatically also freezes mpnn weights.
    """
    freeze_first_only: bool = False
    """
    Determines whether or not to use checkpoint_frzn for just the first encoder.
    Default (False) is to use the checkpoint to freeze all encoders.
    (only relevant for number_of_molecules > 1, where checkpoint model has number_of_molecules = 1)
    """

    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        # The annotated class default is a mutable list. Keep programmatic
        # changes to one training job from leaking into later args instances.
        self.extra_metrics = []
        self._temp_save_dir = None
        self._task_names = None
        self._crossval_index_sets = None
        self._task_names = None
        self._quantiles = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None

    @property
    def metrics(self) -> List[str]:
        """The list of metrics used for evaluation. Only the first is used for early stopping."""
        return [self.metric] + self.extra_metrics

    @property
    def minimize_score(self) -> bool:
        """Whether the model should try to minimize the score metric or maximize it."""
        return self.metric in {'rmse', 'mae', 'mse', 'cross_entropy', 'binary_cross_entropy', 'sid', 'wasserstein', 'bounded_mse', 'bounded_mae', 'bounded_rmse', 'quantile'}

    @property
    def use_input_features(self) -> bool:
        """Whether the model is using additional molecule-level features."""
        return self.features_generator is not None or self.features_path is not None or self.phase_features_path is not None

    @property
    def num_lrs(self) -> int:
        """The number of learning rates to use (currently hard-coded to 1)."""
        return 1

    @property
    def crossval_index_sets(self) -> List[List[List[int]]]:
        """Index sets used for splitting data into train/validation/test during cross-validation"""
        return self._crossval_index_sets

    @property
    def task_names(self) -> List[str]:
        """A list of names of the tasks being trained on."""
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        """The number of tasks being trained on."""
        return len(self.task_names) if self.task_names is not None else 0

    @property
    def quantiles(self) -> List[float]:
        """A list of quantiles to be being trained on."""
        return self._quantiles

    @quantiles.setter
    def quantiles(self, quantiles: List[float]) -> None:
        self._quantiles = quantiles

    @property
    def features_size(self) -> int:
        """The dimensionality of the additional molecule-level features."""
        return self._features_size

    @features_size.setter
    def features_size(self, features_size: int) -> None:
        self._features_size = features_size

    @property
    def train_data_size(self) -> int:
        """The size of the training data set."""
        return self._train_data_size

    @train_data_size.setter
    def train_data_size(self, train_data_size: int) -> None:
        self._train_data_size = train_data_size

    @property
    def atom_descriptor_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional atom features."
        """
        return not self.no_atom_descriptor_scaling

    @property
    def bond_descriptor_scaling(self) -> bool:
        """
        Whether to apply normalization with a :class:`~chemprop.data.scaler.StandardScaler`
        to the additional bond features."
        """
        return not self.no_bond_descriptor_scaling
    
    @property
    def shared_atom_bond_ffn(self) -> bool:
        """
        Whether the FFN weights for atom and bond targets should be shared between tasks.
        """
        return not self.no_shared_atom_bond_ffn

    @property
    def adding_bond_types(self) -> bool:
        """
        Whether the bond types determined by RDKit molecules should be added to the output of bond targets.
        """
        return not self.no_adding_bond_types

    @property
    def atom_constraints(self) -> List[bool]:
        """
        A list of booleans indicating whether constraints applied to output of atomic properties.
        """
        if self.is_atom_bond_targets and self.constraints_path:
            if not self._atom_constraints:
                header = chemprop.data.utils.get_header(self.constraints_path)
                self._atom_constraints = [target in header for target in self.atom_targets]
        else:
            self._atom_constraints = [False] * len(self.atom_targets)
        return self._atom_constraints

    @atom_constraints.setter
    def atom_constraints(self, atom_constraints: List[bool]) -> None:
        self._atom_constraints = atom_constraints

    @property
    def bond_constraints(self) -> List[bool]:
        """
        A list of booleans indicating whether constraints applied to output of bond properties.
        """
        if self.is_atom_bond_targets and self.constraints_path:
            if not self._bond_constraints:
                header = chemprop.data.utils.get_header(self.constraints_path)
                self._bond_constraints = [target in header for target in self.bond_targets]
        else:
            self._bond_constraints = [False] * len(self.bond_targets)
        return self._bond_constraints

    @bond_constraints.setter
    def bond_constraints(self, bond_constraints: List[bool]) -> None:
        self._bond_constraints = bond_constraints

    def _load_config_overrides(self) -> None:
        """Loads validated JSON overrides before deriving any argument state."""
        if self.config_path is None:
            return

        original_config_path = self.config_path
        with open(original_config_path, encoding='utf-8') as config_file:
            config = json.load(config_file)
        if not isinstance(config, dict):
            raise ValueError('config_path must contain a JSON object of argument overrides.')

        annotations = {}
        # Start with base classes so a subclass annotation wins if a name is
        # deliberately redefined.
        for cls in reversed(type(self).mro()):
            annotations.update(getattr(cls, '__annotations__', {}))
        allowed_keys = set(annotations)
        forbidden_keys = {
            key for key in config
            if (
                not isinstance(key, str)
                or key.startswith('_')
                or key == 'config_path'
                or key not in allowed_keys
            )
        }
        if forbidden_keys:
            raise ValueError(
                'Unknown or unsafe config key(s): '
                f'{", ".join(sorted(map(str, forbidden_keys)))}.'
            )

        available_generators = set(get_available_features_generators())
        for key, value in config.items():
            class_default = getattr(type(self), key, object())
            if value is None:
                if class_default is not None:
                    raise ValueError(f'Config value for "{key}" cannot be null.')
            elif not _config_value_matches_annotation(value, annotations[key]):
                raise ValueError(
                    f'Invalid config value for "{key}": expected {annotations[key]!r}, '
                    f'received {value!r}.'
                )
            if key == 'features_generator' and value is not None:
                unknown_generators = sorted(set(value) - available_generators)
                if unknown_generators:
                    raise ValueError(
                        'Unknown feature generator(s) in config: '
                        f'{", ".join(unknown_generators)}.'
                    )
            setattr(self, key, value)

    def _validate_training_numeric_args(self) -> None:
        """Rejects numeric settings which would fail or degenerate at runtime."""
        ffn_hidden_size = self.hidden_size if self.ffn_hidden_size is None else self.ffn_hidden_size
        for name, value in (
            ('hidden_size', self.hidden_size),
            ('depth', self.depth),
            ('hidden_size_solvent', self.hidden_size_solvent),
            ('depth_solvent', self.depth_solvent),
            ('ffn_hidden_size', ffn_hidden_size),
            ('ffn_num_layers', self.ffn_num_layers),
            ('weights_ffn_num_layers', self.weights_ffn_num_layers),
            ('log_frequency', self.log_frequency),
            ('num_folds', self.num_folds),
        ):
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')

        for name, value in (
            ('epochs', self.epochs),
            ('early_stopping', self.early_stopping),
            ('frzn_ffn_layers', self.frzn_ffn_layers),
        ):
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 0:
                raise ValueError(f'{name} must be a non-negative integer.')

        for name, value in (('seed', self.seed), ('pytorch_seed', self.pytorch_seed)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f'{name} must be an integer.')

        if (
            not isinstance(self.multiclass_num_classes, (int, np.integer))
            or isinstance(self.multiclass_num_classes, (bool, np.bool_))
            or self.multiclass_num_classes < 2
        ):
            raise ValueError('multiclass_num_classes must be an integer of at least 2.')

        if not _is_finite_number(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError('dropout must be finite and in the range [0, 1).')
        if not _is_finite_number(self.aggregation_norm) or self.aggregation_norm <= 0:
            raise ValueError('aggregation_norm must be finite and greater than 0.')
        if not _is_finite_number(self.warmup_epochs) or self.warmup_epochs < 0:
            raise ValueError('warmup_epochs must be finite and non-negative.')

        for name, value in (
            ('init_lr', self.init_lr),
            ('max_lr', self.max_lr),
            ('final_lr', self.final_lr),
        ):
            if not _is_finite_number(value) or value <= 0:
                raise ValueError(f'{name} must be finite and greater than 0.')
        if self.max_lr < self.init_lr or self.max_lr < self.final_lr:
            raise ValueError('max_lr must be greater than or equal to init_lr and final_lr.')

        if self.grad_clip is not None and (
            not _is_finite_number(self.grad_clip) or self.grad_clip <= 0
        ):
            raise ValueError('grad_clip must be None or a finite number greater than 0.')
        if (
            not _is_finite_number(self.cache_cutoff)
            and self.cache_cutoff != float('inf')
        ) or self.cache_cutoff < 0:
            raise ValueError('cache_cutoff must be a non-negative finite number or positive infinity.')
        if (
            not _is_finite_number(self.evidential_regularization)
            or self.evidential_regularization < 0
        ):
            raise ValueError('evidential_regularization must be finite and non-negative.')
        if (
            not _is_finite_number(self.quantile_loss_alpha)
            or not 0 <= self.quantile_loss_alpha <= 0.5
        ):
            raise ValueError('quantile_loss_alpha must be finite and in the range [0, 0.5].')

        if (
            not isinstance(self.split_key_molecule, (int, np.integer))
            or isinstance(self.split_key_molecule, (bool, np.bool_))
            or self.split_key_molecule < 0
        ):
            raise ValueError('split_key_molecule must be a non-negative integer.')
        for name, value in (
            ('val_fold_index', self.val_fold_index),
            ('test_fold_index', self.test_fold_index),
        ):
            if value is not None and (
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                or value < 0
            ):
                raise ValueError(f'{name} must be None or a non-negative integer.')

    def process_args(self) -> None:
        # Config values must be applied before CommonArgs resolves checkpoint
        # paths, validates feature combinations, or changes global caches, and
        # before SMILES columns are derived from the selected data file.
        self._load_config_overrides()
        super(TrainArgs, self).process_args()

        self._validate_training_numeric_args()
        if self.test and not self.checkpoint_paths:
            raise ValueError(
                '--test skips optimization and therefore requires an existing '
                '--checkpoint_path, --checkpoint_paths, or --checkpoint_dir.'
            )

        # Adapt the number of molecules for reaction_solvent mode
        if self.reaction_solvent is True and self.number_of_molecules != 2:
            raise ValueError('In reaction_solvent mode, --number_of_molecules 2 must be specified.')

        # Process SMILES columns
        self.smiles_columns = chemprop.data.utils.preprocess_smiles_columns(
            path=self.data_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )

        # Determine the target_columns when training atomic and bond targets
        if self.is_atom_bond_targets:
            self.atom_targets, self.bond_targets, self.molecule_targets = chemprop.data.utils.get_mixed_task_names(
                path=self.data_path,
                smiles_columns=self.smiles_columns,
                target_columns=self.target_columns,
                ignore_columns=self.ignore_columns,
                keep_h=self.explicit_h,
                add_h=self.adding_h,
                keep_atom_map=self.keeping_atom_map,
            )
            self.target_columns = self.atom_targets + self.bond_targets
            # self.target_columns = self.atom_targets + self.bond_targets + self.molecule_targets  # TODO: Support mixed targets
        else:
            self.atom_targets, self.bond_targets = [], []

        # Check whether atomic/bond constraints have been applied on the correct dataset_type
        if self.constraints_path:
            if not self.is_atom_bond_targets:
                raise ValueError('Constraints on atomic/bond targets can only be used in atomic/bond properties prediction.')
            if self.dataset_type != 'regression':
                raise ValueError(f'In atomic/bond properties prediction, atomic/bond constraints are not supported for {self.dataset_type}.')

        # Check whether the number of input columns is one for the atomic/bond mode
        if self.is_atom_bond_targets:
            if self.number_of_molecules != 1:
                raise ValueError('In atomic/bond properties prediction, exactly one smiles column must be provided.')

        # Check whether the number of input columns is two for the reaction_solvent mode
        if self.reaction_solvent is True and len(self.smiles_columns) != 2:
            raise ValueError('In reaction_solvent mode, exactly two smiles column must be provided (one for reactions, and one for molecules)')

        # Validate reaction/reaction_solvent mode
        if self.reaction is True and self.reaction_solvent is True:
            raise ValueError('Only reaction or reaction_solvent mode can be used, not both.')

        # Create temporary directory as save directory if not provided
        if self.save_dir is None:
            self._temp_save_dir = TemporaryDirectory()
            self.save_dir = self._temp_save_dir.name

        # Fix ensemble size if loading checkpoints
        if self.checkpoint_paths is not None and len(self.checkpoint_paths) > 0:
            self.ensemble_size = len(self.checkpoint_paths)

        # Process and validate metric and loss function
        if self.metric is None:
            if self.dataset_type == "classification":
                self.metric = "auc"
            elif self.dataset_type == "multiclass":
                self.metric = "cross_entropy"
            elif self.dataset_type == "spectra":
                self.metric = "sid"
            elif self.dataset_type == "regression" and self.loss_function == "bounded_mse":
                self.metric = "bounded_mse"
            elif self.dataset_type == "regression" and self.loss_function == "quantile_interval":
                self.metric = "quantile"
            elif self.dataset_type == "regression":
                self.metric = "rmse"
            else:
                raise ValueError(f'Dataset type {self.dataset_type} is not supported.')

        if self.metric in self.extra_metrics:
            raise ValueError(f'Metric {self.metric} is both the metric and is in extra_metrics. '
                             f'Please only include it once.')

        for metric in self.metrics:
            if not any([(self.dataset_type == 'classification' and metric in ['auc', 'prc-auc', 'accuracy', 'binary_cross_entropy', 'f1', 'mcc', 'recall', 'precision', 'balanced_accuracy']),
                        (self.dataset_type == 'regression' and metric in ['rmse', 'mae', 'mse', 'r2', 'bounded_rmse', 'bounded_mae', 'bounded_mse', 'quantile']),
                        (self.dataset_type == 'multiclass' and metric in ['cross_entropy', 'accuracy', 'f1', 'mcc']),
                        (self.dataset_type == 'spectra' and metric in ['sid', 'wasserstein'])]):
                raise ValueError(f'Metric "{metric}" invalid for dataset type "{self.dataset_type}".')

            if metric == "quantile" and self.loss_function != "quantile_interval":
                raise ValueError('Metric quantile is only compatible with quantile_interval loss.')

        if self.loss_function is None:
            if self.dataset_type == 'classification':
                self.loss_function = 'binary_cross_entropy'
            elif self.dataset_type == 'multiclass':
                self.loss_function = 'cross_entropy'
            elif self.dataset_type == 'spectra':
                self.loss_function = 'sid'
            elif self.dataset_type == 'regression':
                self.loss_function = 'mse'
            else:
                raise ValueError(f'Default loss function not configured for dataset type {self.dataset_type}.')

        if self.loss_function != 'bounded_mse' and any(metric in ['bounded_mse', 'bounded_rmse', 'bounded_mae'] for metric in self.metrics):
            raise ValueError('Bounded metrics can only be used in conjunction with the regression loss function bounded_mse.')

        if self.dataset_type == 'spectra' and (
            not np.isfinite(self.spectra_target_floor)
            or self.spectra_target_floor <= 0
        ):
            raise ValueError(
                'spectra_target_floor must be a finite number greater than 0.'
            )
        if self.spectra_phase_mask_path is not None and self.phase_features_path is None:
            raise ValueError(
                'spectra_phase_mask_path requires phase_features_path so each '
                'spectrum has a phase assignment.'
            )

        # Validate class balance
        if self.class_balance and self.dataset_type != 'classification':
            raise ValueError('Class balance can only be applied if the dataset type is classification.')

        # Validate features
        if self.features_only and not (self.features_generator or self.features_path):
            raise ValueError('When using features_only, a features_generator or features_path must be provided.')

        # Handle FFN hidden size
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size

        # Handle MPN variants
        if self.atom_messages and self.undirected:
            raise ValueError('Undirected is unnecessary when using atom_messages '
                             'since atom_messages are by their nature undirected.')

        # Validate split type settings
        if not (self.split_type == 'predetermined') == (self.folds_file is not None) == (self.test_fold_index is not None):
            raise ValueError('When using predetermined split type, must provide folds_file and test_fold_index.')

        if not (self.split_type == 'crossval') == (self.crossval_index_dir is not None):
            raise ValueError('When using crossval split type, must provide crossval_index_dir.')

        if not (self.split_type in ['crossval', 'index_predetermined']) == (self.crossval_index_file is not None):
            raise ValueError('When using crossval or index_predetermined split type, must provide crossval_index_file.')

        if self.split_type in ['crossval', 'index_predetermined']:
            with open(self.crossval_index_file, 'rb') as rf:
                self._crossval_index_sets = pickle.load(rf)
            self.num_folds = len(self.crossval_index_sets)
            if self.num_folds < 1:
                raise ValueError('crossval_index_file must contain at least one fold.')
            self.seed = 0

        # Validate split size entry and set default values
        if self.split_sizes is not None:
            try:
                split_sizes = np.asarray(self.split_sizes, dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError('split_sizes must contain only numeric values.') from error
            if split_sizes.ndim != 1 or split_sizes.size not in {2, 3}:
                raise ValueError('split_sizes must contain two or three values.')
            if not np.all(np.isfinite(split_sizes)):
                raise ValueError('split_sizes must contain only finite values.')
            self.split_sizes = split_sizes.tolist()

        if self.split_sizes is None:
            if self.separate_val_path is None and self.separate_test_path is None: # separate data paths are not provided
                self.split_sizes = [0.8, 0.1, 0.1]
            elif self.separate_val_path is not None and self.separate_test_path is None: # separate val path only
                self.split_sizes = [0.8, 0., 0.2]
            elif self.separate_val_path is None and self.separate_test_path is not None: # separate test path only
                self.split_sizes = [0.8, 0.2, 0.]
            else: # both separate data paths are provided
                self.split_sizes = [1., 0., 0.]

        else:
            if not np.isclose(sum(self.split_sizes), 1):
                raise ValueError(f'Provided split sizes of {self.split_sizes} do not sum to 1.')
            if any([size < 0 for size in self.split_sizes]):
                raise ValueError(f'Split sizes must be non-negative. Received split sizes: {self.split_sizes}')


            if len(self.split_sizes) not in [2, 3]:
                raise ValueError(f'Three values should be provided for train/val/test split sizes. Instead received {len(self.split_sizes)} value(s).')

            if self.separate_val_path is None and self.separate_test_path is None:  # separate data paths are not provided
                if len(self.split_sizes) != 3:
                    raise ValueError(f'Three values should be provided for train/val/test split sizes. Instead received {len(self.split_sizes)} value(s).')
                if self.split_sizes[0] == 0.:
                    raise ValueError(f'Provided split size for train split must be nonzero. Received split size {self.split_sizes[0]}')
                if self.split_sizes[1] == 0.:
                    raise ValueError(f'Provided split size for validation split must be nonzero. Received split size {self.split_sizes[1]}')

            elif self.separate_val_path is not None and self.separate_test_path is None: # separate val path only
                if len(self.split_sizes) == 2: # allow input of just 2 values
                    self.split_sizes = [self.split_sizes[0], 0., self.split_sizes[1]]
                if self.split_sizes[0] == 0.:
                    raise ValueError('Provided split size for train split must be nonzero.')
                if self.split_sizes[1] != 0.:
                    raise ValueError(f'Provided split size for validation split must be 0 because validation set is provided separately. Received split size {self.split_sizes[1]}')

            elif self.separate_val_path is None and self.separate_test_path is not None: # separate test path only
                if len(self.split_sizes) == 2: # allow input of just 2 values
                    self.split_sizes = [self.split_sizes[0], self.split_sizes[1], 0.]
                if self.split_sizes[0] == 0.:
                    raise ValueError('Provided split size for train split must be nonzero.')
                if self.split_sizes[1] == 0.:
                    raise ValueError('Provided split size for validation split must be nonzero.')
                if self.split_sizes[2] != 0.:
                    raise ValueError(f'Provided split size for test split must be 0 because test set is provided separately. Received split size {self.split_sizes[2]}')


            else: # both separate data paths are provided
                if self.split_sizes != [1., 0., 0.]:
                    raise ValueError(f'Separate data paths were provided for val and test splits. Split sizes should not also be provided. Received split sizes: {self.split_sizes}')

        # Test settings
        if self.test:
            self.epochs = 0

        # Validate features are provided for separate validation or test set for each of the kinds of additional features
        for (features_argument, base_features_path, val_features_path, test_features_path) in [
            ('`--features_path`', self.features_path, self.separate_val_features_path, self.separate_test_features_path),
            ('`--phase_features_path`', self.phase_features_path, self.separate_val_phase_features_path, self.separate_test_phase_features_path),
            ('`--atom_descriptors_path`', self.atom_descriptors_path, self.separate_val_atom_descriptors_path, self.separate_test_atom_descriptors_path),
            ('`--bond_descriptors_path`', self.bond_descriptors_path, self.separate_val_bond_descriptors_path, self.separate_test_bond_descriptors_path),
            ('`--constraints_path`', self.constraints_path, self.separate_val_constraints_path, self.separate_test_constraints_path)
        ]:
            if base_features_path is not None:
                if self.separate_val_path is not None and val_features_path is None:
                    raise ValueError(f'Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the separate validation set.')
                if self.separate_test_path is not None and test_features_path is None:
                    raise ValueError(f'Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the separate test set.')

        # validate extra atom descriptor options
        if self.overwrite_default_atom_features and self.atom_descriptors != 'feature':
            raise NotImplementedError('Overwriting of the default atom descriptors can only be used if the'
                                      'provided atom descriptors are features.')

        if not self.atom_descriptor_scaling and self.atom_descriptors is None:
            raise ValueError('Atom descriptor scaling is only possible if additional atom features are provided.')

        # validate extra bond descriptor options
        if self.overwrite_default_bond_features and self.bond_descriptors != 'feature':
            raise NotImplementedError('Overwriting of the default bond descriptors can only be used if the'
                                      'provided bond descriptors are features.')

        if not self.bond_descriptor_scaling and self.bond_descriptors is None:
            raise ValueError('Bond descriptor scaling is only possible if additional bond features are provided.')

        if self.bond_descriptors == 'descriptor' and not self.is_atom_bond_targets:
            raise NotImplementedError('Bond descriptors as descriptor can only be used with `--is_atom_bond_targets`.')

        # normalize target weights
        if self.target_weights is not None:
            target_weights = np.asarray(self.target_weights, dtype=float)
            if target_weights.ndim != 1 or target_weights.size == 0:
                raise ValueError('At least one target weight must be provided.')
            if not np.all(np.isfinite(target_weights)):
                raise ValueError('Provided target weights must be finite.')
            if np.any(target_weights < 0):
                raise ValueError('Provided target weights must be non-negative.')
            if float(target_weights.sum()) <= 0:
                raise ValueError('At least one target weight must be positive.')
            target_weights /= float(target_weights.mean())
            self.target_weights = target_weights.tolist()

        # check if key molecule index is outside of the number of molecules
        if self.split_key_molecule >= self.number_of_molecules:
            raise ValueError(
                "The index provided with the argument `--split_key_molecule` must be less than the number of molecules. Note that this index begins with 0 for the first molecule. "
            )

        if (
            not isinstance(self.ensemble_size, (int, np.integer))
            or isinstance(self.ensemble_size, (bool, np.bool_))
            or self.ensemble_size < 1
        ):
            raise ValueError('ensemble_size must be a positive integer.')

        if self.model_type == 'lgbm':
            if self.dataset_type not in {'classification', 'regression'}:
                raise ValueError('LightGBM supports only classification and regression datasets.')
            if not self.features_only:
                raise ValueError(
                    'LightGBM requires --features_only because its MPN encoder is '
                    'not trained. Provide deterministic molecular features, for '
                    'example --features_generator morgan --features_only, or use '
                    '--features_path together with --features_only.'
                )
            if self.is_atom_bond_targets:
                raise NotImplementedError('LightGBM does not support atom/bond target mode.')
            supported_primary_metrics = {
                'classification': {'auc', 'prc-auc', 'binary_cross_entropy'},
                'regression': {'rmse', 'mae', 'mse'},
            }[self.dataset_type]
            if self.metric not in supported_primary_metrics:
                raise NotImplementedError(
                    f'LightGBM cannot use --metric {self.metric} for early stopping. '
                    f'Supported primary metrics are '
                    f'{", ".join(sorted(supported_primary_metrics))}; other metrics '
                    'may still be requested with --extra_metrics for post-training evaluation.'
                )
            if self.target_weights is not None:
                raise NotImplementedError(
                    'LightGBM does not support --target_weights because each target '
                    'is trained by an independent booster.'
                )
            supported_loss = {
                'regression': 'mse',
                'classification': 'binary_cross_entropy',
            }[self.dataset_type]
            if self.loss_function != supported_loss:
                raise ValueError(
                    f'LightGBM {self.dataset_type} supports only '
                    f'--loss_function {supported_loss}; received {self.loss_function}.'
                )
            if self.checkpoint_paths is not None or self.checkpoint_frzn is not None:
                raise NotImplementedError(
                    'LightGBM warm-start/frozen checkpoints are not supported. '
                    'Train a new bundle without checkpoint arguments.'
                )
            if self.test:
                raise NotImplementedError(
                    'LightGBM --test mode is not supported; use chemprop_predict '
                    'with the saved .pkl bundle.'
                )
            if (
                not isinstance(self.lgbm_num_boost_round, int)
                or isinstance(self.lgbm_num_boost_round, bool)
                or self.lgbm_num_boost_round < 1
            ):
                raise ValueError('lgbm_num_boost_round must be a positive integer.')
            if (
                not isinstance(self.lgbm_early_stopping_rounds, int)
                or isinstance(self.lgbm_early_stopping_rounds, bool)
                or self.lgbm_early_stopping_rounds < 0
            ):
                raise ValueError('lgbm_early_stopping_rounds must be non-negative.')
            if not _is_finite_number(self.lgbm_learning_rate) or self.lgbm_learning_rate <= 0:
                raise ValueError('lgbm_learning_rate must be finite and greater than 0.')
            if (
                not isinstance(self.lgbm_num_leaves, int)
                or isinstance(self.lgbm_num_leaves, bool)
                or self.lgbm_num_leaves < 2
            ):
                raise ValueError('lgbm_num_leaves must be at least 2.')
            if (
                not _is_finite_number(self.lgbm_feature_fraction)
                or not 0 < self.lgbm_feature_fraction <= 1
            ):
                raise ValueError('lgbm_feature_fraction must be finite and in (0, 1].')
            if (
                not _is_finite_number(self.lgbm_bagging_fraction)
                or not 0 < self.lgbm_bagging_fraction <= 1
            ):
                raise ValueError('lgbm_bagging_fraction must be finite and in (0, 1].')
            if (
                not isinstance(self.lgbm_bagging_freq, int)
                or isinstance(self.lgbm_bagging_freq, bool)
                or self.lgbm_bagging_freq < 0
            ):
                raise ValueError('lgbm_bagging_freq must be non-negative.')
            if (
                not isinstance(self.lgbm_min_data_in_leaf, int)
                or isinstance(self.lgbm_min_data_in_leaf, bool)
                or self.lgbm_min_data_in_leaf < 1
            ):
                raise ValueError('lgbm_min_data_in_leaf must be positive.')
            if self.lgbm_num_threads is None:
                self.lgbm_num_threads = max(1, self.num_workers)
            elif (
                not isinstance(self.lgbm_num_threads, int)
                or isinstance(self.lgbm_num_threads, bool)
                or self.lgbm_num_threads < 1
            ):
                raise ValueError('lgbm_num_threads must be positive.')


class PredictArgs(CommonArgs):
    """:class:`PredictArgs` includes :class:`CommonArgs` along with additional arguments used for predicting with a Chemprop model."""

    test_path: str
    """Path to CSV file containing testing data for which predictions will be made."""
    preds_path: str
    """Path to CSV or PICKLE file where predictions will be saved."""
    model_type: Literal['FFN', 'lgbm'] = 'FFN'
    """Type of the prediction model."""
    drop_extra_columns: bool = False
    """Whether to drop all columns from the test data file besides the SMILES columns and the new prediction columns."""
    ensemble_variance: bool = False
    """Deprecated. Whether to calculate the variance of ensembles as a measure of epistemic uncertainty. If True, the variance is saved as an additional column for each target in the preds_path."""
    individual_ensemble_predictions: bool = False
    """Whether to return the predictions made by each of the individual models rather than the average of the ensemble"""
    # Uncertainty arguments
    uncertainty_method: Literal[
        'mve',
        'ensemble',
        'evidential_epistemic',
        'evidential_aleatoric',
        'evidential_total',
        'classification',
        'dropout',
        'spectra_roundrobin',
        'dirichlet',
    ] = None
    """The method of calculating uncertainty."""
    calibration_method: Literal[
        "zscaling",
        "tscaling",
        "zelikman_interval",
        "mve_weighting",
        "platt",
        "isotonic",
        "conformal",
        "conformal_adaptive",
        "conformal_regression",
        "conformal_quantile_regression",
    ] = None
    """Methods used for calibrating the uncertainty calculated with uncertainty method."""
    evaluation_methods: List[str] = None
    """The methods used for evaluating the uncertainty performance if the test data provided includes targets.
    Available methods are [nll, miscalibration_area, ence, spearman] or any available classification or multiclass metric."""
    evaluation_scores_path: str = None
    """Location to save the results of uncertainty evaluations."""
    uncertainty_dropout_p: float = 0.1
    """The probability to use for Monte Carlo dropout uncertainty estimation."""
    conformal_alpha: float = 0.1
    """Target error rate for conformal prediction."""
    dropout_sampling_size: int = 10
    """The number of samples to use for Monte Carlo dropout uncertainty estimation. Distinct from the dropout used during training."""
    calibration_interval_percentile: float = 95
    """Sets the percentile used in the calibration methods. Must be in the range (1,100)."""
    regression_calibrator_metric: Literal['stdev', 'interval'] = None
    """Regression calibrators can output either a stdev or an inverval. """
    calibration_path: str = None
    """Path to data file to be used for uncertainty calibration."""
    calibration_features_path: List[str] = None
    """Path to features data to be used with the uncertainty calibration dataset."""
    calibration_phase_features_path: str = None
    """ """
    calibration_atom_descriptors_path: str = None
    """Path to the extra atom descriptors."""
    calibration_bond_descriptors_path: str = None
    """Path to the extra bond descriptors that will be used as bond features to featurize a given molecule."""

    @property
    def ensemble_size(self) -> int:
        """The number of models in the ensemble."""
        return len(self.checkpoint_paths)

    def process_args(self) -> None:
        super(PredictArgs, self).process_args()

        if (self.calibration_method is None) != (self.calibration_path is None):
            raise ValueError(
                '--calibration_method and --calibration_path must be provided together.'
            )
        calibration_auxiliary_paths = (
            self.calibration_features_path,
            self.calibration_phase_features_path,
            self.calibration_atom_descriptors_path,
            self.calibration_bond_descriptors_path,
        )
        if self.calibration_path is None and any(
            path is not None for path in calibration_auxiliary_paths
        ):
            raise ValueError(
                'Calibration feature/descriptor paths require --calibration_path.'
            )
        if self.evaluation_scores_path is not None and self.evaluation_methods is None:
            raise ValueError(
                '--evaluation_scores_path requires --evaluation_methods.'
            )
        if (
            self.individual_ensemble_predictions
            and self.uncertainty_method == 'dropout'
        ):
            raise ValueError(
                '--individual_ensemble_predictions is not supported with '
                '--uncertainty_method dropout because Monte Carlo samples are '
                'not checkpoint ensemble members.'
            )

        if self.regression_calibrator_metric is None:
            if self.calibration_method == 'zelikman_interval':
                self.regression_calibrator_metric = 'interval'
            elif self.calibration_method in ['conformal_regression', 'conformal_quantile_regression']:
                self.regression_calibrator_metric = None
            else:
                self.regression_calibrator_metric = 'stdev'

        if self.uncertainty_method == 'dropout' and version.parse(torch.__version__) < version.parse('1.9.0'):
            raise ValueError('Dropout uncertainty is only supported for pytorch versions >= 1.9.0')

        self.smiles_columns = chemprop.data.utils.preprocess_smiles_columns(
            path=self.test_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )

        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError('Found no checkpoints. Must specify --checkpoint_path <path> or '
                             '--checkpoint_dir <dir> containing at least one checkpoint.')

        if self.ensemble_variance:
            if self.uncertainty_method in ['ensemble', None]:
                warn(
                    'The `--ensemble_variance` argument is deprecated and should \
                        be replaced with `--uncertainty_method ensemble`.',
                    DeprecationWarning,
                )
                self.uncertainty_method = 'ensemble'
            else:
                raise ValueError(
                    f'Only one uncertainty method can be used at a time. \
                        The arguement `--ensemble_variance` was provided along \
                        with the uncertainty method {self.uncertainty_method}. The `--ensemble_variance` \
                        argument is deprecated and should be replaced with `--uncertainty_method ensemble`.'
                )

        if (
            not _is_finite_number(self.calibration_interval_percentile)
            or self.calibration_interval_percentile <= 1
            or self.calibration_interval_percentile >= 100
        ):
            raise ValueError('The calibration interval must be a percentile value in the range (1,100).')

        if (
            not _is_finite_number(self.uncertainty_dropout_p)
            or not 0 < self.uncertainty_dropout_p < 1
        ):
            raise ValueError('The dropout probability must be in the range (0,1).')

        if (
            not isinstance(self.dropout_sampling_size, int)
            or isinstance(self.dropout_sampling_size, bool)
            or self.dropout_sampling_size <= 1
        ):
            raise ValueError('The argument `--dropout_sampling_size` must be an integer greater than 1.')

        # Validate that features provided for the prediction test set are also provided for the calibration set
        for (features_argument, base_features_path, cal_features_path) in [
            ('`--features_path`', self.features_path, self.calibration_features_path),
            ('`--phase_features_path`', self.phase_features_path, self.calibration_phase_features_path),
            ('`--atom_descriptors_path`', self.atom_descriptors_path, self.calibration_atom_descriptors_path),
            ('`--bond_descriptors_path`', self.bond_descriptors_path, self.calibration_bond_descriptors_path)
        ]:
            if (
                base_features_path is not None
                and self.calibration_path is not None
                and cal_features_path is None
            ):
                raise ValueError(
                    f"Additional features were provided using the argument {features_argument}. The same kinds of features must be provided for the calibration dataset."
                )

        if (
            not _is_finite_number(self.conformal_alpha)
            or not 0 < self.conformal_alpha < 1
        ):
            raise ValueError(
                "conformal_alpha should be in the range (0,1)"
            )


class InterpretArgs(CommonArgs):
    """:class:`InterpretArgs` includes :class:`CommonArgs` along with additional arguments used for interpreting a trained Chemprop model."""

    data_path: str
    """Path to data CSV file."""
    batch_size: int = 500
    """Batch size."""
    property_id: int = 1
    """Index of the property of interest in the trained model."""
    rollout: int = 20
    """Number of rollout steps."""
    c_puct: float = 10.0
    """Constant factor in MCTS."""
    max_atoms: int = 20
    """Maximum number of atoms in rationale."""
    min_atoms: int = 8
    """Minimum number of atoms in rationale."""
    prop_delta: float = 0.5
    """Minimum score to count as positive."""

    def process_args(self) -> None:
        super(InterpretArgs, self).process_args()

        for name, value in (
            ('property_id', self.property_id),
            ('rollout', self.rollout),
            ('max_atoms', self.max_atoms),
            ('min_atoms', self.min_atoms),
        ):
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                or value < 1
            ):
                raise ValueError(f'{name} must be a positive integer.')
        if self.min_atoms > self.max_atoms:
            raise ValueError('min_atoms must be less than or equal to max_atoms.')
        if not _is_finite_number(self.c_puct):
            raise ValueError('c_puct must be finite.')
        if not _is_finite_number(self.prop_delta):
            raise ValueError('prop_delta must be finite.')

        self.smiles_columns = chemprop.data.utils.preprocess_smiles_columns(
            path=self.data_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )

        if self.features_path is not None:
            raise ValueError('Cannot use --features_path <path> for interpretation since features '
                             'need to be computed dynamically for molecular substructures. '
                             'Please specify --features_generator <generator>.')

        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError('Found no checkpoints. Must specify --checkpoint_path <path> or '
                             '--checkpoint_dir <dir> containing at least one checkpoint.')


class FingerprintArgs(PredictArgs):
    """:class:`FingerprintArgs` includes :class:`PredictArgs` with additional arguments for the generation of latent fingerprint vectors."""

    fingerprint_type: Literal['MPN', 'last_FFN'] = 'MPN'
    """Choice of which type of latent fingerprint vector to use. Default is the output of the MPNN, excluding molecular features"""


class HyperoptArgs(TrainArgs):
    """:class:`HyperoptArgs` includes :class:`TrainArgs` along with additional arguments used for optimizing Chemprop hyperparameters."""

    num_iters: int = 20
    """Number of hyperparameter choices to try."""
    hyperopt_seed: int = 0
    """The initial seed used for choosing parameters in hyperopt trials. In each trial, the seed will be increased by one, skipping seeds previously used."""
    config_save_path: str
    """Path to :code:`.json` file where best hyperparameter settings will be written."""
    log_dir: str = None
    """(Optional) Path to a directory where all results of the hyperparameter optimization will be written."""
    hyperopt_checkpoint_dir: str = None
    """Path to a directory where hyperopt completed trial data is stored. Hyperopt job will include these trials if restarted.
    Can also be used to run multiple instances in parallel if they share the same checkpoint directory."""
    startup_random_iters: int = None
    """The initial number of trials that will be randomly specified before TPE algorithm is used to select the rest.
    By default will be half the total number of trials."""
    manual_trial_dirs: List[str] = None
    """Paths to save directories for manually trained models in the same search space as the hyperparameter search.
    Results will be considered as part of the trial history of the hyperparameter search."""
    search_parameter_keywords: List[str] = ["basic"]
    """The model parameters over which to search for an optimal hyperparameter configuration.
    Some options are bundles of parameters or otherwise special parameter operations.

    Special keywords are:

    * ``basic``: the default search over depth, FFN layer count, dropout, and
      linked hidden size.
    * ``linked_hidden_size``: search hidden and FFN hidden sizes while keeping
      them equal. Specifying either component separately searches them
      independently.
    * ``learning_rate``: search maximum, initial, and final learning rates plus
      warmup epochs. Initial/final rates are fractions of the maximum rate, and
      warmup is a fraction of total epochs.
    * ``all``: include all thirteen individual parameter keywords.

    Individual parameters are ``activation``, ``aggregation``,
    ``aggregation_norm``, ``batch_size``, ``depth``, ``dropout``,
    ``ffn_hidden_size``, ``ffn_num_layers``, ``final_lr``, ``hidden_size``,
    ``init_lr``, ``max_lr``, and ``warmup_epochs``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(HyperoptArgs, self).__init__(*args, **kwargs)
        # argparse mutates list-valued options in place. Do not expose the
        # annotated class default to mutations performed by another instance.
        self.search_parameter_keywords = list(self.search_parameter_keywords)

    def process_args(self) -> None:
        super(HyperoptArgs, self).process_args()

        if self.model_type != 'FFN':
            raise NotImplementedError(
                'chemprop_hyperopt currently supports only --model_type FFN. '
                'Tune LightGBM with its --lgbm_* training arguments or an '
                'external validation-only search.'
            )

        if (
            not isinstance(self.num_iters, (int, np.integer))
            or isinstance(self.num_iters, (bool, np.bool_))
            or self.num_iters < 1
        ):
            raise ValueError('num_iters must be a positive integer.')
        if (
            not isinstance(self.hyperopt_seed, (int, np.integer))
            or isinstance(self.hyperopt_seed, (bool, np.bool_))
        ):
            raise ValueError('hyperopt_seed must be an integer.')

        # Hyperparameters must be selected exclusively on validation data.
        # Evaluating trials on the test split leaks test labels into model
        # selection and invalidates the final test estimate.
        self.data_type = 'validation'
        self.skip_test_evaluation = True

        # Assign log and checkpoint directories if none provided
        if self.log_dir is None:
            self.log_dir = self.save_dir
        if self.hyperopt_checkpoint_dir is None:
            self.hyperopt_checkpoint_dir = self.log_dir
        
        # Set number of startup random trials
        if self.startup_random_iters is None:
            self.startup_random_iters = self.num_iters // 2
        if (
            not isinstance(self.startup_random_iters, (int, np.integer))
            or isinstance(self.startup_random_iters, (bool, np.bool_))
            or not 0 <= self.startup_random_iters <= self.num_iters
        ):
            raise ValueError(
                'startup_random_iters must be an integer between 0 and num_iters.'
            )

        # Construct set of search parameters
        supported_keywords = [
            "basic", "learning_rate", "linked_hidden_size", "all",
            "activation", "aggregation", "aggregation_norm", "batch_size", "depth",
            "dropout", "ffn_hidden_size", "ffn_num_layers", "final_lr", "hidden_size",
            "init_lr", "max_lr", "warmup_epochs"
        ]
        supported_parameters = [
            "activation", "aggregation", "aggregation_norm", "batch_size", "depth",
            "dropout", "ffn_hidden_size", "ffn_num_layers", "final_lr_ratio", "hidden_size",
            "init_lr_ratio", "linked_hidden_size", "max_lr", "warmup_epochs"
        ]
        unsupported_keywords = set(self.search_parameter_keywords) - set(supported_keywords)
        if len(unsupported_keywords) != 0:
            raise NotImplementedError(
                f"Keywords for what hyperparameters to include in the search are designated \
                    with the argument `--search_parameter_keywords`. The following unsupported\
                    keywords were received: {unsupported_keywords}. The available supported\
                    keywords are: {supported_keywords}"
            )
        search_parameters = set()
        if "all" in self.search_parameter_keywords:
            search_parameters.update(supported_parameters)
        if "basic" in self.search_parameter_keywords:
            search_parameters.update(["depth", "ffn_num_layers", "dropout", "linked_hidden_size"])
        if "learning_rate" in self.search_parameter_keywords:
            search_parameters.update(["max_lr", "init_lr_ratio", "final_lr_ratio", "warmup_epochs"])
        for kw in self.search_parameter_keywords:
            if kw in supported_parameters:
                search_parameters.add(kw)
        if "init_lr" in self.search_parameter_keywords:
            search_parameters.add("init_lr_ratio")
        if "final_lr" in self.search_parameter_keywords:
            search_parameters.add("final_lr_ratio")
        if "linked_hidden_size" in search_parameters and ("hidden_size" in search_parameters or "ffn_hidden_size" in search_parameters):
            search_parameters.remove("linked_hidden_size")
            search_parameters.update(["hidden_size", "ffn_hidden_size"])
        self.search_parameters = list(search_parameters)


class SklearnTrainArgs(TrainArgs):
    """:class:`SklearnTrainArgs` includes :class:`TrainArgs` along with additional arguments for training a scikit-learn model."""

    model_type: Literal['random_forest', 'svm']
    """scikit-learn model to use."""
    class_weight: Literal['balanced'] = None
    """How to weight classes (None means no class balance)."""
    single_task: bool = False
    """Whether to run each task separately (needed when dataset has null entries)."""
    radius: int = 2
    """Morgan fingerprint radius."""
    num_bits: int = 2048
    """Number of bits in morgan fingerprint."""
    num_trees: int = 500
    """Number of random forest trees."""
    impute_mode: Literal['single_task', 'median', 'mean', 'linear', 'frequent'] = None
    """How to impute missing data (None means no imputation)."""

    def process_args(self) -> None:
        super(SklearnTrainArgs, self).process_args()

        for name, value, minimum in (
            ('radius', self.radius, 0),
            ('num_bits', self.num_bits, 1),
            ('num_trees', self.num_trees, 1),
        ):
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                or value < minimum
            ):
                raise ValueError(f'{name} must be an integer of at least {minimum}.')

        if self.dataset_type not in {'classification', 'regression'}:
            raise ValueError(
                'Sklearn models support only classification and regression datasets.'
            )
        if self.target_weights is not None:
            raise NotImplementedError(
                'Sklearn models do not support --target_weights. Use '
                '--data_weights_path for row-wise sample weights instead.'
            )
        if self.class_weight is not None and self.dataset_type != 'classification':
            raise ValueError('--class_weight is only supported for classification.')
        if self.single_task and self.impute_mode is not None:
            raise ValueError(
                '--single_task already removes missing labels per task and cannot '
                'be combined with --impute_mode.'
            )
        regression_imputation = {'single_task', 'median', 'mean', 'linear'}
        classification_imputation = {'single_task', 'linear', 'frequent'}
        supported_imputation = (
            regression_imputation
            if self.dataset_type == 'regression'
            else classification_imputation
        )
        if self.impute_mode is not None and self.impute_mode not in supported_imputation:
            raise ValueError(
                f'--impute_mode {self.impute_mode} is not supported for '
                f'{self.dataset_type} data.'
            )


class SklearnPredictArgs(CommonArgs):
    """:class:`SklearnPredictArgs` contains arguments used for predicting with a trained scikit-learn model."""

    test_path: str
    """Path to CSV file containing testing data for which predictions will be made."""
    smiles_columns: List[str] = None
    """List of names of the columns containing SMILES strings.
    By default, uses the first :code:`number_of_molecules` columns."""
    number_of_molecules: int = 1
    """Number of molecules in each input to the model.
    This must equal the length of :code:`smiles_columns` (if not :code:`None`)."""
    preds_path: str
    """Path to CSV file where predictions will be saved."""
    checkpoint_dir: str = None
    """Path to directory containing model checkpoints (:code:`.pkl` file)"""
    checkpoint_path: str = None
    """Path to model checkpoint (:code:`.pkl` file)"""
    checkpoint_paths: List[str] = None
    """List of paths to model checkpoints (:code:`.pkl` files)"""

    def process_args(self) -> None:
        self._validate_common_numeric_args()
        self.smiles_columns = chemprop.data.utils.preprocess_smiles_columns(
            path=self.test_path,
            smiles_columns=self.smiles_columns,
            number_of_molecules=self.number_of_molecules,
        )

        # Load checkpoint paths
        self.checkpoint_paths = get_checkpoint_paths(
            checkpoint_path=self.checkpoint_path,
            checkpoint_paths=self.checkpoint_paths,
            checkpoint_dir=self.checkpoint_dir,
            ext='.pkl'
        )
        if not self.checkpoint_paths:
            raise ValueError(
                'Found no sklearn checkpoints. Specify --checkpoint_path, '
                '--checkpoint_paths, or --checkpoint_dir.'
            )
