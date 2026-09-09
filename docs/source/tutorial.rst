.. _tutorial:

Tutorial
========

Data
----

In order to train a model, you must provide training data containing molecules (as SMILES strings) and known target values. Targets can either be real numbers, if performing regression, or binary (i.e. 0s and 1s), if performing classification. Target values which are unknown can be left as blanks.

Our model can either train on a single target ("single tasking") or on multiple targets simultaneously ("multi-tasking").

The data file must be be a **CSV file with a header row**. For example:

.. code-block::

   smiles,NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53
   CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0,0,1,,,0,0,1,0,0,0,0
   CCN1C(=O)NC(c2ccccc2)C1=O,0,0,0,0,0,0,0,,0,,0,0
   ...

By default, it is assumed that the SMILES are in the first column and the targets are in the remaining columns. However, the specific columns containing the SMILES and targets can be specified using the :code:`--smiles_column <column>` and :code:`--target_columns <column_1> <column_2> ...` flags, respectively.

Datasets from `MoleculeNet <https://moleculenet.org/>`_ and a 450K subset of ChEMBL from `<http://www.bioinf.jku.at/research/lsc/index.html>`_ have been preprocessed and are available in `data.tar.gz <https://github.com/Kuroki-A/chemprop/blob/master/data.tar.gz>`_. To uncompress them, run :code:`tar xvzf data.tar.gz`.

Training
--------

To train a model, run:

.. code-block::

   chemprop_train --data_path <path> --dataset_type <type> --save_dir <dir>

where :code:`<path>` is the path to a CSV file containing a dataset, :code:`<type>` is either "classification" or "regression" depending on the type of the dataset, and :code:`<dir>` is the directory where model checkpoints will be saved.

For example:

.. code-block::

   chemprop_train --data_path data/tox21.csv --dataset_type classification --save_dir tox21_checkpoints

A full list of available command-line arguments can be found in :ref:`args`.

If installed from source, :code:`chemprop_train` can be replaced with :code:`python train.py`.

Notes:

* The default metric for classification is AUC and the default metric for regression is RMSE. Other metrics may be specified with :code:`--metric <metric>`.
* :code:`--save_dir` may be left out if you don't want to save model checkpoints.
* :code:`--quiet` can be added to reduce the amount of debugging information printed to the console. Both a quiet and verbose version of the logs are saved in the :code:`save_dir`.


Train/Validation/Test Splits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our code supports several methods of splitting data into train, validation, and test sets.

**Random:** By default, the data will be split randomly into train, validation, and test sets.

**Scaffold:** Alternatively, the data can be split by molecular scaffold so that the same scaffold never appears in more than one split. This can be specified by adding :code:`--split_type scaffold_balanced`.

**Separate val/test:** If you have separate data files you would like to use as the validation or test set, you can specify them with :code:`--separate_val_path <val_path>` and/or :code:`--separate_test_path <test_path>`.

Externally supplied index and fold files are validated for integer type,
range, duplicates, and train/validation/test overlap. Rows omitted by an
external split are reported explicitly rather than silently included in
another split.

Note: By default, both random and scaffold split the data into 80% train, 10% validation, and 10% test. This can be changed with :code:`--split_sizes <train_frac> <val_frac> <test_frac>`. For example, the default setting is :code:`--split_sizes 0.8 0.1 0.1`. Both also involve a random component and can be seeded with :code:`--seed <seed>`. The default setting is :code:`--seed 0`.

Cross validation
^^^^^^^^^^^^^^^^

k-fold cross-validation can be run by specifying :code:`--num_folds <k>`. The default is :code:`--num_folds 1`.

Ensembling
^^^^^^^^^^

To train an ensemble, specify the number of models in the ensemble with :code:`--ensemble_size <n>`. The default is :code:`--ensemble_size 1`.

Checkpoint-only evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`chemprop_train --test` skips optimization and re-evaluates supplied
checkpoints on the configured validation and test splits. It reconstructs each
saved architecture and uses the target/input scalers stored in the checkpoint;
evaluation labels are never used to fit scalers. Architecture flags such as
hidden size need not be repeated, but dataset type, ordered targets, molecule
and reaction semantics, generated/external features, descriptors, and spectra
settings must match. Ensemble members must also have compatible scaler state.
:code:`--test` cannot be combined with :code:`--checkpoint_frzn`.

LightGBM Heads
^^^^^^^^^^^^^^

For regression and binary classification, :code:`--model_type lgbm` trains
one LightGBM head per task. LightGBM does not optimize the neural MPN encoder,
so this backend **requires** :code:`--features_only` together with a
deterministic :code:`--features_generator` (for example, Morgan) or a matching
external :code:`--features_path`. An untrained random MPN representation is
seed-dependent and is rejected. Feature scalers, boosters, and metadata are
stored in versioned :code:`.pkl` bundles. Missing multitask targets are
supported.

Prediction also refuses earlier versioned bundles whose recorded training
arguments used :code:`features_only=False`; retrain those models with this
release and a deterministic feature source.

.. code-block::

   chemprop_train --data_path data.csv --dataset_type regression \
      --model_type lgbm --features_generator morgan --features_only \
      --save_dir lgbm_checkpoints
   chemprop_predict --test_path test.csv --checkpoint_dir lgbm_checkpoints \
      --features_generator morgan --preds_path predictions.csv

Prediction infers LightGBM when a checkpoint source contains only
:code:`.pkl` bundles. Boosting rounds, early stopping, learning rate, leaves,
feature/bagging fractions, minimum leaf size, and CPU threads are configurable
with the :code:`--lgbm_*` arguments documented in :ref:`args`. This backend
currently supports mean-squared-error regression and binary-cross-entropy
classification. Checkpoint warm-starting and :code:`chemprop_hyperopt` are
rejected explicitly; use the :code:`--lgbm_*` options for tuning.
:code:`--target_weights` is rejected because each target is trained by an
independent booster; row-wise :code:`--data_weights_path` remains supported.
Reaction SMILES with a molecular generator and atom/bond descriptor inputs are
also rejected because the features-only LightGBM representation would ignore
chemically relevant inputs. Supply explicit reaction-aware molecule-level
features instead. LightGBM does not implement uncertainty calibration,
uncertainty methods, or uncertainty evaluation options.

Hyperparameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although the default message passing architecture works quite well on a variety of datasets, optimizing the hyperparameters for a particular dataset often leads to marked improvement in predictive performance. We have automated hyperparameter optimization via Bayesian optimization (using the `hyperopt <https://github.com/hyperopt/hyperopt>`_ package), which will find the optimal hidden size, depth, dropout, and number of feed-forward layers for our model. Optimization can be run as follows:

.. code-block::

   chemprop_hyperopt --data_path <data_path> --dataset_type <type> --num_iters <n> --config_save_path <config_path>


where :code:`<n>` is the number of hyperparameter settings to try and :code:`<config_path>` is the path to a :code:`.json` file where the optimal hyperparameters will be saved.

If installed from source, :code:`chemprop_hyperopt` can be replaced with :code:`python hyperparameter_optimization.py`.

Once hyperparameter optimization is complete, the optimal hyperparameters can be applied during training by specifying the config path as follows:

.. code-block::

   chemprop_train --data_path <data_path> --dataset_type <type> --config_path <config_path>

Hyperparameter trials are selected only from validation scores and skip test
evaluation. Keep an independent final test set and evaluate it once after the
search. For multiple cross-validation splits, run the optimizer separately for
each training/validation split while retaining the corresponding test fold.

Additional Features
^^^^^^^^^^^^^^^^^^^

While the model works very well on its own, especially after hyperparameter optimization, we have seen that additional features can further improve performance on certain datasets. The additional features can be added at the atom-, bond, or molecule-level. Molecule-level features can be either automatically generated by RDKit or custom features provided by the user.

Molecule-Level RDKit 2D Features
""""""""""""""""""""""""""""""""

As a starting point, we recommend using pre-normalized RDKit features by using the :code:`--features_generator rdkit_2d_normalized --no_features_scaling` flags. In general, we recommend NOT using the :code:`--no_features_scaling` flag (i.e. allow the code to automatically perform feature scaling), but in the case of :code:`rdkit_2d_normalized`, those features have been pre-normalized and don't require further scaling.

The built-in families include:

* RDKit fingerprints: :code:`morgan`, :code:`morgan_count`, :code:`maccs`,
  :code:`rdkit`, :code:`avalon`, :code:`atompair`, :code:`erg`, and
  :code:`erg_float`.
* RDKit/descriptastorus descriptors: :code:`rdkit_2d`,
  :code:`rdkit_2d_normalized`, their :code:`_wo_fr` variants,
  :code:`rdkit_2d_208`, :code:`rdkit_2d_400`,
  :code:`rdkit_2d_autocorr`, :code:`rdkit_2d_bcut`, and
  :code:`rdkit_2d_all`.
* Molfeat 2D/scaffold/pharmacophore features: :code:`fcfp`,
  :code:`fcfp_count`, :code:`topological`, :code:`topological_count`,
  :code:`layered`, :code:`avalon_count`, :code:`rdkit_count`,
  :code:`atompair_count`, :code:`pattern`, :code:`estate`, :code:`secfp`,
  :code:`cats2d`, :code:`scaffoldkeys`, and :code:`pharm2d`.
* Direct MAP4 implementations: legacy/Molfeat-compatible :code:`map4` and
  native :code:`map4_v1_1`.
* Optional :code:`mordred`, :code:`padelpy`, and Molfeat pretrained
  representations shown by :code:`chemprop_train --help`.

Install the :code:`features` extra for stable local backends. Use
:code:`features-pretrained` only for registered pretrained Molfeat models,
whose Transformer and DGL dependencies are intentionally isolated;
:code:`features-all` remains a compatibility alias. Imports are lazy, and
normal data loading uses ordered chunked generation with duplicate input
reuse. :code:`scripts/save_features.py`
provides native batching or bounded multiprocessing, restartable bounded
chunks, disk-backed consolidation, and a schema/hash manifest next to its
:code:`.npz` output. The conda environment supplies OpenJDK 17 for the
:code:`padelpy` backend.

PaDEL calculation or numeric-conversion failures stop feature generation and
report the affected batch row, SMILES, and underlying cause. A failed molecule
is never replaced with an all-zero descriptor row, because that would silently
change model inputs.

:code:`--selected_features_path` accepts a CSV with generator names as columns
and the ordered feature names to retain as values. Fixed fingerprints use
:code:`bit_N` (:code:`morgan`, :code:`maccs`, :code:`rdkit`, :code:`avalon`,
:code:`atompair`), :code:`count_N` (:code:`morgan_count`), :code:`erg_N`
(:code:`erg`, :code:`erg_float`), or :code:`fp_N` (:code:`map4`,
:code:`map4_v1_1`) labels. Repeated names intentionally repeat output columns.

.. warning::
   This maintenance version fixes fixed-fingerprint and pretrained-Molfeat
   generators that previously accepted selected columns but silently returned
   full vectors. Checkpoints trained with that old behavior and a selected
   feature CSV should be retrained if their input width no longer matches.

:code:`map4` is the canonicalized, folded 2,048-bit MAP4 v1.0 definition
expected by Molfeat 0.11. :code:`map4_v1_1` is the canonicalized native
:code:`map4` 1.1.3 definition. They are not bit-compatible, so training and
prediction must use the same name. For example, select the 1.1.3 definition
with :code:`--features_generator map4_v1_1`; complete train, predict, and
offline-generation commands are documented in :ref:`features`.

.. warning::
   Selected descriptor columns are now correctly applied to reaction
   reactants. Legacy reaction checkpoints trained with a selected-feature CSV
   may have the old full-width vector and should be retrained if a feature
   width mismatch is reported.

Molecule-Level Custom Features
""""""""""""""""""""""""""""""

.. warning::
   Python pickle inputs (``.pkl``, ``.pckl``, and ``.pickle``, including
   pandas pickle files) can execute arbitrary code while loading. Open them
   only when they were created by you or another fully trusted source. Prefer
   ``.npz``, ``.npy``, or ``.csv`` for ``--features_path``,
   ``--atom_descriptors_path``, and ``--bond_descriptors_path`` whenever
   possible. Model checkpoints and the optional dataset cache have the same
   trust requirement, as do cross-validation split/index pickle files and
   Hyperopt trial checkpoint directories.

If you install from source, you can modify the code to load custom features as follows:

1. **Generate features:** If you want to generate features in code, you can write a custom features generator function in :code:`chemprop/features/features_generators.py`. Scroll down to the bottom of that file to see a features generator code template.
2. **Load features:** If you have features saved as a numpy :code:`.npy` file or as a :code:`.csv` file, you can load the features by using :code:`--features_path /path/to/features`. Note that the features must be in the same order as the SMILES strings in your data file. Also note that :code:`.csv` files must have a header row and the features should be comma-separated with one line per molecule.

Atom-Level Features
"""""""""""""""""""

Similar to the additional molecular features described above, you can also provide additional atomic features via :code:`--atom_descriptors_path /path/to/features` with valid file formats:

* :code:`.npz` file, where descriptors are saved as 2D array for each molecule in the exact same order as the SMILES strings in your data file.
* :code:`.pkl` / :code:`.pckl` / :code:`.pickle` containing a pandas dataframe with smiles as index and numpy array of descriptors as columns.
* :code:`.sdf` containing all mol blocks with descriptors as entries.

Pickle rows are validated and reordered using the SMILES index before rows with
missing targets are filtered. Every raw CSV row must have one corresponding
pickle row. Duplicate SMILES are accepted only when both inputs already have
exactly the same order; reordered duplicates are rejected as ambiguous.

The order of the descriptors for each atom per molecule must match the ordering of atoms in the RDKit molecule object. Further information on supplying atomic descriptors can be found `here <https://github.com/chemprop/chemprop/releases/tag/v1.1.0>`_.

Users must select in which way atom descriptors are used. The command line option :code:`--atom_descriptors descriptor` concatenates the new features to the embedded atomic features after the D-MPNN with an additional linear layer. The option :code:`--atom_descriptors feature` concatenates the features to each atomic feature vector before the D-MPNN, so that they are used during message-passing. Alternatively, the user can overwrite the default atom features with the custom features using the option :code:`--overwrite_default_atom_features`.

Similar to the molecule-level features, the atom-level descriptors and features are scaled by default. This can be disabled with the option :code:`--no_atom_descriptor_scaling`

Bond-Level Features
"""""""""""""""""""

Bond-level features can be provided in the same format as the atom-level features, using the option :code:`--bond_descriptors_path /path/to/features`. The order of the features for each molecule must match the bond ordering in the RDKit molecule object.

The bond-level features are concatenated with the bond feature vectors before the D-MPNN, such that they are used during message-passing. Alternatively, the user can overwrite the default bond features with the custom features using the option :code:`--overwrite_default_bond_features`.

Similar to molecule-, and atom-level features, the bond-level features are scaled by default. This can be disabled with the option :code:`--no_bond_descriptor_scaling`.

Reaction
^^^^^^^^

As an alternative to molecule SMILES, Chemprop can also process atom-mapped reaction SMILES (see `Daylight manual <https://www.daylight.com/meetings/summerschool01/course/basics/smirks.html>`_ for details on reaction SMILES), which consist of three parts denoting reactants, agents and products, separated by ">". Use the option :code:`--reaction` to enable the input of reactions, which transforms the reactants and products of each reaction to the corresponding condensed graph of reaction and changes the initial atom and bond features to hold information from both the reactant and product (option :code:`--reaction_mode reac_prod`), or from the reactant and the difference upon reaction (option :code:`--reaction_mode reac_diff`, default) or from the product and the difference upon reaction (option :code:`--reaction_mode prod_diff`). In reaction mode, Chemprop thus concatenates information to each atomic and bond feature vector, for example, with option :code:`--reaction_mode reac_prod`, each atomic feature vector holds information on the state of the atom in the reactant (similar to default Chemprop), and concatenates information on the state of the atom in the product, so that the size of the D-MPNN increases slightly. Agents are discarded. Functions incompatible with a reaction as input (scaffold splitting and feature generation) are carried out on the reactants only. If the atom-mapped reaction SMILES contain mapped hydrogens, enable explicit hydrogens via :code:`--explicit_h`. Example of an atom-mapped reaction SMILES denoting the reaction of methanol to formaldehyde without hydrogens: :code:`[CH3:1][OH:2]>>[CH2:1]=[O:2]` and with hydrogens: :code:`[C:1]([H:3])([H:4])([H:5])[O:2][H:6]>>[C:1]([H:3])([H:4])=[O:2].[H:5][H:6]`. The reactions do not need to be balanced and can thus contain unmapped parts, for example leaving groups, if necessary.
For further details and benchmarking, as well as a citable reference, please see `DOI 10.33774/chemrxiv-2021-frfhz <https://doi.org/10.33774/chemrxiv-2021-frfhz>`_.

Pretraining
^^^^^^^^^^^

An existing model, for example from training on a larger, lower quality dataset, can be used for parameter-initialization of a new model by providing a checkpoint of the existing model using either:

 * :code:`--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. :code:`--save_dir` during training of the old model). This will walk the directory, and load all :code:`.pt` files it finds.
 * :code:`--checkpoint_path <path>` Path to a model checkpoint file (:code:`.pt` file).

when training the new model. Warm starts build the requested current
architecture, require a complete compatible MPN encoder, and copy only
shape-compatible non-encoder state; skipped readout entries are logged. All
copied parameters remain trainable. Use :code:`--checkpoint_frzn` when the
loaded encoder (and optionally leading FFN layers) must be frozen. Frozen
transfer validates the complete mapping before modifying the model and rejects
ambiguous combinations such as :code:`features_only`, incompatible shared
encoders, and unsupported partial PReLU FFN freezing.

Missing target values
^^^^^^^^^^^^^^^^^^^^^

When training multitask models (models which predict more than one target simultaneously), sometimes not all target values are known for all molecules in the dataset. Chemprop automatically handles missing entries in the dataset by masking out the respective values in the loss function, so that partial data can be utilized, too. The loss function is rescaled according to all non-missing values, and missing values furthermore do not contribute to validation or test errors. Training on partial data is therefore possible and encouraged (versus taking out datapoints with missing target entries). No keyword is needed for this behavior, it is the default.

In contrast, when using :code:`sklearn_train.py` (a utility script provided within Chemprop that trains standard models such as random forests on Morgan fingerprints via the python package scikit-learn), multi-task models cannot be trained on datasets with partially missing targets. However, one can instead train individual models for each task (via the argument :code:`--single_task`), where missing values are automatically removed from the dataset. Thus, the training still makes use of all non-missing values, but by training individual models for each task, instead of one model with multiple output values. This restriction only applies to sklearn models (via  :code:`sklearn_train` or :code:`python sklearn_train.py`), but NOT to default Chemprop models via :code:`chemprop_train` or :code:`python train.py`.

Caching
^^^^^^^

:code:`--use_cache` stores the fully loaded dataset in a content-addressed
:code:`.chemprop_cache` directory. :code:`CHEMPROP_CACHE_DIR` selects another
location. This cache is pickle-based trusted local data; its directory must be
owned by the current user, private (mode ``0700``), and free of symbolic-link
components. On POSIX, descriptor-relative no-follow access keeps validation,
loading, and atomic replacement bound to the same directory even if a path is
renamed concurrently. The cache key includes source content, feature
configuration, and the relevant implementation identity, so stale entries are
ignored.

By default, the molecule objects created from each SMILES string are cached for all dataset sizes, and the graph objects created from each molecule object are cached for datasets up to 10000 molecules. If memory permits, you may use the keyword :code:`--cache_cutoff inf` to set this cutoff from 10000 to infinity to always keep the generated graphs in cache (or to another integer value for custom behavior). This may speed up training (depending on the dataset size, molecule size, number of epochs and GPU support), since the graphs do not need to be recreated each epoch, but increases memory usage considerably. Below the cutoff, graphs are created sequentially in the first epoch. Above the cutoff, graphs are created in parallel (on :code:`--num_workers <int>` workers) for each epoch. If training on a GPU, training without caching and creating graphs on the fly in parallel is often preferable. On CPU, training with caching if often preferable for medium-sized datasets and a very low number of CPUs. If a very large dataset causes memory issues, you might turn off caching even of the molecule objects via the commands :code:`--no_cache_mol` to reduce memory usage further.
   
Predicting
----------

To load a trained model and make predictions, run :code:`predict.py` and specify:

* :code:`--test_path <path>` Path to the data to predict on.
* A checkpoint by using either:

  * :code:`--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. :code:`--save_dir` during training). This will walk the directory, load all :code:`.pt` files it finds, and treat the models as an ensemble.
  * :code:`--checkpoint_path <path>` Path to a model checkpoint file (:code:`.pt` file).

* :code:`--preds_path` Path where a CSV file containing the predictions will be saved.

For example:

.. code-block::

   chemprop_predict --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints --preds_path tox21_preds.csv

or

.. code-block::

   chemprop_predict --test_path data/tox21.csv --checkpoint_path tox21_checkpoints/fold_0/model_0/model.pt --preds_path tox21_preds.csv

If installed from source, :code:`chemprop_predict` can be replaced with :code:`python predict.py`.

Constrained atom/bond calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For constrained atom/bond checkpoints, the prediction and uncertainty
calibration datasets need separate, row-aligned constraint files. Supply both
:code:`--constraints_path prediction_constraints.csv` and
:code:`--calibration_constraints_path calibration_constraints.csv` whenever
:code:`--calibration_path` is used. A missing counterpart, wrong task width,
row-count mismatch, nonnumeric value, or nonfinite value is rejected. Atom/bond
vectors written to prediction CSV files are JSON arrays; parse them with a JSON
parser rather than Python-literal evaluation.

Interpreting
^^^^^^^^^^^^

It is often helpful to provide explanation of model prediction (i.e., this molecule is toxic because of this substructure). Given a trained model, you can interpret the model prediction using the following command:

.. code-block::

   chemprop_interpret --data_path data/tox21.csv --checkpoint_dir tox21_checkpoints/fold_0/ --property_id 1

If installed from source, :code:`chemprop_interpret` can be replaced with :code:`python interpret.py`.

The output will be like the following:

* The first column is a molecule and second column is its predicted property (in this case NR-AR toxicity).
* The third column is the smallest substructure that made this molecule classified as toxic (which we call rationale).
* The fourth column is the predicted toxicity of that substructure.

As shown in the first row, when a molecule is predicted to be non-toxic, we will not provide any rationale for its prediction.

.. csv-table::
   :header: "smiles", "NR-AR", "rationale", "rationale_score"
   :widths: 20, 10, 20, 10

   "O=[N+]([O-])c1cc(C(F)(F)F)cc([N+](=O)[O-])c1Cl", "0.014", "", ""
   "CC1(C)O[C@@H]2C[C@H]3[C@@H]4C[C@H](F)C5=CC(=O)C=C[C@]5(C)[C@H]4[C@@H](O)C[C@]3(C)[C@]2(C(=O)CO)O1", "0.896", "C[C@]12C=CC(=O)C=C1[CH2:1]C[CH2:1][CH2:1]2", "0.769"
   "C[C@]12CC[C@H]3[C@@H](CC[C@@]45O[C@@H]4C(O)=C(C#N)C[C@]35C)[C@@H]1CC[C@@H]2O", "0.941", "C[C@]12C[CH:1]=[CH:1][C@H]3O[C@]31CC[C@@H]1[C@@H]2CC[C:1][CH2:1]1", "0.808"
   "C[C@]12C[C@H](O)[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]43C)[C@@H]1CC[C@]2(O)C(=O)COP(=O)([O-])[O-]", "0.957", "C1C[CH2:1][C:1][C@@H]2[C@@H]1[C@@H]1CC[C:1][C:1]1C[CH2:1]2", "0.532"

Chemprop's interpretation script explains model prediction one property at a time. :code:`--property_id 1` tells the script to provide explanation for the first property in the dataset (which is NR-AR). In a multi-task training setting, you will need to change :code:`--property_id` to provide explanation for each property in the dataset.

For computational efficiency, we currently restricted the rationale to have maximum 20 atoms and minimum 8 atoms. You can adjust these constraints through :code:`--max_atoms` and :code:`--min_atoms` argument.

Please note that the interpreting framework is currently only available for models trained on properties of single molecules, that is, multi-molecule models generated via the :code:`--number_of_molecules` command are not supported.

TensorBoard
^^^^^^^^^^^

During training, TensorBoard logs are automatically saved to the same directory as the model checkpoints. To view TensorBoard logs, run :code:`tensorboard --logdir=<dir>` where :code:`<dir>` is the path to the checkpoint directory. Then navigate to `<http://localhost:6006>`_.

Web Interface
-------------

For those less familiar with the command line, Chemprop also includes a web interface which allows for basic training and predicting. See :ref:`web` for more details.

Within a python script
----------------------

Model training and predicting can also be embedded within a python script. To train a model, provide arguments as a list of strings (arguments are identical to command line mode),
parse the arguments, and then call :code:`chemprop.train.cross_validate()`::

  import chemprop

  arguments = [
      '--data_path', 'data/tox21.csv',
      '--dataset_type', 'classification',
      '--save_dir', 'tox21_checkpoints'
  ]

  args = chemprop.args.TrainArgs().parse_args(arguments)
  mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

For predicting with a given model, either a list of smiles or a csv file can be used as input. To use a csv file ::

  import chemprop

  arguments = [
      '--test_path', 'data/tox21.csv',
      '--preds_path', 'tox21_preds.csv',
      '--checkpoint_dir', 'tox21_checkpoints'
  ]
  
  args = chemprop.args.PredictArgs().parse_args(arguments)
  preds = chemprop.train.make_predictions(args=args)

If you only want to use the predictions :code:`preds` within the script, and not save the file, set :code:`preds_path` to :code:`/dev/null`. To predict on a list of smiles, run::

  import chemprop

  smiles = [['CCC'], ['CCCC'], ['OCC']]
  arguments = [
      '--test_path', '/dev/null',
      '--preds_path', '/dev/null',
      '--checkpoint_dir', 'tox21_checkpoints'
  ]

  args = chemprop.args.PredictArgs().parse_args(arguments)
  preds = chemprop.train.make_predictions(args=args, smiles=smiles)

where the given :code:`test_path` will be discarded if a list of smiles is provided. If you want to predict multiple sets of molecules consecutively, it is more efficient to
only load the chemprop model once, and then predict with the preloaded model (instead of loading the model for every prediction)::

  import chemprop

  arguments = [
      '--test_path', '/dev/null',
      '--preds_path', '/dev/null',
      '--checkpoint_dir', 'tox21_checkpoints'
  ]

  args = chemprop.args.PredictArgs().parse_args(arguments)

  model_objects = chemprop.train.load_model(args=args)
  
  smiles = [['CCC'], ['CCCC'], ['OCC']]
  preds = chemprop.train.make_predictions(args=args, smiles=smiles, model_objects=model_objects)

  smiles = [['CCCC'], ['CCCCC'], ['COCC']]
  preds = chemprop.train.make_predictions(args=args, smiles=smiles, model_objects=model_objects)
