.. _features:

Features
========

`chemprop.features <https://github.com/Kuroki-A/chemprop/tree/master/chemprop/features>`_ contains functions for featurizing molecules. This includes both atom/bond features used in message passing and additional molecule-level features appended after message passing.

Featurization
-------------

Classes and functions from `chemprop.features.featurization.py <https://github.com/Kuroki-A/chemprop/blob/master/chemprop/features/featurization.py>`_. Featurization specifically includes computation of the atom and bond features used in message passing.

.. automodule:: chemprop.features.featurization
   :members:

Features Generators
-------------------

Classes and functions from `chemprop.features.features_generators.py <https://github.com/Kuroki-A/chemprop/blob/master/chemprop/features/features_generators.py>`_. Features generators are used for computing additional molecule-level features that are appended after message passing. Optional backends are imported lazily. :func:`generate_features_batch` preserves input order while using native batches where available, and the schema/config helpers support reproducible feature manifests and checkpoints.

.. automodule:: chemprop.features.features_generators
   :members:

Reproducibility and offline generation
--------------------------------------

Built-in molecular generators canonicalize structural inputs and remove atom
map labels from a private copy. String and RDKit-molecule inputs therefore use
the same representation. :code:`scripts/save_features.py` follows the same
policy as runtime loading: reaction rows are featurized from the reactant,
hydrogen-only rows use a correctly typed zero vector, input order is preserved,
and native batching or bounded worker processes reuse generator instances.

Runtime :code:`morgan`, :code:`morgan_count`, :code:`rdkit`, and
:code:`atompair` generation uses RDKit native batches with at most four
affinity-visible threads. The offline script defaults these generators to its
bounded process pool because parallel SMILES parsing is faster there.
:code:`--sequential` selects scalar calls; :code:`--num_workers 1` or an
explicit :code:`--batch_size` selects the one-process native-batch path.
Explicit MAP4 worker selection is unchanged.

Feature metadata schema 2 identifies each managed built-in with a targeted
:code:`semantic_revision` plus its configuration and value-affecting dependency
versions. An unrelated edit elsewhere in the generator module no longer
invalidates every checkpoint. A custom/plugin generator remains conservatively
identified by a hash of its complete source module so helper changes are also
detected.

.. warning::
   Schema 1 checkpoints which generate molecular features at prediction time
   cannot prove equivalence to schema 2 and must be retrained. An interrupted
   schema 1 :code:`save_features` job needs :code:`--restart` once. A completed
   schema 1 :code:`.npz` and its original manifest remain usable as materialized
   external features when the exact same files are supplied for training and
   prediction.

Install stable local backends with
:code:`python -m pip install -e '.[features]'`. Registered pretrained Molfeat
models require the larger, historically constrained
:code:`features-pretrained` extra.

MAP4 compatibility
------------------

Two deliberately distinct MAP4 generators are available:

* :code:`map4` canonicalizes the molecule and implements the folded 2,048-bit
  `MAP4 v1.0 algorithm <https://github.com/reymond-group/map4/tree/v1.0>`_
  expected by Molfeat 0.11, including lexicographic atom-environment ordering.
  This is the recommended choice for compatibility with the original MAP4
  definition and Molfeat.
* :code:`map4_v1_1` canonicalizes the molecule and uses the native
  :code:`map4` 1.1.3 implementation and shingle ordering. It is provided for
  explicit experiments and models trained with that exact named generator.

The two vectors are not bit-compatible. A model trained with one generator
must be predicted with that same generator, and an offline feature archive
must not be relabeled or reused as the other kind. New checkpoints and feature
manifests record generator metadata so detectable mismatches fail early.

Both generators retain all disconnected input fragments. Unlike the upstream
v1.0 command-line :code:`--clean-mols` behavior, Chemprop does not silently
select only the largest fragment. The retained-fragment policy is recorded in
feature and checkpoint metadata.

The environment intentionally installs :code:`map4==1.1.3`, not the obsolete
v1.0 package and its :code:`tmap` dependency. Chemprop implements only the
needed folded v1.0 path directly with RDKit and MHFP, and exposes a compatibility
class so Molfeat 0.11 can still import its fingerprint modules. The native
package remains available to :code:`map4_v1_1`.

For example, to use the native 1.1.3 behavior for training and prediction:

.. code-block:: bash

   chemprop_train --data_path data.csv --dataset_type regression \
     --features_generator map4_v1_1 --save_dir map4_v1_1_checkpoints

   chemprop_predict --test_path test.csv --checkpoint_dir map4_v1_1_checkpoints \
     --features_generator map4_v1_1 --preds_path predictions.csv

To precompute the same feature definition instead:

.. code-block:: bash

   python scripts/save_features.py --data_path data.csv \
     --features_generator map4_v1_1 --save_path map4_v1_1_features.npz

For a large offline job, explicitly request a persistent process pool. Each
worker loads its own backend, so size this value for available RAM; omit it for
small jobs where process startup would dominate.

.. code-block:: bash

   python scripts/save_features.py --data_path data.csv \
     --features_generator map4_v1_1 --num_workers 4 \
     --save_path map4_v1_1_features.npz

Replace :code:`map4_v1_1` with :code:`map4` in all commands to use the
legacy/Molfeat-compatible definition.

Older development checkpoints can contain the recorded name :code:`map4` even
when their vectors came from the former native-1.1 adapter. That path used the
v1.1 shingle rule without the whole-molecule canonicalization applied now, so
it is not guaranteed to match :code:`map4_v1_1` numerically. Do not change only
the prediction flag; Chemprop intentionally rejects the name mismatch. Retrain
with :code:`map4_v1_1`, or migrate the checkpoint and feature provenance only
after independently verifying its original vectors.

Utils
-----

Classes and functions from `chemprop.features.utils.py <https://github.com/Kuroki-A/chemprop/blob/master/chemprop/features/utils.py>`_.

.. automodule:: chemprop.features.utils
   :members:
