# Changelog

## 1.7.1+kuroki.1 — 2026-08-20

This is the first explicitly versioned release of the Kuroki-maintained
Chemprop v1 line.

### Highlights

- Repaired and versioned the LightGBM pipeline. Bundles now contain the exact
  frozen MPN encoder, task-specific boosters, scalers, and metadata; regression,
  binary classification, multitask gaps, ensembles, and fresh-process
  prediction are supported.
- Made feature backends lazy and batch-aware. RDKit fingerprints/descriptors,
  Mordred, PaDEL, and Molfeat generators avoid repeated setup and support safe
  multiprocessing or native batching where applicable. Normalized RDKit CDFs
  are applied column-wise in batches, and offline generation now streams
  restartable chunks through disk-backed consolidation.
- Added Molfeat 2D/scaffold/pharmacophore generators: `fcfp`, `fcfp_count`,
  `topological`, `topological_count`, `layered`, `avalon_count`,
  `rdkit_count`, `atompair_count`, `pattern`, `estate`, `secfp`, `cats2d`,
  `scaffoldkeys`, and `pharm2d`; added direct `map4` and `map4_v1_1`
  implementations.
- Made `map4` a canonicalized, folded 2,048-bit implementation of the legacy
  MAP4 v1.0 semantics expected by Molfeat 0.11. The separately named
  `map4_v1_1` generator retains the canonicalized native map4 1.1.3 semantics;
  the two definitions are intentionally not bit-compatible.
- Added compatibility handling for Molfeat 0.11 and map4 1.1, whose upstream
  class rename would otherwise prevent every Molfeat fingerprint from loading.
- Refreshed the reproducible Python 3.10 environment to the newest compatible
  stable scientific stack and the official PyTorch 2.6.0 CUDA 12.4 wheel;
  pretrained Molfeat backends are now isolated in an explicit extra.
- Added feature manifests/checkpoint metadata with generator order,
  configuration, selected-column order, implementation hash, dependency
  versions, dimensions, and input hashes. External feature sidecars now verify
  exact SMILES order, auxiliary row counts, and cache/resume identity.
- Fixed validation-only hyperparameter optimization, resume handling, metric
  aggregation (including extra and quantile metrics), sklearn callback
  compatibility, invalid SMILES handling, cache isolation, weight validation,
  and test score output.
- Hardened the legacy Web interface with signed sessions, CSRF protection,
  owner-scoped database operations, loopback-only defaults, and disabled
  checkpoint uploads unless explicitly trusted.

### Compatibility notes

- The v1 CLI and `.pt` neural-network checkpoint format remain supported.
- New LightGBM `.pkl` bundles cannot recover old raw-Booster LightGBM files,
  because those files did not contain the random MPN encoder used during
  training. Retraining is required.
- Selected descriptor columns are now correctly applied to reaction
  reactants. A legacy reaction checkpoint trained with a selected-feature CSV
  may therefore have the old full-width vector and should be retrained.
- `erg` retains its historical integer output for checkpoint compatibility;
  use `erg_float` for the non-truncated RDKit ErG values.
- `rdkit_2d_all` remains dependent on the installed RDKit descriptor schema.
  New checkpoint and offline-feature metadata detect such schema drift.
- `map4` and `map4_v1_1` are different fingerprint definitions and must not be
  interchanged between training and prediction. Models produced by the earlier
  development adapter used map4 1.1 shingle ordering but not the new
  whole-molecule canonicalization, so numerical compatibility is not
  guaranteed. Retraining with the explicit new name is the safe migration
  path.
- Checkpoints are Python pickle-based and must be treated as trusted local
  artifacts. The Web interface rejects uploads by default.
- The optional loaded-dataset cache also contains trusted pickle data. Its
  directory must be user-owned, non-symlinked, and private (mode `0700`).
- External features, phase features, weights, constraints, and ordered
  atom/bond descriptor archives must contain exactly one entry per raw input
  row, even when `max_data_size` or missing-target filtering is used.
