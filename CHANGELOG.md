# Changelog

## 1.7.1+kuroki.2 — 2026-09-08

This maintenance release follows a second end-to-end correctness, security,
packaging, and CI audit of the Kuroki Chemprop v1 fork.

### Highlights

- Corrected legacy Web prediction handling for empty and partially invalid
  SMILES input, removed stale/colliding uploaded prediction files, validated
  GPU and form values, and prevented failed training runs from leaving empty
  checkpoint records.
- Closed an external-referrer redirect and made uploaded status messages robust
  to malformed query data. Database name collision retries now propagate
  foreign-key and other non-unique integrity failures instead of looping.
- Fixed invalid Web template markup and stale local/SSH startup instructions.
- Repaired the Docker build order for the editable install, updated micromamba,
  and reduced accidental build-context inclusion of local artifacts.
- Migrated CodeQL from the unsupported v2 action to v4, updated the official
  checkout/setup-python actions, applied least-privilege checkout credentials,
  and added documentation and source-distribution checks to CI. Push/PR CI now
  runs the full unit suite and a bounded real train/predict smoke test instead
  of an unbounded collection of multi-fold and Hyperopt integration studies.
- Added packaging and lint tooling to the reproducible development environment.
- Documented the 2026-10-04 Python 3.10 end-of-life boundary while retaining
  the agreed closed environment for reproducibility.
- Replaced quadratic all-atom-pair bond traversal with sparse RDKit bond
  traversal, reducing graph-featurization work from `O(V^2)` to
  `O(V + E log E)` while preserving the legacy directed-bond order.
  Molfeat and fixed-fingerprint selected-column handling, deterministic MAP4
  seeding, and MAP4 intermediate-memory use were also corrected.
- PaDEL failures now stop feature generation with the row, SMILES, and root
  cause instead of silently inserting an all-zero descriptor vector.
- Hardened external feature/descriptor alignment and validation: pickle
  atom/bond descriptors are reindexed by unambiguous SMILES, SDF descriptor
  columns are detected across all rows, non-finite inputs are rejected or
  normalized as documented, and constraint CSVs reject duplicate headers and
  non-numeric/non-finite values.
- Feature-generation resume now rejects legacy temporary directories without
  an identity manifest and compares the ordered-SMILES encoding as well as its
  digest before reusing chunks.
- Documented the arbitrary-code-execution risk of pickle-based external
  feature and atom/bond descriptor inputs, split indices, Hyperopt trials, and
  the safer NumPy/CSV formats.
- Remote Web state must now be user-owned and private (mode `0700`), and the
  deployment guide explicitly requires a single worker/thread for its
  process-local progress and prediction state. The built-in development server
  likewise disables threading and the debug reloader.
- LightGBM training now requires ``--features_only`` with deterministic
  generated or external features; this prevents silently fitting boosters to
  an untrained random MPN representation. Prediction also rejects existing
  versioned bundles recorded with ``features_only=False`` and requires them to
  be retrained.
- Hardened uncertainty prediction and calibration: ensemble model/scaler count
  mismatches are rejected, MC dropout no longer mutates models permanently,
  invalid calibration option combinations fail fast, small/missing conformal
  samples are handled explicitly, and regression/multiclass/spectral
  uncertainty metrics use the correct masks and finite arithmetic.
- Corrected T-scaling to pass a standard deviation (not a variance) to the
  Student-t likelihood, and corrected MVE weighting to apply weights on the
  task axis and reshape predictions against the prediction dataset. Z-, T-,
  and Zelikman scaling now reject unobserved tasks and non-positive or
  non-finite variances instead of persisting a NaN calibration factor.
- Made uncertainty calibration/evaluation work with variable-length atom and
  bond targets under NumPy 2. Task counts, masks, model axes, and every
  per-molecule boundary are now checked, including conformal and Platt paths.
- Applied dtype-aware positive floors to MVE variance and evidential
  lambda/alpha/beta parameters, preventing extreme negative logits from
  producing zero denominators or non-finite losses and uncertainty values.
- Made the Noam scheduler finite for zero warmup, zero-epoch evaluation, and
  short runs whose requested warmup is longer than training; malformed epoch,
  step, and learning-rate settings now fail early.
- Added argument validation for neural hidden sizes/depths, FFN layers,
  dropout, learning rates, gradient clipping, cache thresholds, loss
  coefficients, and split indices. ``--test`` now requires an existing
  checkpoint source while explicit ``--epochs 0`` compatibility is retained.
- Configuration-file overrides are now applied before derived argument state
  is computed, and malformed top-level JSON, unknown/internal keys, and
  attempts to replace ``config_path`` are rejected instead of being silently
  accepted. Config values can no longer bypass CLI types, enumerated choices,
  feature-generator names, or GPU selection. Interpret, Hyperopt, and sklearn
  estimator bounds and counts are also validated.
- StandardScaler now rejects ragged, width-mismatched, infinite, or invalid
  checkpoint state and handles entirely missing feature columns without
  emitting NumPy warnings.
- FFN training now rejects empty training/validation splits, tasks with no
  observed training labels, and invalid class-balanced splits, and fails fast
  when its primary validation metric or mini-batch loss is non-finite.
- Replaced the non-differentiable multiclass MCC ``argmax`` loss with a soft
  confusion-matrix formulation and made degenerate MCC scoring return a finite
  result instead of ``NaN``.
- Tightened dataset target validation for multiclass/non-finite values, fixed
  missing-label handling for atom-level class sizes, and made ordinary FFN
  checkpoint loading reject missing or shape-mismatched weights unless partial
  loading is explicitly requested.
- Fixed GPU bond-descriptor placement so CUDA tensor indices are never used to
  index NumPy arrays, preserved `bias=False` for the solvent MPN, and validated
  molecule/descriptor scopes before MPN and constrained-FFN computation.
  Empty-bond molecules and inconsistent atom/bond output shapes now fail with
  a targeted error rather than being truncated or misassigned.
- CSV readers now reject empty/blank/duplicate headers and short or overlong
  rows before `DictReader` or pandas can hide them, handle UTF-8 BOMs, and
  validate SMILES row width. Empty invalid-SMILES inputs are handled safely;
  selected-feature CSVs likewise reject blank or duplicate generator names.
- Corrected scaffold/time-window split index spaces and output locations,
  overlap detection for multiple SMILES columns, paired Wilcoxon aggregation,
  Welch one-sided direction, HDF5 resource handling, and several script CSV
  alignment/empty-input cases. Morgan similarity now computes each fingerprint
  once and uses RDKit bulk Tanimoto operations.
- Restored collection of a silently overwritten data-feature unit test,
  exported `model_fingerprint` through the public train API, and removed
  remaining meaningful undefined/unused/redefinition static-analysis findings.
- Refreshed the corrected T-scaling and current-stack reaction/solvent
  integration baselines, fixed a misleading atom/bond test label, and made
  prediction integration tests deterministic on restricted runners by using
  zero multiprocessing workers.

### Compatibility notes

- Corrected T-scaling, MVE weighting, atom/bond uncertainty, statistical-test,
  and time-window split behavior can change numerical outputs. Re-run affected
  uncertainty reports, statistical comparisons, and generated split files.
- CSV files with duplicate/blank headers or ragged rows that were previously
  interpreted by truncation or pandas column rewriting are now rejected and
  must be corrected at the source.
- A task with no calibration observations, or an uncertainty source with
  non-positive/non-finite variance, cannot be multiplicatively calibrated and
  now raises `ValueError` instead of producing a non-finite scaler.
- Class-balanced mini-batch sampling is explicitly limited to a single binary
  target with at least one observed member of each class; the former
  any-positive multitask grouping was ambiguous and is no longer accepted.

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
