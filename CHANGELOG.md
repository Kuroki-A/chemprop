# Changelog

## 1.7.1+kuroki.4 — 2026-09-10

### Added

- Added `chemprop_train --class_weight balanced` for single-task binary FFN
  BCE training. Weights are resolved only from post-split training labels,
  exclude missing labels from the counts, retain every row in the training
  loader, and are stored with checkpoint metadata.
- Added `--train_on_full_data` for validation-free, fixed-epoch final FFN fits.
  It fits scalers on all eligible training rows, trains every ensemble member
  for the requested schedule, saves last-epoch checkpoints, and can evaluate a
  strictly held-out external test set without using it for model selection.

### Changed

- Separate validation/test loading no longer inherits `--data_weights_path`.
  Documented that the complete weights file is normalized before filtering and
  splitting and that the retained training weights are not renormalized. A
  retained training split is now rejected if any task's observed labels all
  have zero row weight.
- Prediction, uncertainty evaluation/calibration, and fingerprint inputs no
  longer inherit a training-only `data_weights_path` stored in checkpoint
  arguments. LightGBM and sklearn external validation/test sets follow the
  same isolation rule, including public prediction-argument updates.
- Sklearn training now rejects the previously ignored `--class_balance`
  option and directs users to estimator-native `--class_weight balanced`.
- Restored Chemprop v1 multitask `--class_balance` compatibility (a row is in
  the positive sampling group when any observed task is active), while keeping
  strict binary-target and non-empty-group validation.
- Normal FFN training rejects validation data with no observed labels and
  retains its fail-fast behavior when the primary validation metric is
  undefined, so invalid validation data cannot select a checkpoint. Undefined
  aggregate or per-task values are presented as `not evaluated` and are not
  written to TensorBoard as numeric observations. Hyperopt also rejects
  zero-epoch trials that would otherwise rank untrained initializations.
- LightGBM retains the same fail-fast primary-validation contract; undefined
  auxiliary aggregate values are presented as `not evaluated`, and HPO rejects
  non-finite trial objectives.
- Noam scheduling now uses the effective loader batch count for
  class-balanced sampling, matching fixed-epoch final training while leaving
  the default unbalanced training schedule unchanged.
- Full-data external-test evaluation preserves Python, NumPy, and Torch RNG
  state between ensemble members; undefined saved metrics use JSON `null`, and
  full-data output requires a fresh `fold_0` to prevent stale artifacts. A
  non-empty unlabeled external test file can still produce `--save_preds`
  output without being treated as an evaluated test set.
- Saved split-index files keep the positional `[train, validation, test]`
  contract when validation is deliberately disabled.
- Spectra normalization now accepts NumPy-array inputs used by prediction
  evaluation without ambiguous truth-value errors.
- Conformal quantile calibration and uncertainty evaluation now load each
  observed CSV target once instead of duplicating lower/upper model output
  names, preventing target/prediction shape mismatches.

## 1.7.1+kuroki.3 — 2026-09-09

This production-readiness maintenance release follows a final repository-wide
audit of checkpoint evaluation and transfer, generated features, LightGBM,
atom/bond uncertainty, external splits, Web storage, and packaging.

### Highlights

- Made `chemprop_train --test` a true no-optimization checkpoint evaluation.
  It restores the exact saved architecture and target/input scalers, validates
  every ensemble member's scaler contract and data semantics, avoids dependence
  on training-split labels, and preserves supplied checkpoints byte-for-byte.
  It also rejects frozen-transfer options which would make evaluated and saved
  models disagree.
- Reworked warm starts and frozen transfer. Complete compatible encoders are
  copied into the requested current architecture, non-encoder state is reused
  only when shape-compatible, and frozen values and gradient flags are applied
  only after complete validation. Multi-encoder, reaction/solvent,
  `features_only`, shared atom/bond FFN aliases, and PReLU edge cases now fail
  safely or map deterministically.
- Unified scalar, native-batch, runtime, and `scripts/save_features.py`
  molecular-feature semantics. Inputs are canonical and atom-map-independent;
  offline reactions use reactants; hydrogen-only rows use typed zero vectors;
  generators and duplicate calculations are reused across bounded batches and
  worker processes. Morgan, count-Morgan, RDKit, and AtomPair fingerprints now
  use affinity-bounded RDKit native batches at runtime while offline generation
  retains its faster default SMILES-parsing process pool.
- Reduced peak memory during runtime feature generation by releasing unused
  reaction products, generator temporaries, unique-molecule maps, and each
  source row as soon as its final vector is materialized. Empty offline inputs
  now fail before creating artifacts when a custom generator has no discoverable
  width, while fixed-width generators produce a valid empty archive.
- Replaced the overly broad per-module built-in feature hash with metadata
  schema 2 and a targeted semantic revision per generator. Dependency versions,
  configuration, selected-column order, width, dtype, and source provenance
  remain independently checked; custom/plugin generators retain conservative
  full-module hashing.
- Completed provenance and dependencies for Pharm2D and pretrained Molfeat,
  corrected normalized Descriptastorus parity for explicit-H RDKit molecules,
  and made selected PaDEL schemas available before the first Java calculation.
- Added LightGBM guardrails for untrained MPN representations, reaction
  generators, ignored atom/bond inputs, bundle-owned scalers, incompatible
  ensembles, and unsupported uncertainty options. Versioned bundles retain the
  exact deterministic representation, per-task boosters, and scalers; loading
  now cross-checks target/feature scaler presence and widths, Booster objectives
  and widths, task counts, and redundant encoder metadata before prediction.
- Corrected constrained atom/bond prediction and uncertainty calibration.
  Prediction and calibration constraint files are now distinct and validated
  for task order, rows, numeric finiteness, scaling, device, and dtype.
- Made atom/bond uncertainty and ensemble aggregation safe for variable atom
  and bond counts under NumPy 2, added stable online aggregation, and serialized
  variable-length CSV values as JSON arrays. Classification uncertainty now
  validates every member's class-count shape and values and rejects mixed
  legacy/new ensembles instead of crashing or biasing Bayesian priors.
- Hardened predetermined, index-based, and external cross-validation splits
  against invalid, duplicate, out-of-range, overlapping, and silently omitted
  indices. Fold index zero is handled correctly.
- Anchored pickle-based dataset-cache reads and atomic writes to validated
  descriptor-relative POSIX paths, rejecting symbolic links in every path
  component and directory-replacement races before deserialization.
- Corrected `last_FFN` and reaction/solvent fingerprint widths, restored
  one-molecule MPN fingerprint extraction from shared multi-molecule encoders,
  and explicitly rejected unsupported atom/bond fingerprint export.
- Hardened Web state cleanup against containment escape, symlink traversal,
  and directory-replacement races. Mutable Web state now defaults outside the
  source tree, and database/data/checkpoint reset uses the same safe deletion
  primitive.
- Updated CI and reproducible build declarations to Setuptools 84.x and Wheel
  0.48.x, added missing optional feature dependencies, made Web assets explicit
  package data, and made the source distribution include its documented
  environment, CLI wrappers, scripts, local documentation, and logo without a
  misleading partial test subset. Focused regression coverage checks all
  changes above; CI now exercises both FFN and LightGBM command-line round trips
  and inspects the corresponding release artifacts.

### Compatibility notes

- Recompute scores previously produced by `chemprop_train --test` for
  regression or scaled inputs. Earlier behavior could fit scalers to evaluation
  data or silently run a checkpoint whose required scaler was absent.
- Feature metadata schema 1 and 2 intentionally have a one-time boundary.
  Checkpoints which generate features at prediction time must be retrained, and
  interrupted schema 1 `save_features` jobs require `--restart` once. A
  completed schema 1 `.npz` plus its original manifest remains usable as
  materialized external features when the exact files are retained.
- Regenerate/retrain features affected by canonicalization, atom-map removal,
  reaction-reactant handling, or hydrogen-only policy changes. `map4` and
  `map4_v1_1` remain different definitions and cannot be interchanged.
- Legacy LightGBM models built from an untrained random MPN, a reaction
  molecular generator, or ignored atom/bond descriptors are rejected and need
  retraining with deterministic, explicit molecule-level features.
- Frozen-transfer runs which previously froze random or partially copied
  values should be retrained. Newly ambiguous or incomplete mappings are
  rejected instead of approximated.
- Constrained atom/bond calibration requires both `--constraints_path` for
  prediction rows and `--calibration_constraints_path` for calibration rows.
  Downstream CSV readers should parse atom/bond vectors as JSON.
- Web state formerly kept below the checkout is not moved automatically. Use
  the private default `~/.chemprop-web`, `CHEMPROP_WEB_ROOT`, or an explicit
  service-owned `--root_folder` with mode `0700`.
- Updating the checkout does not update an existing conda environment. Recreate
  it, or update it from `environment.yml` and verify dependencies before
  building this release.

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
- Resolved all findings from the pull-request CodeQL quality gate, including
  request-derived redirect data, explicit control-flow initialization, and
  nested-loop variable shadowing.
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
