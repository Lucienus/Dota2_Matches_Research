# Dota 2 win prediction: reproducible evaluation

Reconstruct match prefixes, retrain selected models, evaluate the frozen holdout,
and generate the numerical tables for *Model Representation and Training-Sample
Sensitivity in Prefix-Based Dota 2 Win Prediction*.

The workflow connects frozen match IDs and input hashes to model predictions,
statistical comparisons, and LaTeX tables. In an independent environment, all
16 reconstructed input/label files matched their original hashes, and all
18 numerical tables and result macros matched the paper's displayed values.
Across 536 prediction vectors, the largest probability difference from the
original run was 9.44 × 10⁻¹⁶. See `VALIDATION_ZH.md` and `validation.json` for the
execution checks and comparison results.

**Version:** `access-reproduction-2026-11-16`  
**Release status:** local package awaiting publication by the author.

## Package contents

| Location | Purpose |
|---|---|
| Root-level Python entry points | Input reconstruction, selected-model training, holdout prediction, scoring, and table export |
| `manifests/` | Frozen IDs, hashes, final model configurations, seeds, and exclusion records |
| `reference_tables/` | Aggregate results from the separate sampling and cost experiments |
| `study_sources/` | Original scripts used for the paper's search, audit, sampling, cost, and figures |
| `CODE_SELECTION.json` | Original runtime modules required by the public workflow |
| `validation.json`, `validation_evidence/` | Independent-environment results and numerical comparisons |

The scripts in `study_sources/` document the research implementations and require
their historical workspace inputs and paths. Use the root-level entry points for
the verified reproduction workflow below.

Author-written code is released under MIT, copyright Luchang Yang. Metadata retain
their third-party terms; see `LICENSE_SCOPE.md`.

## Environment

The verified configuration uses Python 3.12.14 on Windows, CUDA 12.8, and an
RTX 4070 SUPER. Create a new virtual environment with the recorded dependencies:

```text
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements-recorded.txt
.venv/Scripts/python -m pip check
```

For other platforms, activate the environment and use its `python`. The complete
installed package list is recorded in `requirements-full.txt`. The validation
covers installation, reconstruction, inference, scoring, table generation, and a
complete Gold/XP LR training run; full-model retraining uses the supplied entry
point and recorded configurations.

## Data and model access

The package includes frozen match IDs, raw and feature hashes, configurations,
and aggregate source tables. Raw match JSON, NPZ feature packages, checkpoints,
and individual prediction vectors are obtained separately.

To request original snapshots or derived artifacts, contact Luchang Yang at
luchangyang@hbun.edu.cn and specify the files and intended use. Access depends on
file availability and applicable source-data terms; provision is not guaranteed.

Exact reconstruction requires matching raw snapshots. OpenDota API responses may
change or become unavailable. The acquisition log records these differences, and
the reconstruction entry point stops on a raw-hash mismatch. A hash identifies a
snapshot but cannot restore missing content. Evaluate changed API records as a
separately labelled reconstruction, retaining the original reference hashes.

## 1. Verify files and retrieve frozen match IDs

```text
python release.py verify
python fetch_frozen.py --cohort development --out raw/development --limit 3
```

The acquisition command previews the requested records. Add `--download` to fetch
them and set an explicit positive `--limit` for the desired portion of the frozen
list. For holdout records, use `--cohort holdout --out raw/holdout`.

Set `OPENDOTA_API_KEY` in the environment to use an API key. Credentials stay out
of the package and request log. Retrieval preserves the selected match IDs and
records each available response's hash; unavailable records receive no substitute.

## 2. Rebuild development inputs and train selected models

```text
python rebuild_inputs.py --raw-dir raw/development --scope broad --minute 20 --out inputs/broad/20m
python release.py train --data inputs/broad/20m --scope broad --minute 20 --model G --seed 111001 --device cuda --out models/broad/20m/G/seed111001
```

Repeat for `broad`, `listed`, and `broad_matched` at 10, 20, 30, and 40 minutes.
`manifests/model_plan.json` specifies the 68 model–scope–cutoff combinations and
their seeds. The runner verifies input hashes and loads the actual final-fit
configuration for each seed. Constants and LR use one seed; stochastic models
use ten. Select `--device cpu` for conventional models and a new output directory
for each fit. Training outputs report development-validation performance;
holdout scoring takes place in step 5.

The model plan preserves the settings used by individual final fits. The
search-parameter CSV records the final search budget, including a 10,000-tree
ceiling. Some fits had already reached early stopping under a 2,000-tree cap and
were reused. These records distinguish the search allowance from the settings
used to produce each checkpoint.

## 3. Index local inputs and checkpoints

```text
python make_indexes.py --data-root inputs --models-root models --out indexes
```

The command produces two JSON dictionaries and a missing-file report:

- Training keys: `scope/minute`, with `train` and `prior` paths.
- Model keys: `scope/minute/model/seed`, with a checkpoint path.

The indexes can point to existing artifacts in other local directories. Keep
these machine-specific paths local when uploading the release.

## 4. Rebuild the holdout and predict

```text
python pipeline.py rebuild-holdout --raw raw/holdout --training-index indexes/training_index.json --run outputs/evaluation
python pipeline.py predict --run outputs/evaluation --training-index indexes/training_index.json --checkpoints indexes/checkpoint_index.json --device cuda
```

Reconstruction checks raw records and training priors, applies development-match
and shared-series exclusions, and enforces common prefix eligibility. Labels are
stored separately from inference tensors. The 16 resulting array files are
checked against the original hashes.

Prediction verifies input and checkpoint identities and operates on label-free
inputs. Add `--allow-retrained` when using newly trained checkpoints; the output
records this choice. Evaluation uses the same frozen holdout population for both
original-checkpoint replay and retrained-model comparisons.

## 5. Score models and generate paper tables

```text
python pipeline.py summarize --run outputs/evaluation
python pipeline.py tables --run outputs/evaluation
```

Scoring checks the complete 68-cell, 536-vector plan and exact match-ID/seed
pairing. It writes model metrics, primary G−N estimates, scope and category
contrasts, block sensitivities, and common-cohort trajectories to `reports/`.
The primary decision follows the computed scores and declared margins; table
export supports every resulting decision.

LaTeX outputs appear in `tables/`. The listed-category row is calculated from the
current scored results, with all four original categories retained in the SI.
Sampling and cost tables draw on the separate frozen supplementary experiments
in `reference_tables/`. `table_sources.json` identifies both the current scored
reports and those supplementary sources. When changing models or data, rerun the
corresponding supplementary analyses before updating their tables and claims.
Their original implementations are retained in `study_sources/`.
