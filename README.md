# DiCo-NLI English

Reversal-aware baseline and experiment scaffold for **SemEval 2027 Task 2,
DiCo-NLI — English (Track 1)**. It trains a Hugging Face sequence classifier
on the organizer's released English phrase pairs, then jointly decodes each
source-pair's two directions.

Taking over the project? Start with [the handoff](docs/HANDOFF.md), which records
the two completed Colab runs, current best result, known limitations and next steps.
Their small prediction/score snapshots and analysis reports are included in Git.

The task has four labels:

- `EQUIVALENCE` reverses to itself.
- `FORWARD_ENTAILMENT` reverses to `BACKWARD_ENTAILMENT`.
- `BACKWARD_ENTAILMENT` reverses to `FORWARD_ENTAILMENT`.
- `NEGATIVE_OTHER` is not part of the official consistency metrics.

The joint decoder chooses among equivalence/equivalence, forward/backward,
backward/forward, and negative/negative assignments by their summed logits. It
uses only public `pair_id`, phrase texts, and language fields, never gold labels
or `reverse_pair_id`. Two rows must actually swap their texts and languages to
be constrained. Missing pair IDs, singleton directions, and incomplete groups
fall back to independent argmax. Negative/negative is a modeling assumption,
**not** an official reversal rule or a guarantee of SoftCons credit.

## Quick start: macOS / Linux

Use Python 3.10+ (3.12 recommended). Run commands from the repository root:

```bash
make PYTHON=python3.12 setup
make fetch-data
make test
make lint
```

`make fetch-data` retrieves only Track 1's public train/dev CSV files and puts
the organizer's scorer in an ignored local cache. The fetch is pinned to the
official source revision recorded in `scripts/fetch_official_assets.py`; no
organizer data or third-party scorer code is committed to this repository.
Downloads are verified against that revision even when files already exist;
changed files cause an error unless explicitly replaced using `--force`.
SHA-256 provenance manifests are saved with the data and scorer.

Without Make, the equivalent setup is:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python scripts/fetch_official_assets.py --include-scorer
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src tests scripts
```

On Ubuntu/WSL, `sudo apt install python3-venv make git` installs the OS tools;
use its supported Python 3.10+ with `make PYTHON=python3 setup`. The Makefile
defaults to `python3`; on this Mac that is 3.9, so explicitly select 3.12.
Keep a separate Linux `.venv` in WSL rather than reusing a Windows environment.
For the exact versions validated in the local audit, optionally install with
`.venv/bin/python -m pip install -c requirements-lock.txt -e '.[dev]'`.
This constraints file records one tested Python 3.12 macOS environment; wheel
availability and CUDA builds still depend on the target system. Each run saves
its actual software versions regardless of whether constraints are used.

The default model is `microsoft/deberta-v3-base`, selected because the task's
published pilot reports its DeBERTa v3 encoder-only baseline as the strongest
of its representative models. A GPU is useful but not required; the training
script selects CUDA, then Apple MPS, then CPU by default. PyTorch 2.6+ is required
for compatibility with Transformers' loading checks for PyTorch weight files.

## Quick start: Windows PowerShell

Windows does not include GNU `make`, so run the PowerShell bootstrap from the
repository root instead. First install Python 3.10+ from
[python.org](https://www.python.org/downloads/); Python 3.12 is recommended.

```powershell
py -3.12 --version
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1
.\.venv\Scripts\python.exe -m pytest
```

The bootstrap creates `.venv`, installs all dependencies, and fetches the
public English train/dev data plus the official scorer. To use an already
available Python executable instead of the Windows Python launcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1 -Python python
```

Train from PowerShell with:

```powershell
.\.venv\Scripts\python.exe -m dico_nli.train --train-file data/raw/train/dico_nli_train_track1_participant_labeled.csv --dev-file data/raw/dev/dico_nli_dev_track1_participant_labeled.csv --reference-file data/raw/dev/dico_nli_dev_track1_reference.csv --output-dir results/experiments --decode independent --selection-metric weighted_f1
```

If PowerShell reports that script execution is disabled, use the command above
exactly; `-ExecutionPolicy Bypass` applies only to that invocation and does not
change the machine-wide policy.

## First actual training experiment

The first controlled experiment is DeBERTa-v3-base on the official English
training split (3,042 rows), evaluated on the official development split (660
rows). Development labels are used for checkpoint selection and reporting only.
Use independent development Weighted F1 for checkpoint selection, then compare
independent and joint decoding on **the same selected checkpoint**:

```bash
.venv/bin/dico-train \
  --train-file data/raw/train/dico_nli_train_track1_participant_labeled.csv \
  --dev-file data/raw/dev/dico_nli_dev_track1_participant_labeled.csv \
  --reference-file data/raw/dev/dico_nli_dev_track1_reference.csv \
  --output-dir results/experiments \
  --model-name microsoft/deberta-v3-base \
  --model-revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
  --seed 42 --epochs 8 --patience 3 --batch-size 16 --max-length 128 \
  --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.1 \
  --class-weight balanced --decode independent --selection-metric weighted_f1
```

Alternatively: `make train-dev TRAIN_ARGS="--decode independent --selection-metric weighted_f1"`.
`make train-dev` always optimizes on `TRAIN` and evaluates on `DEV`; it does not
train on development examples. Its historical defaults (`joint` and `mean3`)
are still supported. `mean3` is our checkpoint-selection heuristic, **not an
official combined competition score**. This protocol compares both decoders
without assuming that joint decoding improves performance.

Each invocation creates `results/experiments/<UTC-time>-<unique-id>/`. An explicit
`--output-dir` also names a parent directory, and a new run subdirectory is
always created; previous runs are never reused. The console prints its path.

```text
run_config.json              All CLI settings, resolved model revision, input
                             hashes/upstream provenance, Git SHA/dirty state,
                             hardware, seed, checkpoint path
source_snapshot/             Source as executed, including uncommitted changes
software_versions.json       Installed Python package versions
training.log                 Training progress, metrics, errors
status.json                  running / completed / failed (with failure reason)
split_audit.json              Source-ID and exact-text overlap audit
metrics_history.json          Development metrics per epoch
best_metrics.json             Selected epoch and selection metrics
best_model/                   Reloadable weights, tokenizer, inference settings
dev_logits.npy                Selected-checkpoint logits, in participant row order
predictions_dev_independent.csv / predictions_dev_joint.csv
official_independent/ / official_joint/    Actual organizer scores.json/scores.txt
dev_metrics_independent.json / dev_metrics_joint.json    Local parity checks
predictions_dev.csv / dev_metrics.json      Aliases for the requested --decode
```

Final evaluation automatically invokes the unmodified official scorer for both
decoders and fails if its three scores differ from local scores. Fetch it
before training. When training without `--reference-file`, explicitly select
`--selection-metric weighted_f1`; unavailable consistency scores are `null`,
and official metrics cannot be produced.

Compare completed actual runs (smoke runs are excluded):

```bash
.venv/bin/python -m dico_nli.compare
.venv/bin/python -m dico_nli.compare --diff RUN_ID_A RUN_ID_B
```

The table includes both decoders, scores, model/revision, hyperparameters, seed,
Git state, dataset hashes, and a linkable configuration path. Winners for each
metric are reported separately within matching train/dev dataset hash groups.
Use the saved configuration diff to inspect every change between runs. Results
and snapshots stay ignored by Git; copy the whole run folder to persistent
storage when using temporary Colab disks.

The two completed seed-42 runs have curated, immutable evidence snapshots under
[`docs/experiments/`](docs/experiments/README.md). To verify their saved predictions
and regenerate the error analyses on CPU, run `make analyze-saved` after fetching
the official assets. This does not train a model or require model weights.

## Prediction and scoring

Predict an organizer-supplied unlabeled test file:

```bash
.venv/bin/dico-predict \
  --model-dir results/experiments/RUN_ID/best_model \
  --input-file path/to/dico_nli_test_track1_participant_unlabeled.csv \
  --output-file results/test_predictions.csv
```

The output has exactly the required `instance_id,label` columns and preserves
input IDs and row counts. Labels, reference files, and reverse-pair links are
not required. Saved training sequence length is reused by default. Existing
prediction files are protected unless `--force` is supplied. For local
development scoring, use either the built-in compatible scorer or the cached
organizer scorer:

```bash
.venv/bin/dico-score \
  --reference-file data/raw/dev/dico_nli_dev_track1_reference.csv \
  --prediction-file results/experiments/RUN_ID/predictions_dev_joint.csv \
  --official-cache .cache/official_dico_nli
```

## Experiment notes

- Keep the released split intact. It is grouped by source phrase pair to avoid
  directional leakage.
- Training rejects overlapping instance/source-pair IDs. The pinned organizer
  release nevertheless repeats four exact unordered text pairs under different
  source IDs across train/dev; this is warned about and recorded without
  modifying the official data. See [the audit](docs/PRETRAINING_AUDIT.md).
- `--decode independent` is useful as an ablation. `--decode joint` is the
  default and uses `pair_id` only at inference, never gold labels.
- `--selection-metric mean3` treats Weighted F1, SoftCons, and HardCons as a
  three-score profile. Select `weighted_f1`, `soft_cons`, or `hard_cons` for a
  targeted ablation.
- All test gold labels remain out of this repository. Do not use hidden labels
  to choose models or hyperparameters.

## The three official metrics

| Metric | Interpretation |
| --- | --- |
| Weighted F1 | Support-weighted item-level F1 across all four labels. |
| SoftCons | Fraction of reciprocal **reversible gold pairs** whose predictions obey E↔E or F↔B; ignores whether the predictions are correct. |
| HardCons | Fraction of those gold pairs with both directions predicted correctly. |

`NEGATIVE_OTHER` gold items participate only in item-level F1. Predicting
negative/negative on a reversible gold pair earns neither consistency measure.
High SoftCons alone does not imply high accuracy. The three scores are reported
separately by the competition; our `mean3` does not replace them.

The old `results/smoke/perfect_predictions.csv` simply copies public dev gold
labels to validate scoring. Its 1.0 scores are **not model performance**.
The new end-to-end CPU test uses a tiny randomly initialized local BERT and
synthetic examples, marks its run `cpu_smoke`, and is also not task performance.
Tests do not download models or require a GPU; integration tests use the small
pinned official scorer fetched by `make fetch-data` (otherwise those checks skip).

## Device readiness

For Google Colab, select a GPU runtime, clone this repo, install with
`python -m pip install -e '.[dev]'`, fetch assets, and run the training module
using `python` instead of `.venv/bin/python`. First confirm
`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
Use `--device cuda` to require a GPU rather than silently run on CPU. GPU use
requires a compatible PyTorch build and NVIDIA driver; see
[PyTorch installation](https://pytorch.org/get-started/locally/) and
[Microsoft's WSL CUDA documentation](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).
CUDA/Colab performance has not been tested on this Mac.

`--device cpu` is fully covered by the synthetic training/reload test. MPS is
available on the audit machine; it is selected automatically when CUDA is absent.
If device memory runs out, reduce `--batch-size` (e.g. 4 or 1) or sequence length.
Failed runs keep logs/status rather than appearing as completed experiments.
Seeds cover Python, NumPy, and Torch. `--deterministic` additionally requires
deterministic Torch kernels and may fail on unsupported operations. Matching
seeds do not promise identical results across devices/software versions.

## Layout

```text
src/dico_nli/       Training, inference, validation, joint decoding, scoring
scripts/            Pinned asset fetcher and Windows bootstrap
tests/              Unit tests, official scorer parity, tiny offline CPU run
AGENTS.md           Repository-specific development instructions
docs/               Handoff, audits, analyses, and curated experiment evidence
requirements-lock.txt  Optional validated environment constraints
data/raw/           Ignored downloaded Track 1 data
results/, models/   Ignored experiment artifacts
```

## Sources

Task materials and metric definitions come from the official
[DiCo-NLI repository](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI)
and [task site](https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/).
The official evaluator remains the authority for submissions.
