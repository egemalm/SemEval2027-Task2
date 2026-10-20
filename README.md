# DiCo-NLI English

Reversal-aware baseline and experiment scaffold for **SemEval 2027 Task 2,
DiCo-NLI — English (Track 1)**. It trains a Hugging Face sequence classifier
on the organizer's released English phrase pairs, then jointly decodes each
source-pair's two directions.

The task has four labels:

- `EQUIVALENCE` reverses to itself.
- `FORWARD_ENTAILMENT` reverses to `BACKWARD_ENTAILMENT`.
- `BACKWARD_ENTAILMENT` reverses to `FORWARD_ENTAILMENT`.
- `NEGATIVE_OTHER` is not part of the official consistency metrics.

The joint decoder considers only coherent two-row assignments: equivalence in
both directions, either entailment orientation, or negative/other in both
directions. This makes the submitted labels compatible with the directional
structure whenever a complete `pair_id` is present, while still allowing the
model to identify negative pairs.

## Quick start: macOS / Linux

The organizer requires Python 3.10+. This project was set up with Python 3.12
on macOS.

```bash
make setup
make fetch-data
make test
make train-dev
```

`make fetch-data` retrieves only Track 1's public train/dev CSV files and puts
the organizer's scorer in an ignored local cache. The fetch is pinned to the
official source revision recorded in `scripts/fetch_official_assets.py`; no
organizer data or third-party scorer code is committed to this repository.

The default model is `microsoft/deberta-v3-base`, selected because the task's
published pilot reports its DeBERTa v3 encoder-only baseline as the strongest
of its representative models. A GPU is useful but not required; the training
script selects CUDA, then Apple MPS, then CPU by default.

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
.\.venv\Scripts\python.exe -m dico_nli.train --train-file data/raw/train/dico_nli_train_track1_participant_labeled.csv --dev-file data/raw/dev/dico_nli_dev_track1_participant_labeled.csv --reference-file data/raw/dev/dico_nli_dev_track1_reference.csv --output-dir results/deberta-v3-base
```

If PowerShell reports that script execution is disabled, use the command above
exactly; `-ExecutionPolicy Bypass` applies only to that invocation and does not
change the machine-wide policy.

## Commands

Train while selecting the checkpoint using the mean of the three official
development measures:

```bash
.venv/bin/dico-train \
  --train-file data/raw/train/dico_nli_train_track1_participant_labeled.csv \
  --dev-file data/raw/dev/dico_nli_dev_track1_participant_labeled.csv \
  --reference-file data/raw/dev/dico_nli_dev_track1_reference.csv \
  --output-dir results/deberta-v3-base
```

The run directory contains `best_model/`, a valid `predictions_dev.csv`,
per-epoch JSON metrics, and `run_config.json`.

Predict an organizer-supplied unlabeled test file:

```bash
.venv/bin/dico-predict \
  --model-dir results/deberta-v3-base/best_model \
  --input-file path/to/dico_nli_test_track1_participant_unlabeled.csv \
  --output-file results/test_predictions.csv
```

The output has exactly the required `instance_id,label` columns. For local
development scoring, use either the built-in compatible scorer or the cached
organizer scorer:

```bash
.venv/bin/dico-score \
  --reference-file data/raw/dev/dico_nli_dev_track1_reference.csv \
  --prediction-file results/deberta-v3-base/predictions_dev.csv \
  --official-cache .cache/official_dico_nli
```

## Experiment notes

- Keep the released split intact. It is grouped by source phrase pair to avoid
  directional leakage.
- `--decode independent` is useful as an ablation. `--decode joint` is the
  default and uses `pair_id` only at inference, never gold labels.
- `--selection-metric mean3` treats Weighted F1, SoftCons, and HardCons as a
  three-score profile. Select `weighted_f1`, `soft_cons`, or `hard_cons` for a
  targeted ablation.
- All test gold labels remain out of this repository. Do not use hidden labels
  to choose models or hyperparameters.

## Layout

```text
src/dico_nli/       Training, inference, validation, joint decoding, scoring
scripts/            Pinned organizer-asset fetcher
scripts/bootstrap_windows.ps1  Windows setup without GNU Make
tests/              Fast, model-free checks of task-critical logic
data/raw/           Ignored downloaded Track 1 data
results/, models/   Ignored experiment artifacts
```

## Sources

Task materials and metric definitions come from the official
[DiCo-NLI repository](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI)
and [task site](https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/).
The official evaluator remains the authority for submissions.
