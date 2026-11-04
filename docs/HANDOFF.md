# Project handoff

Start here when taking over SemEval 2027 Task 2 DiCo-NLI, English Track 1.
The repository contains a working fine-tuning/prediction/evaluation pipeline,
development rules, verified evidence from two real Colab T4 runs, and analyses.
There is no need to reconstruct the project state from the chat.

## Current result

| Training | Decode | Seed | Weighted F1 | SoftCons | HardCons | Correct / 660 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Balanced | Independent | 42 | 84.52% | 88.81% | 85.20% | 556 |
| Balanced | Joint | 42 | 83.84% | 90.61% | 85.92% | 551 |
| Unweighted | Independent | 42 | 84.26% | 88.09% | 83.75% | 553 |
| Unweighted | Joint | 42 | **84.98%** | **90.61%** | **85.92%** | **558** |

Unweighted + joint is the best observed configuration. The improvement over the
previous best is only 0.46 F1 points and two net correct answers. It is based on
one seed and is not established as a reliable improvement or a leaderboard rank.
These are public development results; no competition submission has been made
through this workflow.

The user ran both experiments in Colab on a Tesla T4. Their uploaded predictions
were independently checked against the official scorer, and saved logits
reproduce both decoding modes. The checkpoint weights were not uploaded for
analysis; GPU inference itself was not independently repeated locally.

## What has been built

- Pinned official English train/dev assets and official evaluator download.
- Four-label DeBERTa-v3 sequence-classifier fine-tuning, early stopping and
  checkpoint selection on independent development Weighted F1 for these runs.
- Independent and joint decoding from the same selected checkpoint's logits.
- Prediction without gold labels, gold references or reverse-pair metadata.
- Strict prediction IDs/format, split-overlap checks, official scorer parity.
- Unique run folders with config, provenance, source hashes/snapshot, logs,
  checkpoints, predictions, scores, hardware and software information.
- File-based experiment comparison, CPU tests, CI definition, Windows and WSL
  setup guidance, and repository-specific [AGENTS.md](../AGENTS.md).
- A fix for the `_Tee.close` shutdown exception, with regression tests. A retained
  third-party logger can now close/flush its stream after the training log ends
  without closing the console or accessing the closed file. The old error occurred
  after saved results; it did not invalidate the two runs documented here.

## Completed experiments

| Setting | Run ID | Selected epoch | Epochs actually run |
| --- | --- | ---: | ---: |
| Balanced | `20261103T232335Z-d1105fd8` | 7 | 8 |
| Unweighted | `20261104T000315Z-7162c5fd` | 4 | 7 (early stopped) |

Both use `microsoft/deberta-v3-base`, model revision
`8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, seed 42, batch size 8, max length
128, learning rate 2e-5, weight decay 0.01, warmup 0.1, maximum epochs 8,
patience 3, and independent Weighted F1 for checkpoint selection. Only class
weighting was changed. Joint scores are computed afterwards on each run's same
selected checkpoint; a separate joint-optimized checkpoint was not selected.
Use the explicit settings above for reproduction: the historical CLI/Makefile
defaults remain unchanged and do not themselves describe these Colab runs.

The original metadata records base Git commit `7d67277` plus uncommitted audit
changes. Their ten training source-module hashes match published audit commit
`b5a84b5`; the source-hash dictionaries are identical between the two runs.
The newer logger fix does not change either saved model or archived predictions.

## Main findings

- The main weakness is NEGATIVE_OTHER versus the three inference relations, not
  simply swapping forward and backward labels. The latest joint model has 102
  individual errors: 76 cross the negative/other boundary, 22 confuse equivalence
  with entailment, and four directly swap entailment direction.
- Comparing joint decoders, genuine NEGATIVE_OTHER items correct rose from 75 to
  82 of 106. Valid inference rows incorrectly rejected as other stayed at 52.
  The original hypothesis that removing balancing would reduce those rejections
  was not supported by this seed; the gain came from better true-other recall.
- The same HardCons score hides turnover: nine pairs were fixed and nine damaged
  between the joint models. Do not claim the new model simply retains all old
  correct answers and adds new ones.
- Raw softmax scores are overconfident: 84 wrong unweighted independent answers
  have a score of at least 99%. Those scores are not calibrated correctness odds.
- The official data has four exact unordered text pairs repeated under different
  source IDs across train/dev (seven dev rows). Source IDs do not overlap. Keep
  the official split and report this known lexical overlap.
- Pair-group bootstrap intervals for the observed F1 gains span zero. They do
  not account for training-seed variance or checkpoint/model selection on dev.

Read the [historical baseline analysis](analysis/BASELINE.md) and the
[verified comparison](analysis/UNWEIGHTED_COMPARISON.md) for examples, counts,
training history, limitations and copy-paste Colab commands.

## Start locally and reproduce the analyses

Use Python 3.10+; 3.12 is the locally tested environment. From the repo root:

```bash
make PYTHON=python3.12 setup
make fetch-data
make test
make lint
make analyze-saved
```

On Ubuntu/WSL use an available supported interpreter, e.g. `make PYTHON=python3
setup`. See the main [README](../README.md) for Windows PowerShell and setup
without Make.

The final command reads the [checked-in evidence](experiments/README.md), verifies
data/scorer hashes, saved logits and original scores, then writes fresh error
CSVs, pair summaries and comparison statistics under ignored
`results/analyses/<analysis-id>/`. It requires no GPU, model weights or private
Drive access. Keep these regenerated files outside Git.

## Next experiment, not yet run

Repeat **both** balanced and unweighted training with seed 43, keeping the above
settings and independent-F1 checkpoint selection fixed. Both decoders are
evaluated automatically. If the comparison remains useful, finish matched seed
44 runs and report averages/variation rather than only the best seed. The full
two-run Colab cell is in the [comparison report](analysis/UNWEIGHTED_COMPARISON.md#recommended-next-step-matched-seed-43-replication).

Use a CUDA runtime and confirm `torch.cuda.is_available()` before training.
Use `/content/SemEval2027-Task2` for code/data and a persistent output directory
such as `/content/drive/MyDrive/DiCoNLI/experiments`. A new clone now contains all
source changes; the earlier source ZIP overlay is no longer necessary.

A later candidate is a DeBERTa already trained on inference datasets, but its
three-label output head must be deliberately adapted to this four-label task.
That experiment has not been implemented or run. Do not just change the model
ID and assume its labels or classifier shape are compatible.

## Checkpoints and deliberately external files

The full original run folders live in the owner's Google Drive:

```text
MyDrive/DiCoNLI/experiments/20261103T232335Z-d1105fd8/
MyDrive/DiCoNLI/experiments/20261104T000315Z-7162c5fd/
```

`best_model/` in each folder is needed to make fresh predictions with that
already-trained model. These weights were not present in either uploaded ZIP
or the local workspace and therefore cannot be published by this handoff.
The owner must share/copy them separately if a collaborator needs inference
from those exact checkpoints. Training new experiments does not require them.
No public checkpoint URL is available; these are owner-relative Drive paths.

Small evidence snapshots include configuration, history, logits, predictions,
compressed logs and official scores. They omit `software_versions.json`,
`status.json` and `source_snapshot/` because those files were not supplied in
the uploads; do not fabricate replacements. The full Drive folders should
contain the normal runtime artifacts.

Downloaded input data/scorer code, virtual environments, caches, credentials,
trained weights and generated analyses stay outside Git. Official input data
and scorer code are reproducibly fetched at the recorded organizer revision.
The current trainer saves a selected model checkpoint; optimizer-state resume
of an interrupted training run is not implemented.
