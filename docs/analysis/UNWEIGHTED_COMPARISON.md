> Verified analysis of the two seed-42 runs. See [current handoff](../HANDOFF.md) for project state and checkpoint locations.

# Balanced versus unweighted training: verified development comparison

Baseline: `20261103T232335Z-d1105fd8`, class weighting `balanced`, seed 42.
New run: `20261104T000315Z-7162c5fd`, class weighting `none`, seed 42.

The new archive is the correct unweighted run. Its joint decoder is the strongest
observed configuration so far: 84.9783 Weighted F1, 90.6137 SoftCons, 85.9206
HardCons. It gets 558 of 660 individual examples correct (84.5455% accuracy).
The earlier best-F1 configuration got 556 correct. The improvement is modest
and is not established as reliable by one training seed.

## Verification and scope

- All declared training settings are identical except class weighting. Metadata
  differences are the run ID, start time and checkpoint location.
- Both runs record identical source-file hashes, data hashes, model revision,
  hardware metadata, seed and checkpoint-selection rule.
- All ten source-module hashes match published audit commit `b5a84b5`.
  The local logging-only fix made after these runs is intentionally excluded
  from that historical source comparison.
- The same public English training/development files are used. Instance and
  source-pair IDs are disjoint across the official splits.
- Both new prediction files preserve all 660 IDs and their order. Their labels
  are reproduced exactly by decoding the uploaded `(660, 4)` saved logits.
- The official scorer was rerun for both new prediction files. Its classification
  and consistency reports exactly match the uploaded reports; local metrics
  agree within 1e-12. The scorer's hashes match the run's recorded versions.
- Both archives omit checkpoint weights, status.json and software_versions.json.
  This analysis does not reload models, independently repeat GPU inference, or
  verify that every installed dependency was identical between runs.
- No new training, prediction corrections, gold-label edits, submissions, commits
  or pushes were performed. Existing local logging-fix changes were preserved.

## All four evaluations

| Training | Decoding | Weighted F1 | Accuracy | Correct / 660 | SoftCons | HardCons |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Balanced | Independent | 84.5184% | 84.2424% | 556 | 88.8087% | 85.1986% |
| Balanced | Joint | 83.8441% | 83.4848% | 551 | 90.6137% | 85.9206% |
| Unweighted | Independent | 84.2641% | 83.7879% | 553 | 88.0866% | 83.7545% |
| Unweighted | Joint | 84.9783% | 84.5455% | 558 | 90.6137% | 85.9206% |

Removing weighting alone lowered independent F1 by 0.2543 percentage points.
With joint decoding it raised F1 by 1.1342 points. Against the previous best
configuration (balanced independent), the gain is 0.4599 points and two net
correct individual answers.

The same checkpoint is used for both decoders within each run. Between training
runs the chosen checkpoints differ: epoch 7 for balanced, epoch 4 for unweighted.
That is a consequence of the unchanged independent-dev-F1 selection rule; this
comparison evaluates the complete training/selection recipe.

## Where the improvement came from

Compare joint decoding with joint decoding:

| Quantity | Balanced joint | Unweighted joint |
| --- | ---: | ---: |
| Genuine NEGATIVE_OTHER examples answered correctly | 75 / 106 | 82 / 106 |
| Genuine NEGATIVE_OTHER examples mistaken for inference | 31 | 24 |
| Valid inference rows incorrectly rejected as NEGATIVE_OTHER | 52 | 52 |
| Correct reversible individual rows | 476 / 554 | 476 / 554 |
| Fully correct reversible pairs | 238 / 277 | 238 / 277 |
| NEGATIVE_OTHER label F1 | 64.38% | 68.33% |

The net gain of seven correct answers comes entirely from the negative/other
gold items. Across the 106 such items, eight previous mistakes were fixed and
one previously correct answer became wrong. Joint decoding leaves these
singleton items unchanged within each run.

The earlier hypothesis was that removing class balancing might reduce excessive
rejection of valid inference examples. This result does not support that
mechanism: the false rejections remain at 52 for joint decoding and rise from
48 to 55 for independent decoding. Instead, the model recognizes more actual
negative/other examples. Training dynamics and checkpoint selection interact;
loss weighting does not impose a simple prediction-frequency rule.

Examples of negative/other items that became correct, using the official labels:

- `Syria opposition unity` / `Syria divided opposition`: previously equivalence,
  now NEGATIVE_OTHER.
- `on a runway` / `close to the ground`: previously forward entailment, now
  NEGATIVE_OTHER.
- `as Manager` / `Ferguson`: previously backward entailment, now NEGATIVE_OTHER.

One new negative/other mistake is `Two black and white dogs` / `A dog next to
another standing dog`: previously NEGATIVE_OTHER, now forward entailment.
These are comparisons against supplied labels, not a claim of independent
annotation adjudication. Gold labels remain unchanged.

## Prediction changes and remaining errors

Balanced joint versus unweighted joint changes 47 rows: 26 fixed, 19 damaged,
and two wrong-to-different-wrong changes. Of the 109 old joint errors, 83 remain
errors in the new model, and 19 additional errors appear, giving 102 errors.

The identical HardCons scores conceal turnover: nine reversible pairs became
fully correct and nine previously fully correct pairs became wrong. Thus the
new model does not simply retain all previous strengths and add seven fixes.

Within the unweighted run, joint decoding changes eight rows: six fixes, one
damaged answer and one wrong-to-different-wrong change. It completes six
previously half-correct pairs and loses no previously fully correct pair,
raising HardCons from 232/277 to 238/277.

The new joint model's 102 errors consist of:

- 76 negative/other-boundary errors: 52 valid inference rows rejected as other,
  plus 24 other examples assigned an inference label.
- 22 equivalence-versus-entailment errors.
- 4 direct forward-versus-backward swaps.

Of 39 reversible pairs not fully correct, 26 are predicted NEGATIVE_OTHER in
both directions and 13 receive a wrong but internally compatible inference
relation. The negative/other boundary remains the main priority.

## Confidence and training

Unweighted training completed seven epochs and early-stopped after three epochs
without an independent-F1 improvement over epoch 4. Epoch 4 had F1 84.2641%;
epochs 5–7 scored 83.1515%, 82.5275% and 83.2551%. Its mean training loss kept
falling, from 1.1262 at epoch 1 to 0.0141 at epoch 7.

This shows the selected epoch was preserved correctly and supports keeping
early stopping. It does not identify a universally optimal epoch limit or
prove the cause of the later decline.

Overconfidence persists. For raw independent scores at least 99%:

| Run | Predictions above threshold | Incorrect against gold |
| --- | ---: | ---: |
| Balanced | 613 | 83 |
| Unweighted | 599 | 84 |

These are uncalibrated softmax outputs, not trustworthy probabilities of being
correct. No confidence thresholds or predictions were tuned during this analysis.

## How much evidence is there for improvement?

A paired percentile bootstrap resampled the 383 development source units
(277 reciprocal pairs plus 106 singletons), keeping each pair's directions
together. With 10,000 resamples and analysis RNG seed 2027:

- Unweighted joint minus balanced independent: observed +0.46 F1 points;
  conditional 95% bootstrap interval approximately -1.89 to +2.79 points.
- Unweighted joint minus balanced joint: observed +1.13 F1 points;
  conditional interval approximately -1.41 to +3.72 points.

Both intervals span zero. This does not establish a reliable advantage from
these development data alone. The bootstrap also does not measure training-seed
variance or correct for selecting checkpoints/configurations on this same dev
set. It is not a prediction interval for the hidden competition test score.

## Recommended next step: matched seed-43 replication

Keep unweighted + joint as the current best observed configuration, and retain
the balanced baseline. Before adding further tuning choices, repeat both
training settings with seed 43, holding everything else fixed. This tests
whether the apparent gain survives another training initialization. If useful,
complete the same pair with seed 44 and compare all three official metrics
across matched seeds. Do not select only the best seed as evidence of progress.

Run this in the existing Colab environment with Drive mounted. It launches two
new training runs, saving each in its own unique experiment directory:

```python
import subprocess
import sys

for weighting in ("balanced", "none"):
    subprocess.run([
        sys.executable, "-u", "-m", "dico_nli.train",
        "--train-file", "data/raw/train/dico_nli_train_track1_participant_labeled.csv",
        "--dev-file", "data/raw/dev/dico_nli_dev_track1_participant_labeled.csv",
        "--reference-file", "data/raw/dev/dico_nli_dev_track1_reference.csv",
        "--output-dir", "/content/drive/MyDrive/DiCoNLI/experiments",
        "--model-name", "microsoft/deberta-v3-base",
        "--model-revision", "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        "--device", "cuda", "--seed", "43",
        "--epochs", "8", "--patience", "3", "--batch-size", "8",
        "--max-length", "128", "--learning-rate", "2e-5",
        "--weight-decay", "0.01", "--warmup-ratio", "0.1",
        "--class-weight", weighting,
        "--decode", "independent", "--selection-metric", "weighted_f1",
    ], cwd="/content/SemEval2027-Task2", check=True)
```

Both decoding methods are evaluated automatically at the end of each run. These
replications have not been executed by this analysis. A later comparison with
an inference-pretrained backbone remains a separate experiment requiring a
deliberately adapted four-label output head. No guarantee of approaching 100%
or achieving a particular leaderboard rank follows from these runs.

## Reproducing the analysis

Small original evidence files for this run are tracked in
[`docs/experiments/20261104T000315Z-7162c5fd/`](../experiments/20261104T000315Z-7162c5fd/).
They include configuration, history, predictions, saved logits, compressed logs
and the original official scores, but no model weights or task input datasets.

From the repository root after setup and `make fetch-data`, run:

```bash
make analyze-saved
```

This invokes `scripts/analyze_saved_runs.py`, checks dataset/scorer hashes,
reproduces both prediction files and official reports, and regenerates the
error CSVs, pair summaries and comparison statistics under a fresh ignored
`results/analyses/<analysis-id>/` directory. Per-run `artifacts/` paths in this
report refer to `<analysis-id>/20261104T000315Z-7162c5fd/artifacts/`;
between-run comparisons are written at `<analysis-id>/` itself.

The bootstrap uses 10,000 pair-group resamples and RNG seed 2027. It performs
no model training and does not need a GPU. Original evidence remains untouched.
