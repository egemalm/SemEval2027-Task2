> Historical baseline analysis. Its proposed unweighted experiment has now been completed; see [the comparison](UNWEIGHTED_COMPARISON.md) and [current handoff](../HANDOFF.md).

# English baseline: development error analysis

Run: `20261103T232335Z-d1105fd8` (seed 42, DeBERTa-v3-base).

The model's main weakness is distinguishing NEGATIVE_OTHER from the three
inference relations. Of its 104 independent prediction errors, 79 (76.0%) cross
that boundary. Only four errors directly swap forward and backward entailment.
This suggests that negative/other classification is the highest-priority area
to investigate before adding further directional decoding rules.

This is analysis of the submitted run, not a new training experiment. Nothing
was trained, corrected in the gold data, submitted, or published during analysis.

## Verification

- All 660 prediction IDs and their row order match the pinned public English
  development file. There are no duplicate or missing predictions.
- Uploaded train, dev, and dev-reference SHA-256 values match the local pinned
  organizer files. Training/development instance and source-pair IDs are disjoint.
- The saved logits have shape `(660, 4)` and are finite. Decoding them using the
  reviewed code reproduces both uploaded prediction files exactly.
- All ten source-module hashes recorded in the run match the local code.
- The official scorer's twelve source-file hashes match the recorded versions.
  Rerunning it reproduces both uploaded classification and consistency reports
  exactly; the local three metrics agree within 1e-12.
- Training metadata records a Tesla T4 with CUDA; the history contains eight
  completed epochs. The saved best checkpoint is epoch 7.
- The checkpoint weights and software_versions.json were not in this archive;
  this analysis did not reload the model or independently rerun GPU inference.
  The source hash and log checks support artifact consistency, not independent
  authentication of every action performed on the remote machine.

## Overall results

| Measure | Independent | Joint |
| --- | ---: | ---: |
| Weighted F1 | 84.5184% | 83.8441% |
| Actual item accuracy | 84.2424% | 83.4848% |
| Correct answers / 660 | 556 | 551 |
| Wrong answers / 660 | 104 | 109 |
| SoftCons | 88.8087% (246/277) | 90.6137% (251/277) |
| HardCons | 85.1986% (236/277) | 85.9206% (238/277) |

F1 is not accuracy. Now that the full scorer reports are available, the exact
number of correct individual answers can be stated as 556 for the baseline.

## Which labels cause errors?

| Expected label | Dev rows | Independent correct | Independent label F1 | Joint label F1 |
| --- | ---: | ---: | ---: | ---: |
| EQUIVALENCE | 174 | 143 | 84.37% | 83.78% |
| FORWARD_ENTAILMENT | 190 | 168 | 89.60% | 89.54% |
| BACKWARD_ENTAILMENT | 190 | 170 | 90.19% | 89.07% |
| NEGATIVE_OTHER | 106 | 75 | 65.50% | 64.38% |

Independent errors divide into three disjoint groups:

- 79 negative/other-boundary errors: 48 valid inference examples rejected as
  NEGATIVE_OTHER, plus 31 gold NEGATIVE_OTHER examples assigned an inference label.
- 21 equivalence-versus-entailment errors.
- 4 direct forward-versus-backward errors.

The model predicts NEGATIVE_OTHER 123 times, although there are only 106 such
gold items. Of those 123 predictions, 75 are correct: precision 60.98%; recall
75/106 = 70.75%. This is a two-sided boundary problem, not simply an inability to
recognize unrelated words. NEGATIVE_OTHER includes the task's weaker semantic
relations as well as noise/unrelated items.

## What joint decoding actually did

Only ten individual predictions changed:

- 2 wrong answers became correct.
- 7 correct answers became wrong.
- 1 wrong answer changed into a different wrong answer.

Both fixes completed a previously half-correct reversible pair, producing two
additional fully correct pairs. The seven damaged answers belonged to pairs
that already had an incorrect direction, so no previously fully correct pair
was lost. This explains the higher HardCons alongside lower item accuracy/F1.

All 106 negative gold items are singletons in this release, so their predictions
are unchanged by the paired decoder. Joint decoding also rejects four additional
valid inference rows as NEGATIVE_OTHER (48 becomes 52).

The 26 reversible pairs that fail joint SoftCons are all predicted
NEGATIVE_OTHER in both directions. The official metric gives those no credit;
the other 251 pairs satisfy the reversal mapping. Compatibility alone does not
make a wrong pair correct.

Examples, interpreted strictly against the supplied gold labels:

| Text 1 | Text 2 | Gold | Independent | Joint | Effect |
| --- | --- | --- | --- | --- | --- |
| Rabbi Ovadia Yosef | Ovadia Yosef | EQUIVALENCE | FORWARD_ENTAILMENT | EQUIVALENCE | Fixed |
| from deep Mediterranean Sea | in Mediterranean | FORWARD_ENTAILMENT | NEGATIVE_OTHER | FORWARD_ENTAILMENT | Fixed |
| fresh presidential election | fresh election for president | EQUIVALENCE | EQUIVALENCE | BACKWARD_ENTAILMENT | Damaged |
| across the snow | around the flat snowy street | BACKWARD_ENTAILMENT | BACKWARD_ENTAILMENT | NEGATIVE_OTHER | Damaged |

See `artifacts/joint_changes.csv` for all ten changes, including the phrases and
gold labels. No post-hoc correction was applied to submitted predictions.

## Confidence and training history

Of 613 independent predictions with raw maximum softmax at least 99%, 83 are
wrong against the official gold labels. Of all 104 wrong answers, 83 (79.8%)
have that high score. These outputs are uncalibrated model scores, not verified
probabilities of correctness. High confidence by itself cannot identify which
examples should be trusted or corrected.

| Epoch | Mean training loss | Independent dev Weighted F1 | Dev HardCons |
| --- | ---: | ---: | ---: |
| 1 | 1.1353 | 78.49% | 79.78% |
| 2 | 0.4694 | 82.57% | 83.03% |
| 3 | 0.2237 | 82.31% | 84.12% |
| 4 | 0.1283 | 83.47% | 82.31% |
| 5 | 0.0666 | 83.31% | 83.75% |
| 6 | 0.0348 | 83.87% | 83.03% |
| 7 | 0.0321 | 84.52% | 85.20% |
| 8 | 0.0197 | 84.01% | 83.75% |

The model fits the training examples very closely. Development improvement is
much smaller late in training; epoch 8 has lower training loss but worse dev
scores than epoch 7. This supports preserving checkpoint selection and avoiding
the assumption that more epochs will help. One small final-epoch decline, with
no recorded dev loss or repeated seeds, does not establish a precise onset or
cause of overfitting. The patience limit was not reached; all eight epochs ran.

Some gold expectations are difficult to infer confidently from isolated phrases.
For example, `Australian senator` / `Australian lawmaker` is NEGATIVE_OTHER in
this release, while the model predicts forward entailment. Such examples merit
manual review of original annotation context/provenance. They are not proof that
the gold is wrong, and gold labels must not be changed to improve the score.

The four known repeated unordered phrase pairs under distinct train/dev source
IDs account for seven dev rows. Both methods answer all seven correctly. The
official split was preserved; this modest overlap should be reported rather
than used as extra training or a source of handcrafted prediction rules.

## Recommended next experiment: remove class balancing

Before changing the backbone, run one controlled ablation of the existing loss
weights. Keep the model, data, seed 42, batch size 8, learning rate, max length,
epoch limit, independent-F1 checkpoint selection, and final two-decoder comparison
the same; change only `--class-weight balanced` to `--class-weight none`.

Current inverse-frequency training weights are:

| Class | Weight |
| --- | ---: |
| EQUIVALENCE | 0.9483 |
| FORWARD_ENTAILMENT | 0.8632 |
| BACKWARD_ENTAILMENT | 0.8632 |
| NEGATIVE_OTHER | 1.5910 |

An individual NEGATIVE_OTHER example therefore has about 1.84 times the loss
weight of an entailment example. Weighted cross-entropy changes the relative
contribution of classes ([PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)).
Together with the 48 valid inference examples rejected as negative, this motivates
a specific hypothesis: removing balancing may reduce false negative/other
assignments. It may also reduce negative recall or lower F1; it is not a proven
fix, and the current run does not show that weighting caused the errors.

Run in the existing Colab environment, from the repository root:

```python
!python -u -m dico_nli.train \
  --train-file data/raw/train/dico_nli_train_track1_participant_labeled.csv \
  --dev-file data/raw/dev/dico_nli_dev_track1_participant_labeled.csv \
  --reference-file data/raw/dev/dico_nli_dev_track1_reference.csv \
  --output-dir /content/drive/MyDrive/DiCoNLI/experiments \
  --model-name microsoft/deberta-v3-base \
  --model-revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
  --device cuda --seed 42 --epochs 8 --patience 3 \
  --batch-size 8 --max-length 128 --learning-rate 2e-5 \
  --weight-decay 0.01 --warmup-ratio 0.1 \
  --class-weight none --decode independent --selection-metric weighted_f1
```

Inspect all three official metrics and the negative-class confusion counts.
Retain both artifacts even if the new result is worse. If this comparison looks
useful, repeat balanced and unweighted variants with matched seeds 43 and 44 to
measure variability. If not, keep the current baseline and move to an inference-
pretrained backbone with a deliberately adapted four-label head as a separate
experiment. That change has not been implemented or tested here.

Do not tune many thresholds or hand-fix dev examples to chase a higher dev score.
The dev set already selects checkpoints, and further choices can overfit it.
Do not turn the presence/absence of a reversed row into a gold-label shortcut;
final test construction and organizer permission for such a shortcut are not
established by these files.

## Reproducing the analysis

Small original evidence files for this run are tracked in
[`docs/experiments/20261103T232335Z-d1105fd8/`](../experiments/20261103T232335Z-d1105fd8/).
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
report refer to `<analysis-id>/20261103T232335Z-d1105fd8/artifacts/`;
between-run comparisons are written at `<analysis-id>/` itself.

The bootstrap uses 10,000 pair-group resamples and RNG seed 2027. It performs
no model training and does not need a GPU. Original evidence remains untouched.
