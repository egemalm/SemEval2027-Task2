# Curated experiment evidence

These small snapshots are intentionally tracked for project handoff, with the
user's explicit publication authorization. They came from the two uploaded
Colab run archives and contain no model weights or task input datasets.
See [the handoff](../HANDOFF.md) and [analysis reports](../analysis/UNWEIGHTED_COMPARISON.md).

```text
<run-id>/
  run_config.json
  best_metrics.json
  metrics_history.json
  dev_logits.npy
  predictions_dev_independent.csv
  predictions_dev_joint.csv
  official_independent/scores.json
  official_joint/scores.json
  training.log.gz
```

Log gzip compression is lossless; other evidence files preserve the uploaded
content. Configurations retain their original Colab paths and Git metadata.
Two run IDs are included:

- `20261103T232335Z-d1105fd8`: balanced training, seed 42, selected epoch 7.
- `20261104T000315Z-7162c5fd`: unweighted training, seed 42, selected epoch 4.

Both train on the organizer's 3,042 English rows and evaluate on 660 dev rows.
These are public development results selected using development labels. They
are not hidden-test scores or official leaderboard placements.

## Reproduce

After installing the project and running `make fetch-data`:

```bash
make analyze-saved
```

Without Make:

```bash
python scripts/analyze_saved_runs.py
```

The script verifies file provenance and reproduces each saved prediction from
its logits, reruns the unmodified organizer scorer, checks all reported scores,
and generates the error CSVs and source-group bootstrap comparison. It uses
only the checked-in snapshots plus the reproducible official data/scorer
download. It does not need checkpoint weights or a GPU.

`metrics.csv` is the compact score table. `manifest.json` records checksums of
all snapshot files. Detailed generated analyses go under ignored `results/`.
The usual `dico-compare` command expects full training-run directories with
`status.json`; use `analyze-saved` for these deliberately limited snapshots.

Do not silently edit an existing snapshot to improve results. Add a separately
identified snapshot only when a real experiment has completed and publication
has been authorized. Missing original software-version and checkpoint files
remain missing; the handoff documents where to request them.
