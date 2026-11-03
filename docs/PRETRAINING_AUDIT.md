# Pre-training audit — 2026-11-03

This report records the pre-training audit before publication. During that
audit, no full DeBERTa training experiment, task-quality benchmark, commit,
push, or publication was performed. Later training and publication are separate
from the checks documented here.

## Repository and upstream verification

- Initial working tree: clean, branch `main`, tracking `origin/main`.
- Remote: `https://github.com/egemalm/SemEval2027-Task2.git`.
- Local HEAD and `git ls-remote origin refs/heads/main` both identify
  `7d6727747fae13ee2d73950a8c5903174e221dba`.
- The public [GitHub commit page](https://github.com/egemalm/SemEval2027-Task2/commit/7d6727747fae13ee2d73950a8c5903174e221dba)
  confirms the previous setup exists. The repository is not currently empty.
  Initial setup commit `ab28184` is its parent. The earlier empty-repository
  observation does not establish why any particular clone lacked files.
- Official source HEAD matches the asset pin:
  `588968e610197ddc4c440314701cbc587afc4c1b`. Public data and cached evaluator
  files were compared byte-for-byte with this revision. Evaluator source was
  not edited; upstream license/README accompany the cache.

## Confirmed findings and corrections

| Severity | Finding | Resolution |
| --- | --- | --- |
| High | Reusing `--output-dir` overwrote checkpoints, config, predictions and history. | Every invocation now creates a unique run subdirectory; completed artifacts are preserved. Prediction/scorer outputs also refuse existing files. |
| Medium | Without a reference, SoftCons/HardCons were invented as 0 and `mean3` was mislabeled Weighted F1. | Unavailable values are `null`; consistency selection still requires a reference. |
| Medium | Default pandas inference could change IDs such as `001` into `1`. | Load task CSV fields as strings; preserve IDs and row order through unlabeled prediction. |
| Medium | Joint decoding assumed any two rows sharing `pair_id` were actual reversals. | Check swapped texts and languages before constraining; otherwise retain independent predictions. |
| Medium | Local scorer accepted malformed references including self-reversal and mismatched source IDs. | Validate source IDs, labels, self-links, reciprocity and matching pair IDs. Official scorer remains authoritative. |
| Medium | No enforcement prevented accidentally training with overlapping train/dev instance or source IDs. | Reject either overlap before loading model weights; preserve and report exact-text duplicates separately. No source-ID leakage was found in the released files. |
| Medium | Fetching a new revision silently kept old cached files. | Verify existing bytes against the requested revision and save hash/URL/revision manifests. Conflicting files require explicit `--force`. |
| Medium | Broad dependencies allowed Torch 2.4/2.5 with Transformers versions that reject loading PyTorch weight files below 2.6. | Raise the Torch lower bound to 2.6; add optional validated environment constraints and record actual installed versions. |
| Medium | Runs lacked immutable model revision, Git/source provenance, package/hardware details, saved logs and automatic official final scoring. | Record all metadata, source snapshots/hashes, run status, logs, both decoders' predictions and official scores. |
| Low | Inference defaulted to length 128 even after training with another length; checkpoint class order was unchecked. | Save/reuse inference length and validate the four-label checkpoint order. |
| Low | Best-model reload held training weights and optimizer memory at the same time. | Release them before loading the selected checkpoint; save failed-run status and actionable OOM guidance. |
| Low | PowerShell could report success after an external installer/download failure. | Check native exit codes and stop on failure. No Windows host was available to run it. |
| Low | Make assumed `python3.12`, which may be unavailable on Ubuntu/WSL. | Default to `python3`, document the 3.10+ requirement and interpreter overrides. |

The original implementation already used separate train/dev frames, optimized
only training batches, dropped labels before model prediction, wrote the two
submission columns, mapped F↔B/E↔E correctly, and fell back on single directions.
Those behaviors were retained. `make -n train-dev` confirms its explicit
training/dev/reference paths; the name never meant train-on-dev.

## Data audit: confirmed upstream text duplication

The public English split contains 3,042 training rows and 660 development rows.
Train has 1,282 reversible pairs and 478 negative singletons; development has
277 reversible pairs and 106 negative singletons. Instance/source-pair IDs do
not overlap across splits.

Four unordered phrase-pair texts nevertheless repeat under different source IDs:

- `a baby` / `a small baby`
- `are playing` / `play`
- `at the camera` / `into the camera`
- `driving` / `driving down`

This is confirmed lexical overlap in the organizer release, not evidence that
our optimizer consumed dev labels. We preserve the official split for comparable
scores, warn about these duplicates, and save them in each `split_audit.json`.
Any future deduplicated sensitivity study must be separately named and reported.

## Evaluation and controlled comparison

The [official evaluator](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/tree/588968e610197ddc4c440314701cbc587afc4c1b/evaluation_functions)
defines Weighted F1 across all labels. SoftCons/HardCons use only reciprocal
reversible gold pairs. Negative predictions on those gold pairs fail SoftCons,
even when both directions are predicted negative. `mean3` is only a local
selection heuristic and is explicitly distinguished from official scores.

Local metrics were compared with the actual unmodified scorer on twelve
synthetic edge/random prediction sets. Additional checks on all 660 public dev
rows used public-gold-copy, all-negative, and random predictions; all three
metrics agreed within 1e-12. These are scorer checks, not model results.
Their labeled artifacts are stored locally under
`results/audits/20261103T231119Z-7528e71b/`.

The proposed first experiment selects one checkpoint by independent dev Weighted
F1 and evaluates both independent and joint decoding from its single saved
logit matrix. This isolates decoder effects. It does not test a new loss or
claim that joint decoding improves task performance. The four joint assignment
choices include negative/negative as an inference assumption; the official
reversal operator itself remains undefined for NEGATIVE_OTHER.

## Development foundation and artifacts

- `AGENTS.md` defines repository-specific inspection, data separation, honest
  reporting, test, preservation and no-publication rules.
- `src/dico_nli/experiments.py` creates unique run folders and captures config,
  provenance, source snapshots, versions, device, status and logs.
- `src/dico_nli/compare.py` reads completed official scores, excludes synthetic
  runs by default, reports winners per metric/dataset hash group, and provides
  full configuration differences with `--diff`.
- Existing training/prediction/scoring modules, data fetcher, Windows bootstrap,
  Makefile, pyproject.toml, .gitignore and README were extended in place.
- Existing tests remain. New data/label, official-parity, experiment-preservation
  and CPU end-to-end tests cover the gaps. GitHub Actions is configured for CPU
  tests and lint; the workflow has not been published or run on GitHub.
- `requirements-lock.txt` records the locally validated package versions as
  optional constraints. It is not a platform-independent CUDA wheel lock.

## Verification and limits

- Expanded pytest suite: 30 tests pass, including actual one-epoch CPU training
  of a tiny local random BERT, checkpoint saving/reload, both decoders, official
  scoring, and entirely unlabeled inference. HF network access is disabled for
  this test. Its synthetic scores are marked `cpu_smoke` and excluded from
  normal comparisons. Machine-readable results are saved beside the scorer
  audit as `pytest.xml`.
- Ruff lint/format, pip dependency checks and Git whitespace checks pass locally.
- Tiny random DeBERTa forward/backward operations passed on CPU and Apple MPS
  with finite loss and four logits. This is kernel validation, not full training.
- CPU fallback and unavailable-CUDA errors are tested. CUDA is unavailable on
  this machine; no NVIDIA/Colab/WSL GPU run, Windows execution, cross-OS install
  or GitHub-hosted CI execution is claimed.
- The default DeBERTa configuration/tokenizer are available, and a concrete model
  commit is supplied for the first experiment. Full pretrained-weight loading,
  real training memory use/time and task accuracy remain to be measured.
- Seeds cover Python/NumPy/Torch. Strict deterministic mode is optional because
  kernels may be unsupported; cross-hardware identical results are not promised.
- Pretraining contamination and the future hidden-test metadata format cannot be
  proven from public files. Inference supports singleton rows and absent pair IDs;
  it never requires hidden labels or reference metadata.

The project is ready for its first controlled real training run, subject to
confirming the target GPU/PyTorch installation and memory capacity. Use the
pinned command in README.md; no serious training was started during this audit.
