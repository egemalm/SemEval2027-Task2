# DiCo-NLI development rules

This is the existing SemEval 2027 Task 2 English (Track 1) project. Read README.md,
Makefile, pyproject.toml, relevant src/dico_nli modules, and their tests before
editing. Inspect git status, branch, remotes, and recent commits; preserve user
changes and working command-line behavior. Prefer small, testable changes over
rewrites, duplicate pipelines, unnecessary abstractions, or new tracking services.

- Train on the official training split only. Development labels may be used for
  reported evaluation and explicitly documented checkpoint/hyperparameter selection.
  Reject overlap in instance_id or pair_id, including reversed source pairs.
  Report duplicate phrase texts separately; do not silently change official splits.
- Never use hidden test labels. Prediction must work without labels or reference
  files; do not feed reverse_pair_id or gold labels into decoding/model features.
- The organizer's evaluator is authoritative. Never silently modify it or replace
  official metrics. NEGATIVE_OTHER affects Weighted F1 but is excluded from
  SoftCons/HardCons gold-pair denominators; negative predictions on reversible gold
  pairs do not count as soft-consistent.
- Preserve all submission IDs, row counts, and the exact instance_id,label format.
- Never fabricate scores, experiments, hardware checks, or push results. Identify
  synthetic smoke tests explicitly; perfect-gold scoring is only a scorer check.
- Save actual runs in unique results/experiments subdirectories, including exact
  config, model revision, seed, input hashes/provenance, Git state, source snapshot,
  software, hardware, logs, checkpoints, predictions, and official metrics. Never
  overwrite completed runs. Use the comparison command before claiming improvement.
- Compare independent and joint decoding on the same checkpoint. Document any
  change in selection rules, external data/models, and reproducibility assumptions.
- Run relevant tests and lint before reporting completion. Tests must run on CPU
  without downloading models; use tiny locally initialized checkpoints. Do not run
  serious GPU experiments unless requested. Report checks that could not run.
- Keep downloaded data/scorers, weights, environments, credentials, and results
  untracked. Review git diff and git status before any commit.
- Never push or publish changes without explicit user permission. Prior pushes in
  this conversation do not authorize future publication.
