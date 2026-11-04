#!/usr/bin/env python3
"""Reproduce the checked-in seed-42 analyses without downloading model weights."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dico_nli.data import load_examples, validate_split_integrity
from dico_nli.decode import decode_logits
from dico_nli.experiments import create_run, fingerprint, write_json
from dico_nli.labels import LABEL_TO_ID, LABELS, REVERSAL_MAP
from dico_nli.metrics import score_predictions
from dico_nli.score import run_official_scorer

REPO = Path(__file__).resolve().parents[1]
BASELINE = "20261103T232335Z-d1105fd8"
UNWEIGHTED = "20261104T000315Z-7162c5fd"
SETTINGS = (
    "model_name",
    "resolved_model_revision",
    "seed",
    "batch_size",
    "max_length",
    "epochs",
    "patience",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "class_weight",
    "decode",
    "selection_metric",
    "deterministic",
    "num_workers",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_run(
    raw: Path, output: Path, paths: dict, dev: pd.DataFrame, reference: pd.DataFrame, cache: Path
) -> tuple[pd.DataFrame, dict, dict]:
    config = read_json(raw / "run_config.json")
    if config["run_id"] != raw.name or config["label_order"] != list(LABELS):
        raise ValueError(f"Invalid run identity or class order: {raw}")
    for name, path in paths.items():
        if fingerprint(path)["sha256"] != config["datasets"][name]["sha256"]:
            raise ValueError(f"{raw.name}: {name} differs from the recorded dataset.")
    for name, expected in config["official_scorer_files"].items():
        if fingerprint(cache / "evaluation_functions" / name)["sha256"] != expected:
            raise ValueError(f"Official scorer differs from the run: {name}")

    output.mkdir(parents=True, exist_ok=False)
    frame = dev.copy()
    frame["reverse_pair_id"] = (
        reference.set_index("instance_id").loc[frame.instance_id, "reverse_pair_id"].to_numpy()
    )
    logits = np.load(raw / "dev_logits.npy", allow_pickle=False)
    if logits.shape != (len(frame), len(LABELS)) or not np.isfinite(logits).all():
        raise ValueError(f"Invalid saved logits in {raw}")
    probabilities = np.exp(logits.astype(float) - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    frame["raw_max_softmax"] = probabilities.max(axis=1)
    summary = {
        "run_id": raw.name,
        "class_weight": config["class_weight"],
        "seed": config["seed"],
        "scores": {},
    }
    for mode in ("independent", "joint"):
        prediction_path = raw / f"predictions_dev_{mode}.csv"
        predictions = pd.read_csv(prediction_path, dtype=str, keep_default_na=False)
        if (
            predictions.columns.tolist() != ["instance_id", "label"]
            or predictions.instance_id.tolist() != frame.instance_id.tolist()
        ):
            raise ValueError(f"Prediction format, IDs, or order differs: {prediction_path}")
        if decode_logits(frame, logits, strategy=mode) != predictions.label.tolist():
            raise ValueError(f"Saved logits do not reproduce {prediction_path}")
        official = run_official_scorer(
            cache, paths["dev_reference"], prediction_path, output / f"recomputed_official_{mode}"
        )
        uploaded = read_json(raw / f"official_{mode}" / "scores.json")
        local = score_predictions(paths["dev_reference"], predictions)
        for metric in ("weighted_f1", "soft_cons", "hard_cons"):
            if not np.isclose(local[metric], official[metric], atol=1e-12, rtol=0):
                raise ValueError(f"Local/official mismatch: {raw.name}/{mode}/{metric}")
        if uploaded != official:
            raise ValueError(f"Recomputed official report differs: {raw.name}/{mode}")
        frame[mode] = predictions.label.to_numpy()
        frame[mode + "_correct"] = frame[mode].eq(frame.label)
        wrong = frame.loc[~frame[mode + "_correct"]]
        boundary = wrong.label.eq("NEGATIVE_OTHER") | wrong[mode].eq("NEGATIVE_OTHER")
        swaps = wrong.label.isin(LABELS[1:3]) & wrong[mode].isin(LABELS[1:3])
        summary["scores"][mode] = {
            **{key: official[key] for key in ("weighted_f1", "soft_cons", "hard_cons")},
            "accuracy": official["classification"]["accuracy"],
            "correct_rows": int(frame[mode + "_correct"].sum()),
            "wrong_rows": len(wrong),
            "negative_boundary_errors": int(boundary.sum()),
            "direction_swap_errors": int(swaps.sum()),
            "equivalence_vs_direction_errors": int((~boundary & ~swaps).sum()),
            "valid_inference_rejected_as_other": int(wrong[mode].eq("NEGATIVE_OTHER").sum()),
            "other_mistaken_for_valid_inference": int(wrong.label.eq("NEGATIVE_OTHER").sum()),
            "per_label": official["classification"]["per_label"],
        }
        wrong.sort_values("raw_max_softmax", ascending=False).to_csv(
            output / f"{mode}_errors.csv", index=False
        )
    frame["joint_effect"] = np.select(
        [
            ~frame.independent_correct & frame.joint_correct,
            frame.independent_correct & ~frame.joint_correct,
            frame.independent.ne(frame.joint),
        ],
        ["fixed", "damaged", "wrong_to_different_wrong"],
        default="unchanged",
    )
    changed = frame.loc[frame.joint_effect.ne("unchanged")]
    summary["joint_effect"] = changed.joint_effect.value_counts().to_dict()
    changed.to_csv(output / "joint_changes.csv", index=False)
    frame.to_csv(output / "all_predictions_with_gold.csv", index=False)

    pairs, seen = [], set()
    lookup = frame.set_index("instance_id")
    for row in frame.itertuples(index=False):
        if not row.reverse_pair_id or row.instance_id in seen:
            continue
        twin = lookup.loc[row.reverse_pair_id]
        pair = {"pair_id": row.pair_id, "text1": row.text1, "text2": row.text2}
        for mode in ("independent", "joint"):
            first, second = getattr(row, mode), twin[mode]
            pair[mode + "_hard"] = first == row.label and second == twin.label
            pair[mode + "_soft"] = first in REVERSAL_MAP and second == REVERSAL_MAP[first]
        pairs.append(pair)
        seen.update((row.instance_id, row.reverse_pair_id))
    pd.DataFrame(pairs).to_csv(output / "reversible_pairs.csv", index=False)
    history = pd.DataFrame(read_json(raw / "metrics_history.json")["epochs"])
    best = read_json(raw / "best_metrics.json")
    selected = int(history.loc[history[config["selection_metric"]].idxmax(), "epoch"])
    if selected != best["epoch"]:
        raise ValueError(f"Selected epoch does not match history: {raw.name}")
    selected_scores = summary["scores"][config["decode"]]
    for metric in ("weighted_f1", "soft_cons", "hard_cons"):
        if not np.isclose(best[metric], selected_scores[metric], atol=1e-12, rtol=0):
            raise ValueError(f"Selected checkpoint metrics differ: {raw.name}/{metric}")
    history.to_csv(output / "epoch_history.csv", index=False)
    summary["training"] = {"actual_epochs": len(history), "best_epoch": best["epoch"]}
    summary["confidence"] = {
        str(threshold): {
            "rows": int(frame.raw_max_softmax.ge(threshold).sum()),
            "wrong": int((frame.raw_max_softmax.ge(threshold) & ~frame.independent_correct).sum()),
        }
        for threshold in (0.9, 0.99)
    }
    if not gzip.decompress((raw / "training.log.gz").read_bytes()):
        raise ValueError(f"Empty training log: {raw.name}")
    write_json(output / "summary.json", summary)
    return frame, summary, config


def compare_runs(frames: dict, configs: dict, output: Path, draws: int) -> dict:
    first, second = configs[BASELINE], configs[UNWEIGHTED]
    differences = {key: [first[key], second[key]] for key in SETTINGS if first[key] != second[key]}
    if differences != {"class_weight": ["balanced", "none"]}:
        raise ValueError(f"Unexpected configuration changes: {differences}")
    if first["source_files_sha256"] != second["source_files_sha256"]:
        raise ValueError("The training runs used different source code.")
    before, after = frames[BASELINE], frames[UNWEIGHTED]
    table = after[["instance_id", "pair_id", "text1", "text2", "label"]].copy()
    for prefix, frame in (("balanced", before), ("none", after)):
        for mode in ("independent", "joint"):
            table[prefix + "_" + mode] = frame[mode]
    summary = {"only_declared_setting_changed": differences, "comparisons": {}}
    for a, b in (
        ("balanced_independent", "none_independent"),
        ("balanced_joint", "none_joint"),
        ("balanced_independent", "none_joint"),
    ):
        old_ok, new_ok = table[a].eq(table.label), table[b].eq(table.label)
        key = a + "_vs_" + b
        table[key + "_effect"] = np.select(
            [~old_ok & new_ok, old_ok & ~new_ok, table[a].ne(table[b])],
            ["fixed", "damaged", "wrong_to_different_wrong"],
            default="unchanged",
        )
        summary["comparisons"][key] = {
            "fixed": int((~old_ok & new_ok).sum()),
            "damaged": int((old_ok & ~new_ok).sum()),
        }
    table.to_csv(output / "balanced_vs_unweighted_all.csv", index=False)
    table.loc[table.balanced_joint.ne(table.none_joint)].to_csv(
        output / "balanced_vs_unweighted_joint_changes.csv", index=False
    )
    table.loc[table.label.eq("NEGATIVE_OTHER") & table.balanced_joint.ne(table.none_joint)].to_csv(
        output / "other_label_changes.csv", index=False
    )

    group_index, groups = pd.factorize(table.pair_id, sort=True)
    count = len(groups)
    weights = np.random.default_rng(2027).multinomial(count, np.full(count, 1 / count), size=draws)
    gold = np.array([LABEL_TO_ID[x] for x in table.label])
    bootstrap = {}
    for name in ("balanced_independent", "balanced_joint", "none_joint"):
        pred = np.array([LABEL_TO_ID[x] for x in table[name]])
        by_group = np.zeros((count, 16), dtype=np.int64)
        np.add.at(by_group, (group_index, gold * 4 + pred), 1)
        confusion = (weights @ by_group).reshape(-1, 4, 4)
        support, predicted = confusion.sum(axis=2), confusion.sum(axis=1)
        tp = np.diagonal(confusion, axis1=1, axis2=2)
        f1 = np.divide(
            2 * tp,
            support + predicted,
            out=np.zeros_like(tp, dtype=float),
            where=(support + predicted) > 0,
        )
        bootstrap[name] = (support * f1).sum(axis=1) / support.sum(axis=1)
    summary["paired_cluster_bootstrap"] = {
        "draws": draws,
        "seed": 2027,
        "unit": "pair_id",
        "source_units": count,
        "limitation": "Conditional dev uncertainty; no seed variance or selection correction",
        **{
            name: (
                100 * np.quantile(bootstrap["none_joint"] - bootstrap[name], [0.025, 0.975])
            ).tolist()
            for name in ("balanced_independent", "balanced_joint")
        },
    }
    write_json(output / "comparison_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=REPO / "docs/experiments")
    parser.add_argument("--data-dir", type=Path, default=REPO / "data/raw")
    parser.add_argument("--official-cache", type=Path, default=REPO / ".cache/official_dico_nli")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/analyses")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive.")
    manifest_path = args.evidence_dir / "manifest.json"
    if manifest_path.is_file():
        for name, expected in read_json(manifest_path).items():
            if fingerprint(args.evidence_dir / name)["sha256"] != expected:
                raise ValueError(f"Archived evidence differs from manifest: {name}")
    paths = {
        "train": args.data_dir / "train/dico_nli_train_track1_participant_labeled.csv",
        "dev": args.data_dir / "dev/dico_nli_dev_track1_participant_labeled.csv",
        "dev_reference": args.data_dir / "dev/dico_nli_dev_track1_reference.csv",
    }
    train, dev = (load_examples(paths[name], require_labels=True) for name in ("train", "dev"))
    reference = pd.read_csv(paths["dev_reference"], dtype=str, keep_default_na=False)
    split_audit = validate_split_integrity(train, dev)
    output = create_run(args.output_dir)
    write_json(output / "split_audit.json", split_audit)
    frames, summaries, configs, rows = {}, {}, {}, []
    for run_id in (BASELINE, UNWEIGHTED):
        frames[run_id], summaries[run_id], configs[run_id] = analyze_run(
            args.evidence_dir / run_id,
            output / run_id / "artifacts",
            paths,
            dev,
            reference,
            args.official_cache,
        )
        for mode, metrics in summaries[run_id]["scores"].items():
            rows.append(
                {
                    "run_id": run_id,
                    "class_weight": configs[run_id]["class_weight"],
                    "decode": mode,
                    **{
                        key: metrics[key]
                        for key in (
                            "weighted_f1",
                            "soft_cons",
                            "hard_cons",
                            "accuracy",
                            "correct_rows",
                        )
                    },
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output / "metrics.csv", index=False)
    compare_runs(frames, configs, output, args.bootstrap_draws)
    write_json(output / "run_summaries.json", summaries)
    write_json(
        output / "evidence_manifest.json",
        {
            str(path.relative_to(args.evidence_dir)): fingerprint(path)["sha256"]
            for run_id in (BASELINE, UNWEIGHTED)
            for path in sorted((args.evidence_dir / run_id).rglob("*"))
            if path.is_file()
        },
    )
    print(table.to_string(index=False))
    print(f"Verified both runs and wrote regenerated analysis to {output}")


if __name__ == "__main__":
    main()
