"""Compare completed actual training runs and decoding methods as a CSV table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def compare_runs(root: Path, include_smoke: bool = False) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("run_config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        run_dir = path.parent
        status_file = run_dir / "status.json"
        if not status_file.is_file():
            continue
        status = json.loads(status_file.read_text())
        if status["status"] != "completed":
            continue
        if config["run_kind"] != "training" and not include_smoke:
            continue
        for decode in ("independent", "joint"):
            scores = run_dir / f"official_{decode}" / "scores.json"
            if not scores.is_file():
                continue
            metrics = json.loads(scores.read_text())
            rows.append(
                {
                    "run_id": config["run_id"],
                    "decode": decode,
                    "run_kind": config["run_kind"],
                    **{name: metrics[name] for name in ("weighted_f1", "soft_cons", "hard_cons")},
                    "model": config["model_name"],
                    "revision": config["resolved_model_revision"],
                    **{
                        name: config[name]
                        for name in (
                            "seed",
                            "learning_rate",
                            "weight_decay",
                            "warmup_ratio",
                            "epochs",
                            "patience",
                            "batch_size",
                            "max_length",
                            "class_weight",
                            "selection_metric",
                            "git_commit",
                            "git_dirty",
                            "deterministic",
                        )
                    },
                    "config_file": str(path),
                    "train_sha256": config["datasets"]["train"]["sha256"],
                    "dev_sha256": config["datasets"]["dev"]["sha256"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/experiments"))
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument(
        "--diff",
        nargs=2,
        metavar=("RUN_ID_A", "RUN_ID_B"),
        help="Print exact configuration differences between two runs.",
    )
    args = parser.parse_args()
    if args.diff:
        configs = []
        for run_id in args.diff:
            paths = list(args.results_dir.glob(f"{run_id}/run_config.json"))
            if len(paths) != 1:
                parser.error(f"Expected exactly one configuration for {run_id}.")
            configs.append(json.loads(paths[0].read_text()))
        first, second = configs
        print(
            json.dumps(
                {
                    key: {"a": first.get(key), "b": second.get(key)}
                    for key in sorted(first.keys() | second.keys())
                    if first.get(key) != second.get(key)
                },
                indent=2,
            )
        )
        return
    frame = compare_runs(args.results_dir, args.include_smoke)
    if frame.empty:
        print("No completed training runs with official metrics found.")
        return
    print(frame.to_csv(index=False), end="")
    for hashes, group in frame.groupby(["train_sha256", "dev_sha256"]):
        print(f"Dataset group: train={hashes[0][:12]} dev={hashes[1][:12]}", file=sys.stderr)
        for metric in ("weighted_f1", "soft_cons", "hard_cons"):
            best = group.loc[group[metric].idxmax()]
            print(
                f"Best {metric}: {best['run_id']} / {best['decode']} = {best[metric]:.6f}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
