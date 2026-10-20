"""Score a DiCo-NLI prediction file locally and optionally invoke the official scorer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .metrics import score_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument(
        "--official-cache",
        type=Path,
        help="Cache created with scripts/fetch_official_assets.py --include-scorer.",
    )
    parser.add_argument("--official-output-dir", type=Path)
    return parser.parse_args()


def run_official_scorer(cache: Path, reference: Path, prediction: Path, output_dir: Path) -> None:
    package = cache / "evaluation_functions" / "__main__.py"
    if not package.is_file():
        raise FileNotFoundError(
            f"Official scorer not found in {cache}. Run: make fetch-data"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(cache) + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation_functions",
            "--gold",
            str(reference),
            "--predictions",
            str(prediction),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        env=environment,
    )


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.prediction_file, keep_default_na=False)
    report = score_predictions(args.reference_file, predictions)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.official_cache is not None:
        output_dir = args.official_output_dir or args.prediction_file.with_suffix("").with_name(
            f"{args.prediction_file.stem}_official_scores"
        )
        run_official_scorer(
            args.official_cache,
            args.reference_file,
            args.prediction_file,
            output_dir,
        )
        print(f"Official scorer output written to {output_dir}")


if __name__ == "__main__":
    main()
