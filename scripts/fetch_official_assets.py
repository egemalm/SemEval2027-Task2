#!/usr/bin/env python3
"""Fetch pinned public Track 1 materials from the official DiCo-NLI repository."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

OFFICIAL_REPOSITORY = "https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI"
DEFAULT_REVISION = "588968e610197ddc4c440314701cbc587afc4c1b"
RAW_BASE = "https://raw.githubusercontent.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI"
TRACK1_FILES = (
    "final_data/train/dico_nli_train_track1_participant_labeled.csv",
    "final_data/train/dico_nli_train_track1_reference.csv",
    "final_data/train/dico_nli_train_track1_submission_template.csv",
    "final_data/dev/dico_nli_dev_track1_participant_labeled.csv",
    "final_data/dev/dico_nli_dev_track1_reference.csv",
    "final_data/dev/dico_nli_dev_track1_submission_template.csv",
)
SCORER_FILES = (
    "evaluation_functions/__init__.py",
    "evaluation_functions/__main__.py",
    "evaluation_functions/aggregation.py",
    "evaluation_functions/cli.py",
    "evaluation_functions/errors.py",
    "evaluation_functions/io.py",
    "evaluation_functions/labels.py",
    "evaluation_functions/metrics.py",
    "evaluation_functions/schemas.py",
    "evaluation_functions/scorer.py",
    "evaluation_functions/templates.py",
    "evaluation_functions/validation.py",
)


def download(url: str, destination: Path, *, force: bool) -> None:
    if destination.exists() and not force:
        print(f"exists  {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with urlopen(url, timeout=30) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except (HTTPError, URLError, OSError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {url}: {error}") from error
    temporary.replace(destination)
    print(f"fetched  {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--scorer-dir", type=Path, default=Path(".cache/official_dico_nli"))
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--include-scorer", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = list(TRACK1_FILES)
    if args.include_scorer:
        paths.extend(SCORER_FILES)
    for source_path in paths:
        is_scorer = source_path.startswith("evaluation_functions/")
        target_root = args.scorer_dir if is_scorer else args.data_dir
        relative = source_path if is_scorer else source_path.removeprefix("final_data/")
        url = f"{RAW_BASE}/{args.revision}/{source_path}"
        download(url, target_root / relative, force=args.force)
    print(f"Official source: {OFFICIAL_REPOSITORY} @ {args.revision}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
