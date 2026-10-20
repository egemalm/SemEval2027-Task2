"""Local DiCo-NLI-compatible metrics used for checkpoint selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import f1_score

from .labels import LABELS, REVERSAL_MAP


def score_predictions(reference_file: str | Path, predictions: pd.DataFrame) -> dict[str, Any]:
    """Score a complete submission using the published DiCo-NLI metric definition.

    The organizer evaluator remains authoritative. This implementation is kept
    small and is used during training to select checkpoints every epoch.
    """
    reference = pd.read_csv(reference_file, keep_default_na=False)
    required_reference = {"instance_id", "label", "reverse_pair_id"}
    missing = required_reference - set(reference.columns)
    if missing:
        raise ValueError(f"Reference file is missing columns: {sorted(missing)}")
    if list(predictions.columns) != ["instance_id", "label"]:
        raise ValueError("Predictions must have exactly: instance_id,label")
    if reference["instance_id"].duplicated().any() or predictions["instance_id"].duplicated().any():
        raise ValueError("Reference and predictions must have unique instance_id values.")
    if set(predictions["label"]) - set(LABELS):
        raise ValueError("Predictions include an unknown task label.")

    gold_ids = set(reference["instance_id"])
    predicted_ids = set(predictions["instance_id"])
    if gold_ids != predicted_ids:
        missing_ids = len(gold_ids - predicted_ids)
        extra_ids = len(predicted_ids - gold_ids)
        raise ValueError(
            f"Prediction ids do not match gold ids (missing={missing_ids}, extra={extra_ids})."
        )

    predicted_by_id = dict(zip(predictions["instance_id"], predictions["label"], strict=True))
    y_true = reference["label"].tolist()
    y_pred = [predicted_by_id[instance_id] for instance_id in reference["instance_id"]]
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=list(LABELS), average="weighted", zero_division=0)
    )

    gold_by_id = reference.set_index("instance_id", drop=False)
    visited: set[str] = set()
    soft_correct = 0
    hard_correct = 0
    reversible_pairs = 0
    for row in reference.itertuples(index=False):
        reverse_id = row.reverse_pair_id
        if not reverse_id or row.instance_id in visited:
            continue
        if reverse_id not in gold_by_id.index:
            raise ValueError(f"Unknown reverse_pair_id: {reverse_id}")
        reverse_row = gold_by_id.loc[reverse_id]
        if reverse_row.reverse_pair_id != row.instance_id:
            raise ValueError(f"Non-reciprocal pair: {row.instance_id} <-> {reverse_id}")
        if row.label not in REVERSAL_MAP or reverse_row.label != REVERSAL_MAP[row.label]:
            raise ValueError(f"Invalid gold reversal pair: {row.instance_id} <-> {reverse_id}")

        reversible_pairs += 1
        first_prediction = predicted_by_id[row.instance_id]
        second_prediction = predicted_by_id[reverse_id]
        if first_prediction in REVERSAL_MAP and second_prediction == REVERSAL_MAP[first_prediction]:
            soft_correct += 1
        if first_prediction == row.label and second_prediction == reverse_row.label:
            hard_correct += 1
        visited.update((row.instance_id, reverse_id))

    if reversible_pairs == 0:
        raise ValueError("Reference contains no reversible pairs.")
    soft_cons = soft_correct / reversible_pairs
    hard_cons = hard_correct / reversible_pairs
    return {
        "weighted_f1": weighted_f1,
        "soft_cons": soft_cons,
        "hard_cons": hard_cons,
        "mean3": (weighted_f1 + soft_cons + hard_cons) / 3,
        "reversible_pairs": reversible_pairs,
    }
