"""Independent and reversal-aware decoding for DiCo-NLI logits."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .labels import ID_TO_LABEL, LABEL_TO_ID, LABELS

JOINT_ASSIGNMENTS = (
    ("EQUIVALENCE", "EQUIVALENCE"),
    ("FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"),
    ("BACKWARD_ENTAILMENT", "FORWARD_ENTAILMENT"),
    ("NEGATIVE_OTHER", "NEGATIVE_OTHER"),
)


def decode_logits(frame: pd.DataFrame, logits: np.ndarray, *, strategy: str = "joint") -> list[str]:
    """Decode class logits, optionally constraining complete source pairs jointly.

    The organizer participant files expose `pair_id` but not `reverse_pair_id`.
    For exactly two rows with the same `pair_id`, the maximum-scoring valid
    assignment is selected from the four possible source-pair relations. Rows
    without a complete pair use their independent classifier argmax.
    """
    if len(frame) != len(logits):
        raise ValueError("The number of logits must equal the number of input rows.")
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] != len(LABELS):
        raise ValueError(f"Expected logits with shape (n, {len(LABELS)}), got {logits.shape}.")

    labels = [ID_TO_LABEL[int(index)] for index in np.argmax(logits, axis=1)]
    if strategy == "independent":
        return labels
    if strategy != "joint":
        raise ValueError("strategy must be 'independent' or 'joint'.")
    if "pair_id" not in frame.columns:
        return labels

    by_pair: dict[str, list[int]] = defaultdict(list)
    for row_index, pair_id in enumerate(frame["pair_id"].astype(str)):
        if pair_id:
            by_pair[pair_id].append(row_index)

    for indices in by_pair.values():
        if len(indices) != 2:
            continue
        first, second = indices
        assignment_scores = [
            logits[first, LABEL_TO_ID[first_label]] + logits[second, LABEL_TO_ID[second_label]]
            for first_label, second_label in JOINT_ASSIGNMENTS
        ]
        first_label, second_label = JOINT_ASSIGNMENTS[int(np.argmax(assignment_scores))]
        labels[first] = first_label
        labels[second] = second_label

    return labels
