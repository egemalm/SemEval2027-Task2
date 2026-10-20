"""Task label inventory and directional reversal rules."""

from __future__ import annotations

LABELS = (
    "EQUIVALENCE",
    "FORWARD_ENTAILMENT",
    "BACKWARD_ENTAILMENT",
    "NEGATIVE_OTHER",
)

REVERSAL_MAP = {
    "EQUIVALENCE": "EQUIVALENCE",
    "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT",
    "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT",
}

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def reverse_label(label: str) -> str:
    """Return the task-defined reverse label or raise for NEGATIVE_OTHER."""
    try:
        return REVERSAL_MAP[label]
    except KeyError as error:
        raise ValueError(f"{label!r} does not have a reversible counterpart.") from error
