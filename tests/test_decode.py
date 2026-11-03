import numpy as np
import pandas as pd
import pytest

from dico_nli.decode import decode_logits


def test_joint_decoder_repairs_inconsistent_independent_entailment_predictions() -> None:
    frame = pd.DataFrame(
        {
            "instance_id": ["first", "second"],
            "pair_id": ["pair-1", "pair-1"],
            "text1": ["a", "b"],
            "text2": ["b", "a"],
        }
    )
    logits = np.array(
        [
            [0.0, 4.0, 3.0, 0.0],
            [0.0, 6.0, 1.0, 0.0],
        ]
    )

    assert decode_logits(frame, logits, strategy="independent") == [
        "FORWARD_ENTAILMENT",
        "FORWARD_ENTAILMENT",
    ]
    assert decode_logits(frame, logits, strategy="joint") == [
        "BACKWARD_ENTAILMENT",
        "FORWARD_ENTAILMENT",
    ]


def test_joint_decoder_leaves_unpaired_rows_independent() -> None:
    frame = pd.DataFrame(
        {
            "instance_id": ["only"],
            "pair_id": ["incomplete"],
            "text1": ["a"],
            "text2": ["b"],
        }
    )
    logits = np.array([[0.0, 0.0, 0.0, 9.0]])

    assert decode_logits(frame, logits) == ["NEGATIVE_OTHER"]


@pytest.mark.parametrize(
    "winner,expected",
    [
        (0, ["EQUIVALENCE", "EQUIVALENCE"]),
        (3, ["NEGATIVE_OTHER", "NEGATIVE_OTHER"]),
    ],
)
def test_symmetric_assignments_and_no_gold_columns(winner, expected):
    frame = pd.DataFrame({"pair_id": ["p", "p"], "text1": ["a", "b"], "text2": ["b", "a"]})
    logits = np.zeros((2, 4))
    logits[:, winner] = 10
    assert decode_logits(frame, logits) == expected


def test_non_reversed_rows_with_same_pair_id_are_not_forced():
    frame = pd.DataFrame({"pair_id": ["p", "p"], "text1": ["a", "c"], "text2": ["b", "d"]})
    logits = np.array([[0, 10, 1, 0], [0, 10, 1, 0]])
    assert decode_logits(frame, logits) == decode_logits(frame, logits, strategy="independent")


def test_missing_pair_ids_and_shuffled_rows():
    frame = pd.DataFrame({"pair_id": ["p", "p"], "text1": ["a", "b"], "text2": ["b", "a"]})
    logits = np.array([[0, 10, 1, 0], [0, 2, 9, 0]])
    assert decode_logits(frame.iloc[::-1], logits[::-1]) == decode_logits(frame, logits)[::-1]
    assert len(decode_logits(frame.drop(columns="pair_id"), logits)) == 2


def test_nonfinite_logits_rejected():
    with pytest.raises(ValueError, match="finite"):
        decode_logits(
            pd.DataFrame({"text1": ["a"], "text2": ["b"]}), np.array([[0, 1, float("nan"), 0]])
        )
