import numpy as np
import pandas as pd

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
