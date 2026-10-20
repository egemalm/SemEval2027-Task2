import pandas as pd
import pytest

from dico_nli.metrics import score_predictions


def reference_file(tmp_path):
    path = tmp_path / "reference.csv"
    pd.DataFrame(
        {
            "instance_id": ["a", "b", "negative"],
            "pair_id": ["one", "one", "other"],
            "reverse_pair_id": ["b", "a", ""],
            "label": ["FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"],
        }
    ).to_csv(path, index=False)
    return path


def test_perfect_predictions_score_perfectly(tmp_path) -> None:
    predictions = pd.DataFrame(
        {
            "instance_id": ["a", "b", "negative"],
            "label": ["FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"],
        }
    )

    report = score_predictions(reference_file(tmp_path), predictions)

    assert report["weighted_f1"] == 1.0
    assert report["soft_cons"] == 1.0
    assert report["hard_cons"] == 1.0
    assert report["reversible_pairs"] == 1


def test_scorer_rejects_missing_instance_ids(tmp_path) -> None:
    predictions = pd.DataFrame(
        {
            "instance_id": ["a", "b"],
            "label": ["FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT"],
        }
    )

    with pytest.raises(ValueError, match="do not match"):
        score_predictions(reference_file(tmp_path), predictions)
