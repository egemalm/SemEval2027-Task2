import numpy as np
import pandas as pd
import pytest

from dico_nli.labels import LABELS
from dico_nli.metrics import score_predictions
from dico_nli.score import run_official_scorer


@pytest.fixture
def gold(tmp_path):
    frame = pd.DataFrame(
        {
            "instance_id": ["f", "b", "e1", "e2", "n"],
            "pair_id": ["direction", "direction", "equiv", "equiv", "negative"],
            "reverse_pair_id": ["b", "f", "e2", "e1", ""],
            "label": [
                "FORWARD_ENTAILMENT",
                "BACKWARD_ENTAILMENT",
                "EQUIVALENCE",
                "EQUIVALENCE",
                "NEGATIVE_OTHER",
            ],
        }
    )
    path = tmp_path / "gold.csv"
    frame.to_csv(path, index=False)
    return frame, path


def test_local_and_actual_official_scorer_match_random_and_edge_predictions(
    tmp_path, official_cache, gold
):
    frame, path = gold
    rng = np.random.default_rng(42)
    cases = [
        frame.label.tolist(),
        ["NEGATIVE_OTHER"] * 5,
        ["EQUIVALENCE"] * 5,
        ["FORWARD_ENTAILMENT"] * 5,
    ]
    cases.extend(rng.choice(LABELS, size=5).tolist() for _ in range(8))
    for index, labels in enumerate(cases):
        predictions = pd.DataFrame({"instance_id": frame.instance_id, "label": labels})
        prediction_path = tmp_path / f"predictions_{index}.csv"
        predictions.to_csv(prediction_path, index=False)
        local = score_predictions(path, predictions)
        official = run_official_scorer(
            official_cache, path, prediction_path, tmp_path / f"scores_{index}"
        )
        for metric in ("weighted_f1", "soft_cons", "hard_cons"):
            assert local[metric] == pytest.approx(official[metric], abs=1e-12)
        if index == 1:
            assert local["soft_cons"] == local["hard_cons"] == 0
            assert local["reversible_pairs"] == 2
        if index == 2:
            assert local["soft_cons"] == 1
            assert local["hard_cons"] == 0.5


@pytest.mark.parametrize("mutation", ["self", "different_pair", "unknown_label", "duplicate"])
def test_invalid_reference_rejected(gold, mutation):
    frame, path = gold
    predictions = frame[["instance_id", "label"]].copy()
    if mutation == "self":
        frame.loc[0, "reverse_pair_id"] = "f"
        frame.loc[0, "label"] = "EQUIVALENCE"
    elif mutation == "different_pair":
        frame.loc[0, "pair_id"] = "other"
    elif mutation == "unknown_label":
        frame.loc[4, "label"] = "BAD"
    else:
        frame.loc[4, "instance_id"] = "f"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        score_predictions(path, predictions)


@pytest.mark.parametrize("mutation", ["missing", "extra_column", "duplicate", "invalid_label"])
def test_invalid_submissions_rejected(gold, mutation):
    frame, path = gold
    predictions = frame[["instance_id", "label"]].copy()
    if mutation == "missing":
        predictions = predictions.iloc[:-1]
    elif mutation == "extra_column":
        predictions["score"] = 1
    elif mutation == "duplicate":
        predictions.loc[4, "instance_id"] = "f"
    else:
        predictions.loc[4, "label"] = "BAD"
    with pytest.raises(ValueError):
        score_predictions(path, predictions)
