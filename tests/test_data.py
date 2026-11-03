import pandas as pd
import pytest

from dico_nli.data import load_examples, validate_split_integrity
from dico_nli.labels import ID_TO_LABEL, LABEL_TO_ID, LABELS, reverse_label


def test_label_mapping_is_invertible_and_reversal_is_an_involution():
    assert reverse_label("EQUIVALENCE") == "EQUIVALENCE"
    assert reverse_label("FORWARD_ENTAILMENT") == "BACKWARD_ENTAILMENT"
    assert reverse_label("BACKWARD_ENTAILMENT") == "FORWARD_ENTAILMENT"
    assert [ID_TO_LABEL[LABEL_TO_ID[label]] for label in LABELS] == list(LABELS)
    for label in LABELS[:3]:
        assert reverse_label(reverse_label(label)) == label
    with pytest.raises(ValueError, match="reversible"):
        reverse_label("NEGATIVE_OTHER")


def test_unlabeled_csv_preserves_numeric_looking_ids_and_literal_na(tmp_path):
    path = tmp_path / "unlabeled.csv"
    path.write_text("instance_id,text1,text2\n001,NA,something\n002,a,b\n")
    frame = load_examples(path)
    assert frame.instance_id.tolist() == ["001", "002"]
    assert frame.text1.tolist() == ["NA", "a"]
    assert "label" not in frame


@pytest.mark.parametrize("labels", [["EQUIVALENCE", ""], ["BAD", ""]])
def test_partial_or_invalid_labels_rejected(tmp_path, labels):
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {"instance_id": ["a", "b"], "text1": ["a", "b"], "text2": ["b", "a"], "label": labels}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_examples(path)


@pytest.mark.parametrize("column", ["instance_id", "pair_id"])
def test_split_overlap_rejected_including_reversed_source_pair(column):
    train = pd.DataFrame(
        {"instance_id": ["a"], "pair_id": ["source"], "text1": ["a"], "text2": ["b"]}
    )
    dev = pd.DataFrame(
        {"instance_id": ["c"], "pair_id": ["different"], "text1": ["b"], "text2": ["a"]}
    )
    dev[column] = train[column]
    with pytest.raises(ValueError, match=f"{column} overlap"):
        validate_split_integrity(train, dev)


def test_repeated_text_reported_but_official_splits_preserved():
    train = pd.DataFrame({"instance_id": ["a"], "pair_id": ["one"], "text1": ["a"], "text2": ["b"]})
    dev = pd.DataFrame({"instance_id": ["b"], "pair_id": ["two"], "text1": ["b"], "text2": ["a"]})
    with pytest.warns(UserWarning, match="share 1"):
        result = validate_split_integrity(train, dev)
    assert result["shared_unordered_text_pairs"] == [("a", "b")]
    assert len(train) == len(dev) == 1
