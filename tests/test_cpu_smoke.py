"""Actual tiny CPU training/reload/inference; synthetic data, no model downloads."""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from dico_nli.compare import compare_runs
from dico_nli.labels import ID_TO_LABEL, LABEL_TO_ID
from dico_nli.train import score_dev, select_device


def run_cli(*arguments):
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    result = subprocess.run(
        [sys.executable, "-m", *map(str, arguments)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_cpu_training_checkpoint_both_decoders_and_unlabeled_inference(tmp_path, official_cache):
    model_dir = tmp_path / "tiny_model"
    model_dir.mkdir()
    vocab = model_dir / "vocab.txt"
    vocab.write_text(
        "\n".join(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "small",
                "cat",
                "animal",
                "dog",
                "bird",
                "fish",
                "tree",
            ]
        )
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(model_dir)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=12,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=4,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )
    )
    model.save_pretrained(model_dir)

    def pair(prefix, first, second, label):
        inverse = {"FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT", "EQUIVALENCE": "EQUIVALENCE"}
        return [
            dict(
                instance_id=prefix + "a",
                pair_id=prefix,
                text1=first,
                text2=second,
                reverse_pair_id=prefix + "b",
                label=label,
            ),
            dict(
                instance_id=prefix + "b",
                pair_id=prefix,
                text1=second,
                text2=first,
                reverse_pair_id=prefix + "a",
                label=inverse[label],
            ),
        ]

    train = pd.DataFrame(
        pair("train1", "small cat", "cat", "FORWARD_ENTAILMENT")
        + pair("train2", "dog", "small dog", "EQUIVALENCE")
        + [
            dict(
                instance_id="trainn",
                pair_id="trainn",
                text1="fish",
                text2="tree",
                reverse_pair_id="",
                label="NEGATIVE_OTHER",
            )
        ]
    )
    dev = pd.DataFrame(
        pair("dev1", "small bird", "bird", "FORWARD_ENTAILMENT")
        + [
            dict(
                instance_id="0001",
                pair_id="devn",
                text1="dog",
                text2="tree",
                reverse_pair_id="",
                label="NEGATIVE_OTHER",
            )
        ]
    )
    train_path, dev_path, reference_path = [
        tmp_path / name for name in ("train.csv", "dev.csv", "reference.csv")
    ]
    train.drop(columns="reverse_pair_id").to_csv(train_path, index=False)
    dev.drop(columns="reverse_pair_id").to_csv(dev_path, index=False)
    dev.to_csv(reference_path, index=False)
    root = tmp_path / "experiments"
    run_cli(
        "dico_nli.train",
        "--train-file",
        train_path,
        "--dev-file",
        dev_path,
        "--reference-file",
        reference_path,
        "--official-cache",
        official_cache,
        "--model-name",
        model_dir,
        "--output-dir",
        root,
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--max-length",
        "16",
        "--run-kind",
        "cpu_smoke",
        "--deterministic",
    )
    (run_dir,) = list(root.iterdir())
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"
    config = json.loads((run_dir / "run_config.json").read_text())
    for field in ("run_id", "datasets", "git_commit", "hardware", "resolved_model_revision"):
        assert field in config
    assert config["hardware"]["device"] == "cpu"
    assert (run_dir / "training.log").stat().st_size > 0
    assert (run_dir / "source_snapshot/src/dico_nli/train.py").is_file()
    assert compare_runs(root).empty  # Synthetic scores never enter actual-run comparisons.
    assert len(compare_runs(root, include_smoke=True)) == 2
    comparison = run_cli("dico_nli.compare", "--results-dir", root, "--include-smoke")
    assert "independent" in comparison.stdout and "joint" in comparison.stdout
    assert "Best weighted_f1:" in comparison.stderr
    diff = run_cli("dico_nli.compare", "--results-dir", root, "--diff", run_dir.name, run_dir.name)
    assert json.loads(diff.stdout) == {}
    for strategy in ("independent", "joint"):
        predictions = pd.read_csv(run_dir / f"predictions_dev_{strategy}.csv", dtype=str)
        assert predictions.columns.tolist() == ["instance_id", "label"]
        assert predictions.instance_id.tolist() == dev.instance_id.tolist()
        assert (run_dir / f"official_{strategy}/scores.json").is_file()

    unlabeled = tmp_path / "unlabeled.csv"
    dev.drop(columns=["label", "reverse_pair_id", "pair_id"]).to_csv(unlabeled, index=False)
    output = tmp_path / "submission.csv"
    run_cli(
        "dico_nli.predict",
        "--model-dir",
        run_dir / "best_model",
        "--input-file",
        unlabeled,
        "--output-file",
        output,
        "--device",
        "cpu",
    )
    submission = pd.read_csv(output, dtype=str)
    assert submission.columns.tolist() == ["instance_id", "label"]
    assert submission.instance_id.tolist() == dev.instance_id.tolist()
    assert len(submission) == len(dev)


def test_unavailable_metrics_are_missing_not_zero():
    frame = pd.DataFrame({"label": ["EQUIVALENCE"]})
    metrics = score_dev(frame, frame, None)
    assert metrics["weighted_f1"] == 1
    assert metrics["soft_cons"] is metrics["hard_cons"] is metrics["mean3"] is None


def test_cpu_fallback_and_unavailable_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert str(select_device("auto")) == "cpu"
    with pytest.raises(RuntimeError, match="--device cpu"):
        select_device("cuda")
