"""Fine-tune a sequence classifier for the DiCo-NLI English track."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .data import PairDataset, load_examples
from .decode import decode_logits
from .labels import ID_TO_LABEL, LABELS
from .metrics import score_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, help="Matching organizer dev reference CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--decode", choices=("joint", "independent"), default="joint")
    parser.add_argument(
        "--selection-metric",
        choices=("mean3", "weighted_f1", "soft_cons", "hard_cons"),
        default="mean3",
    )
    parser.add_argument("--class-weight", choices=("balanced", "none"), default="balanced")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    """Choose a supported accelerator in a portable, explicit order."""
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu": True,
    }
    if requested == "auto":
        requested = next(name for name in ("cuda", "mps", "cpu") if available[name])
    if not available[requested]:
        raise RuntimeError(f"Requested device '{requested}' is not available.")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.inference_mode()
def predict_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for batch in loader:
        features = move_batch_to_device(batch, device)
        features.pop("labels", None)
        outputs.append(model(**features).logits.detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def build_predictions(frame: pd.DataFrame, logits: np.ndarray, strategy: str) -> pd.DataFrame:
    return pd.DataFrame({
        "instance_id": frame["instance_id"],
        "label": decode_logits(frame, logits, strategy=strategy),
    })


def score_dev(
    dev_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    reference_file: Path | None,
) -> dict[str, float | int]:
    if reference_file is not None:
        return score_predictions(reference_file, predictions)
    weighted_f1 = float(
        f1_score(
            dev_frame["label"],
            predictions["label"],
            labels=list(LABELS),
            average="weighted",
            zero_division=0,
        )
    )
    return {"weighted_f1": weighted_f1, "soft_cons": 0.0, "hard_cons": 0.0, "mean3": weighted_f1}


def calculate_class_weights(frame: pd.DataFrame, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    counts = frame["label"].value_counts()
    total = len(frame)
    return torch.tensor(
        [total / (len(LABELS) * counts.get(ID_TO_LABEL[index], 1)) for index in range(len(LABELS))],
        dtype=torch.float,
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch-size must be positive.")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0, 1).")
    if args.selection_metric != "weighted_f1" and args.reference_file is None:
        raise ValueError("A reference file is required for consistency-aware checkpoint selection.")

    set_seed(args.seed)
    device = select_device(args.device)
    train_frame = load_examples(args.train_file, require_labels=True)
    dev_frame = load_examples(args.dev_file, require_labels=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = args.output_dir / "best_model"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id={label: index for index, label in ID_TO_LABEL.items()},
    ).to(device)
    train_dataset = PairDataset(train_frame, tokenizer, args.max_length)
    dev_dataset = PairDataset(dev_frame, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    weights = calculate_class_weights(train_frame, args.class_weight == "balanced")
    loss_function = CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)

    config = vars(args).copy()
    config.update(
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "label_order": list(LABELS),
            "train_rows": len(train_frame),
            "dev_rows": len(dev_frame),
        }
    )
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in config.items()
    }
    write_json(args.output_dir / "run_config.json", config)

    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            batch = move_batch_to_device(batch, device)
            targets = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = loss_function(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        logits = predict_logits(model, dev_loader, device)
        predictions = build_predictions(dev_frame, logits, args.decode)
        metrics = score_dev(dev_frame, predictions, args.reference_file)
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            **metrics,
        }
        history.append(epoch_record)
        write_json(args.output_dir / "metrics_history.json", {"epochs": history})
        print(
            "epoch={epoch} loss={loss:.4f} weighted_f1={weighted:.4f} soft_cons={soft:.4f} "
            "hard_cons={hard:.4f} mean3={mean3:.4f}".format(
                epoch=epoch,
                loss=epoch_record["train_loss"],
                weighted=metrics["weighted_f1"],
                soft=metrics["soft_cons"],
                hard=metrics["hard_cons"],
                mean3=metrics["mean3"],
            )
        )

        score = float(metrics[args.selection_metric])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            write_json(args.output_dir / "best_metrics.json", epoch_record)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)
    final_logits = predict_logits(best_model, dev_loader, device)
    final_predictions = build_predictions(dev_frame, final_logits, args.decode)
    final_predictions.to_csv(
        args.output_dir / "predictions_dev.csv", index=False, lineterminator="\n"
    )
    final_metrics = score_dev(dev_frame, final_predictions, args.reference_file)
    write_json(args.output_dir / "dev_metrics.json", final_metrics)
    print(f"Saved best epoch {best_epoch} and {args.output_dir / 'predictions_dev.csv'}")
    print(json.dumps(final_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
