"""Create a valid DiCo-NLI submission from a saved model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import PairDataset, load_examples
from .labels import LABELS
from .train import build_predictions, predict_logits, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Defaults to the training checkpoint's saved length (legacy: 128).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--decode", choices=("joint", "independent"), default="joint")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if args.output_file.exists() and not args.force:
        raise FileExistsError(f"Prediction file already exists: {args.output_file}")

    set_seed(args.seed)
    device = select_device(args.device)
    frame = load_examples(args.input_file)
    # Inference never consumes labels or gold reverse-pair links, even if supplied.
    frame = frame.drop(columns=["label", "reverse_pair_id"], errors="ignore")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    if [model.config.id2label.get(i) for i in range(len(LABELS))] != list(LABELS):
        raise ValueError("Checkpoint label order does not match this project's four task labels.")
    inference_config = args.model_dir / "inference_config.json"
    saved = json.loads(inference_config.read_text()) if inference_config.is_file() else {}
    max_length = args.max_length if args.max_length is not None else saved.get("max_length", 128)
    if max_length < 1:
        raise ValueError("max-length must be positive.")
    dataset = PairDataset(frame, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    logits = predict_logits(model, loader, device)
    predictions = build_predictions(frame, logits, args.decode)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False, lineterminator="\n")
    print(f"Wrote {len(predictions)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
