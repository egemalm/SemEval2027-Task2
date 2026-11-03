"""Fine-tune a sequence classifier for the DiCo-NLI English track."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

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

from .data import PairDataset, load_examples, validate_split_integrity
from .decode import decode_logits
from .experiments import (
    capture_environment,
    create_run,
    dataset_provenance,
    fingerprint,
    training_log,
    write_json,
)
from .labels import ID_TO_LABEL, LABELS
from .metrics import score_predictions
from .score import run_official_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, help="Matching organizer dev reference CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments"),
        help="Parent directory; each invocation creates a unique run subdirectory.",
    )
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--official-cache", type=Path, default=Path(".cache/official_dico_nli"))
    parser.add_argument("--run-kind", choices=("training", "cpu_smoke"), default="training")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Require deterministic Torch operations; unsupported kernels fail.",
    )
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
        raise RuntimeError(
            f"Requested device '{requested}' is not available. "
            "Check the PyTorch/driver installation or select --device cpu."
        )
    return torch.device(requested)


def set_seed(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


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
    return pd.DataFrame(
        {
            "instance_id": frame["instance_id"],
            "label": decode_logits(frame, logits, strategy=strategy),
        }
    )


def score_dev(
    dev_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    reference_file: Path | None,
) -> dict:
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
    return {"weighted_f1": weighted_f1, "soft_cons": None, "hard_cons": None, "mean3": None}


def calculate_class_weights(frame: pd.DataFrame, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    counts = frame["label"].value_counts()
    total = len(frame)
    return torch.tensor(
        [total / (len(LABELS) * counts.get(ID_TO_LABEL[index], 1)) for index in range(len(LABELS))],
        dtype=torch.float,
    )


def run_training(args: argparse.Namespace, run_dir: Path, config: dict) -> None:
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch-size must be positive.")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0, 1).")
    if args.max_length < 1 or args.num_workers < 0 or args.learning_rate <= 0:
        raise ValueError("max-length/lr must be positive; num-workers must be nonnegative.")
    if args.selection_metric != "weighted_f1" and args.reference_file is None:
        raise ValueError("A reference file is required for consistency-aware checkpoint selection.")

    set_seed(args.seed, args.deterministic)
    device = select_device(args.device)
    train_frame = load_examples(args.train_file, require_labels=True)
    dev_frame = load_examples(args.dev_file, require_labels=True)
    split_audit = validate_split_integrity(train_frame, dev_frame)
    write_json(run_dir / "split_audit.json", split_audit)
    best_model_dir = run_dir / "best_model"
    config.update(capture_environment(run_dir, device))
    config["datasets"] = {
        "train": dataset_provenance(args.train_file),
        "dev": dataset_provenance(args.dev_file),
    }
    if args.reference_file:
        config["datasets"]["dev_reference"] = dataset_provenance(args.reference_file)
        scorer_dir = args.official_cache / "evaluation_functions"
        if not (scorer_dir / "__main__.py").is_file():
            raise FileNotFoundError(
                "Official scorer missing. Run scripts/fetch_official_assets.py "
                "--include-scorer before training."
            )
        config["official_scorer_files"] = {
            p.name: fingerprint(p)["sha256"] for p in sorted(scorer_dir.glob("*.py"))
        }
        manifest = args.official_cache / "provenance.json"
        if manifest.is_file():
            provenance = json.loads(manifest.read_text(encoding="utf-8"))
            config["official_scorer_provenance"] = provenance
            for name, digest in config["official_scorer_files"].items():
                declared = provenance["files"].get(f"evaluation_functions/{name}", {})
                if declared.get("sha256") != digest:
                    raise ValueError(
                        f"Cached official scorer {name} differs from its provenance. "
                        "Inspect and re-fetch before training."
                    )
        # Validate the matching reference before downloading/loading model weights.
        score_predictions(args.reference_file, dev_frame[["instance_id", "label"]])
        reference = pd.read_csv(args.reference_file, dtype=str, keep_default_na=False)
        aligned = reference.set_index("instance_id").loc[dev_frame.instance_id]
        for column in ("label", "text1", "text2", "pair_id"):
            if column in aligned and column in dev_frame:
                if aligned[column].tolist() != dev_frame[column].tolist():
                    raise ValueError(f"Development reference {column} does not match dev examples.")
    write_json(run_dir / "run_config.json", config)

    # Resolve a mutable model reference once, then load tokenizer and weights from that commit.
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(args.model_name, revision=args.model_revision)
    resolved_revision = getattr(model_config, "_commit_hash", None)
    config["resolved_model_revision"] = resolved_revision or "local_snapshot"
    if Path(args.model_name).is_dir():
        config["local_model_files"] = {
            str(p.relative_to(args.model_name)): fingerprint(p)["sha256"]
            for p in sorted(Path(args.model_name).rglob("*"))
            if p.is_file()
        }
    load_revision = resolved_revision or args.model_revision
    write_json(run_dir / "run_config.json", config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, revision=load_revision, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        revision=load_revision,
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

    config.update(
        {
            "label_order": list(LABELS),
            "train_rows": len(train_frame),
            "dev_rows": len(dev_frame),
            "checkpoint_location": str(best_model_dir.resolve()),
        }
    )
    write_json(run_dir / "run_config.json", config)
    print(f"Run: {run_dir} | device={device} | kind={args.run_kind}")

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
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss is non-finite. Inspect inputs and learning rate.")
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
        write_json(run_dir / "metrics_history.json", {"epochs": history})
        print(json.dumps(epoch_record, sort_keys=True))

        score = float(metrics[args.selection_metric])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            write_json(
                best_model_dir / "inference_config.json",
                {"max_length": args.max_length, "label_order": list(LABELS)},
            )
            write_json(run_dir / "best_metrics.json", epoch_record)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    # Release the training model and optimizer before reloading the checkpoint.
    del model, optimizer, scheduler, loss, logits
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)
    final_logits = predict_logits(best_model, dev_loader, device)
    np.save(run_dir / "dev_logits.npy", final_logits)
    for strategy in ("independent", "joint"):
        paired_predictions = build_predictions(dev_frame, final_logits, strategy)
        prediction_path = run_dir / f"predictions_dev_{strategy}.csv"
        paired_predictions.to_csv(prediction_path, index=False, lineterminator="\n")
        local = score_dev(dev_frame, paired_predictions, args.reference_file)
        write_json(run_dir / f"dev_metrics_{strategy}.json", local)
        if args.reference_file:
            official = run_official_scorer(
                args.official_cache,
                args.reference_file,
                prediction_path,
                run_dir / f"official_{strategy}",
            )
            for metric in ("weighted_f1", "soft_cons", "hard_cons"):
                if not np.isclose(local[metric], official[metric], rtol=0, atol=1e-12):
                    raise RuntimeError(f"Local/official scorer mismatch for {metric}.")
    final_predictions = build_predictions(dev_frame, final_logits, args.decode)
    final_predictions.to_csv(run_dir / "predictions_dev.csv", index=False, lineterminator="\n")
    final_metrics = score_dev(dev_frame, final_predictions, args.reference_file)
    write_json(run_dir / "dev_metrics.json", final_metrics)
    print(f"Saved best epoch {best_epoch} and {run_dir / 'predictions_dev.csv'}")
    print(json.dumps(final_metrics, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    run_dir = create_run(args.output_dir)
    config = {
        **vars(args),
        "run_id": run_dir.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "status.json", {"status": "running", "run_kind": args.run_kind})
    with training_log(run_dir):
        try:
            run_training(args, run_dir, config)
        except BaseException as error:
            import traceback

            traceback.print_exc()
            write_json(run_dir / "status.json", {"status": "failed", "error": str(error)})
            if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
                print(
                    "Device memory exhausted. Retry with --batch-size 4 (or 1), "
                    "--max-length 64, or --device cpu."
                )
            raise
        write_json(
            run_dir / "status.json",
            {
                "status": "completed",
                "run_kind": args.run_kind,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )


if __name__ == "__main__":
    main()
