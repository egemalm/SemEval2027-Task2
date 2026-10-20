"""CSV validation and PyTorch data handling for DiCo-NLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .labels import LABEL_TO_ID, LABELS

REQUIRED_INPUT_COLUMNS = ("instance_id", "text1", "text2")


def load_examples(path: str | Path, *, require_labels: bool = False) -> pd.DataFrame:
    """Load and validate an organizer participant CSV without changing its row order."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")

    frame = pd.read_csv(source, keep_default_na=False)
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{source} contains no instances.")
    if frame["instance_id"].duplicated().any():
        raise ValueError(f"{source} contains duplicate instance_id values.")
    if frame[list(REQUIRED_INPUT_COLUMNS)].eq("").any().any():
        raise ValueError(f"{source} contains blank required input fields.")

    has_labels = "label" in frame.columns and frame["label"].ne("").all()
    if require_labels and not has_labels:
        raise ValueError(f"{source} must contain a non-empty label column.")
    if has_labels:
        unknown = sorted(set(frame["label"]) - set(LABELS))
        if unknown:
            raise ValueError(f"{source} contains unknown labels: {unknown}")

    return frame


class PairDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenized ordered phrase pairs, retaining labels only when supplied."""

    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.encodings = tokenizer(
            self.frame["text1"].tolist(),
            self.frame["text2"].tolist(),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        self.labels = (
            torch.tensor([LABEL_TO_ID[label] for label in self.frame["label"]], dtype=torch.long)
            if "label" in self.frame.columns and self.frame["label"].ne("").all()
            else None
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item
