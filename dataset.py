from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


@dataclass(slots=True)
class DatasetRecord:
    text: str
    label: str | None = None
    metadata: dict | None = None


@dataclass(slots=True)
class DatasetSplit:
    train: list[DatasetRecord]
    validation: list[DatasetRecord]
    test: list[DatasetRecord]


class TextDataset:
    def __init__(self, records: Iterable[DatasetRecord] = ()) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.records)

    def __getitem__(self, index: int) -> DatasetRecord:
        return self.records[index]

    @classmethod
    def from_text_file(cls, path: str | Path) -> "TextDataset":
        records = [
            DatasetRecord(text=line.strip())
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(records)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TextDataset":
        records: list[DatasetRecord] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(
                DatasetRecord(
                    text=str(payload["text"]),
                    label=None if payload.get("label") is None else str(payload["label"]),
                    metadata=payload.get("metadata"),
                )
            )
        return cls(records)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        text_column: str = "text",
        label_column: str | None = "label",
    ) -> "TextDataset":
        records: list[DatasetRecord] = []
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if text_column not in (reader.fieldnames or []):
                raise ValueError(f"missing text column: {text_column}")
            for row in reader:
                text = (row.get(text_column) or "").strip()
                if not text:
                    continue
                label = None
                if label_column and label_column in row:
                    raw = row.get(label_column)
                    label = raw if raw not in {None, ""} else None
                metadata = {
                    key: value
                    for key, value in row.items()
                    if key not in {text_column, label_column}
                } or None
                records.append(DatasetRecord(text=text, label=label, metadata=metadata))
        return cls(records)

    def to_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "\n".join(
                json.dumps(asdict(record), ensure_ascii=False)
                for record in self.records
            )
            + ("\n" if self.records else ""),
            encoding="utf-8",
        )

    def to_csv(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["text", "label", "metadata"])
            writer.writeheader()
            for record in self.records:
                writer.writerow(
                    {
                        "text": record.text,
                        "label": record.label or "",
                        "metadata": json.dumps(record.metadata, ensure_ascii=False)
                        if record.metadata is not None
                        else "",
                    }
                )

    def shuffled(self, *, seed: int = 42) -> "TextDataset":
        records = list(self.records)
        random.Random(seed).shuffle(records)
        return TextDataset(records)

    def split(
        self,
        *,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> DatasetSplit:
        total_ratio = train_ratio + validation_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-9:
            raise ValueError("train/validation/test ratios must sum to 1.0")
        if min(train_ratio, validation_ratio, test_ratio) < 0:
            raise ValueError("split ratios cannot be negative")

        items = self.shuffled(seed=seed).records
        total = len(items)
        train_end = round(total * train_ratio)
        validation_end = train_end + round(total * validation_ratio)

        return DatasetSplit(
            train=items[:train_end],
            validation=items[train_end:validation_end],
            test=items[validation_end:],
        )

    def batches(self, batch_size: int, *, drop_last: bool = False) -> Iterator[list[DatasetRecord]]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        for start in range(0, len(self.records), batch_size):
            batch = self.records[start:start + batch_size]
            if drop_last and len(batch) < batch_size:
                break
            yield batch

    def texts(self) -> list[str]:
        return [record.text for record in self.records]


def build_tokenized_dataset(
    dataset: TextDataset,
    tokenizer,
    *,
    max_length: int | None = None,
) -> list[dict]:
    output: list[dict] = []
    for index, record in enumerate(dataset):
        if max_length is None:
            token_ids = tokenizer.encode(record.text)
        else:
            try:
                token_ids = tokenizer.encode(record.text, max_length=max_length)
            except TypeError:
                token_ids = tokenizer.encode(record.text)[:max_length]

        output.append(
            {
                "id": index,
                "text": record.text,
                "label": record.label,
                "input_ids": token_ids,
            }
        )
    return output
