from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from advanced_tokenizers import train_tokenizer
from dataset import TextDataset
from pretokenizer import Pretokenizer
from token_repository import TokenRepository


@dataclass(slots=True)
class BotReport:
    command: str
    status: str
    details: dict


class DataTokenBot:
    def __init__(self, workspace: str | Path = "bot_workspace") -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.workspace / "models"
        self.datasets_dir = self.workspace / "datasets"
        self.models_dir.mkdir(exist_ok=True)
        self.datasets_dir.mkdir(exist_ok=True)

    def status(self) -> BotReport:
        token_repo = TokenRepository()
        details = {
            "token_repository_valid": token_repo.verify(),
            "registered_models": len(token_repo.list()),
            "workspace": self.workspace.as_posix(),
        }
        return BotReport("status", "ok", details)

    def prepare_dataset(
        self,
        source: str | Path,
        *,
        source_format: str = "jsonl",
        lowercase: bool = False,
    ) -> BotReport:
        source = Path(source)
        if source_format == "jsonl":
            dataset = TextDataset.from_jsonl(source)
        elif source_format == "csv":
            dataset = TextDataset.from_csv(source)
        elif source_format == "text":
            dataset = TextDataset.from_text_file(source)
        else:
            raise ValueError(f"unsupported dataset format: {source_format}")

        pre = Pretokenizer(lowercase=lowercase)
        output = self.datasets_dir / f"{source.stem}.pretokenized.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(dataset):
                handle.write(
                    json.dumps(
                        {
                            "id": index,
                            "text": record.text,
                            "label": record.label,
                            "tokens": pre.split(record.text),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        split = dataset.split(seed=42)
        return BotReport(
            "prepare-dataset",
            "ok",
            {
                "records": len(dataset),
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
                "output": output.as_posix(),
            },
        )

    def train_and_register(
        self,
        corpus: str | Path,
        *,
        kind: str,
        name: str,
        version: str = "1.0.0",
        vocab_size: int = 128,
    ) -> BotReport:
        texts = [
            line.strip()
            for line in Path(corpus).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        tokenizer = train_tokenizer(kind, texts, vocab_size=vocab_size)
        model_path = self.models_dir / f"{name}-{kind}-{version}.json"
        tokenizer.save(model_path)

        repository = TokenRepository()
        artifact = repository.add(
            model_path,
            name=name,
            model_type=kind,
            version=version,
        )

        return BotReport(
            "train-register",
            "ok",
            {
                "model": artifact.file,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "vocab_size": len(tokenizer.token_to_id),
            },
        )

    def export_repository(self) -> BotReport:
        repository = TokenRepository()
        archive = repository.export_download_bundle()
        return BotReport(
            "export",
            "ok",
            {
                "archive": archive.as_posix(),
                "registered_models": len(repository.list()),
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Data and token management bot")
    parser.add_argument("--workspace", default="bot_workspace")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")

    prepare = sub.add_parser("prepare-dataset")
    prepare.add_argument("source")
    prepare.add_argument("--format", choices=["jsonl", "csv", "text"], default="jsonl")
    prepare.add_argument("--lowercase", action="store_true")

    train = sub.add_parser("train-register")
    train.add_argument("corpus")
    train.add_argument("--kind", choices=["bpe", "wordpiece", "sentencepiece"], required=True)
    train.add_argument("--name", required=True)
    train.add_argument("--version", default="1.0.0")
    train.add_argument("--vocab-size", type=int, default=128)

    sub.add_parser("export")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    bot = DataTokenBot(args.workspace)

    if args.command == "status":
        report = bot.status()
    elif args.command == "prepare-dataset":
        report = bot.prepare_dataset(
            args.source,
            source_format=args.format,
            lowercase=args.lowercase,
        )
    elif args.command == "train-register":
        report = bot.train_and_register(
            args.corpus,
            kind=args.kind,
            name=args.name,
            version=args.version,
            vocab_size=args.vocab_size,
        )
    else:
        report = bot.export_repository()

    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
