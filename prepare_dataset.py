from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset import TextDataset
from pretokenizer import Pretokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and pre-tokenize a text dataset")
    parser.add_argument("input", type=Path)
    parser.add_argument("--format", choices=["text", "jsonl", "csv"], default="text")
    parser.add_argument("--output", type=Path, default=Path("build/pretokenized.jsonl"))
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--strip-accents", action="store_true")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.format == "text":
        dataset = TextDataset.from_text_file(args.input)
    elif args.format == "jsonl":
        dataset = TextDataset.from_jsonl(args.input)
    else:
        dataset = TextDataset.from_csv(
            args.input,
            text_column=args.text_column,
            label_column=args.label_column,
        )

    pretokenizer = Pretokenizer(
        lowercase=args.lowercase,
        strip_accents=args.strip_accents,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(dataset):
            payload = {
                "id": index,
                "text": record.text,
                "label": record.label,
                "tokens": pretokenizer.split(record.text),
                "offsets": [
                    {"text": span.text, "start": span.start, "end": span.end}
                    for span in pretokenizer.split_with_offsets(record.text)
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Prepared {len(dataset)} records")
    print(f"Saved pre-tokenized dataset to {args.output}")


if __name__ == "__main__":
    main()
