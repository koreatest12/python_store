from __future__ import annotations

import argparse
from pathlib import Path

from advanced_tokenizers import train_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tokenizer vocabulary")
    parser.add_argument("kind", choices=["bpe", "wordpiece", "sentencepiece"])
    parser.add_argument("input", type=Path, help="UTF-8 text file, one sample per line")
    parser.add_argument("--output", type=Path, default=Path("tokenizer_model.json"))
    parser.add_argument("--vocab-size", type=int, default=200)
    parser.add_argument("--lowercase", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    texts = [
        line.strip()
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer = train_tokenizer(
        args.kind,
        texts,
        vocab_size=args.vocab_size,
        lowercase=args.lowercase,
    )
    tokenizer.save(args.output)
    print(f"Saved {args.kind} tokenizer to {args.output}")
    print(f"Vocabulary size: {len(tokenizer.token_to_id)}")


if __name__ == "__main__":
    main()
