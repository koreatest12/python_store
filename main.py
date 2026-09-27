from __future__ import annotations

import argparse
import platform
import sys

from tokenizer import SimpleTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="python_store utility")
    parser.add_argument(
        "text",
        nargs="?",
        default="안녕하세요 Python tokenizer 3.14!",
        help="Text to tokenize",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase English text before tokenization",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("python_store")
    print(f"Python: {platform.python_version()}")
    print(f"Executable: {sys.executable}")

    tokenizer = SimpleTokenizer(lowercase=args.lowercase)
    tokenizer.fit([args.text])

    tokens = tokenizer.tokenize(args.text)
    token_ids = tokenizer.encode(args.text)

    print(f"Input: {args.text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Vocabulary: {tokenizer.token_to_id}")


if __name__ == "__main__":
    main()
