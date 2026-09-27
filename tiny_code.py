from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class TinyCodeArtifact:
    name: str
    path: str
    template: str
    generated_at: str
    valid_python: bool


class TinyCodeGenerator:
    """Generate small dependency-free Python examples for this repository."""

    def __init__(self, output_dir: str | Path = "bot_workspace/tiny_code") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _render(self, template: str) -> str:
        if template == "tokenize":
            return '''from tokenizer import SimpleTokenizer


def main() -> None:
    text = "안녕하세요 Python tokenizer 3.14!"
    tokenizer = SimpleTokenizer(lowercase=True)
    tokenizer.fit([text])

    print("tokens:", tokenizer.tokenize(text))
    print("ids:", tokenizer.encode(text))


if __name__ == "__main__":
    main()
'''
        if template == "pretokenize":
            return '''from pretokenizer import Pretokenizer


def main() -> None:
    text = "ㅋㅋㅋ café 東京 snake_case 😀"
    pre = Pretokenizer(lowercase=True)

    print("tokens:", pre.split(text))
    for span in pre.split_with_offsets(text):
        print(span)


if __name__ == "__main__":
    main()
'''
        if template == "dataset":
            return '''from dataset import TextDataset


def main() -> None:
    dataset = TextDataset.from_jsonl("examples/dataset.jsonl")
    split = dataset.split(seed=42)

    print("records:", len(dataset))
    print("train:", len(split.train))
    print("validation:", len(split.validation))
    print("test:", len(split.test))


if __name__ == "__main__":
    main()
'''
        if template == "train-bpe":
            return '''from advanced_tokenizers import train_tokenizer


def main() -> None:
    corpus = [
        line.strip()
        for line in open("examples/corpus.txt", encoding="utf-8")
        if line.strip()
    ]
    tokenizer = train_tokenizer("bpe", corpus, vocab_size=128)
    print("vocab size:", len(tokenizer.token_to_id))
    print("tokens:", tokenizer.tokenize("안녕하세요 Python"))


if __name__ == "__main__":
    main()
'''
        if template == "repository":
            return '''from token_repository import TokenRepository


def main() -> None:
    repository = TokenRepository()
    print("valid:", repository.verify())
    print("models:", repository.list())


if __name__ == "__main__":
    main()
'''
        if template == "bot-cycle":
            return '''from bot import DataTokenBot


def main() -> None:
    report = DataTokenBot().cycle(vocab_size=128)
    print("status:", report.status)
    print("failed stages:", report.details["failed_stage_count"])


if __name__ == "__main__":
    main()
'''
        raise ValueError(f"unsupported tiny code template: {template}")

    def generate(self, template: str, *, name: str | None = None) -> TinyCodeArtifact:
        code = self._render(template)
        ast.parse(code)

        safe_name = name or template.replace("-", "_")
        if not safe_name.replace("_", "").isalnum():
            raise ValueError("tiny code name must contain only letters, numbers, and underscore")

        target = self.output_dir / f"{safe_name}.py"
        target.write_text(code, encoding="utf-8")

        return TinyCodeArtifact(
            name=safe_name,
            path=target.as_posix(),
            template=template,
            generated_at=datetime.now(timezone.utc).isoformat(),
            valid_python=True,
        )

    def generate_all(self) -> list[TinyCodeArtifact]:
        templates = [
            "tokenize",
            "pretokenize",
            "dataset",
            "train-bpe",
            "repository",
            "bot-cycle",
        ]
        return [self.generate(template) for template in templates]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate tiny Python code examples")
    parser.add_argument(
        "template",
        choices=[
            "all",
            "tokenize",
            "pretokenize",
            "dataset",
            "train-bpe",
            "repository",
            "bot-cycle",
        ],
    )
    parser.add_argument("--output-dir", default="bot_workspace/tiny_code")
    parser.add_argument("--name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generator = TinyCodeGenerator(args.output_dir)

    if args.template == "all":
        artifacts = generator.generate_all()
    else:
        artifacts = [generator.generate(args.template, name=args.name)]

    print(json.dumps([asdict(item) for item in artifacts], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
