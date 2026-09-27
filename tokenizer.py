from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_TOKEN_PATTERN = re.compile(
    r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]",
    re.UNICODE,
)


@dataclass(slots=True)
class SimpleTokenizer:
    """A small dependency-free tokenizer for Korean/English text.

    The tokenizer performs Unicode normalization, token splitting, vocabulary
    building, integer encoding/decoding, and JSON persistence.
    """

    lowercase: bool = False
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    token_to_id: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_to_id:
            self.token_to_id = {
                self.pad_token: 0,
                self.unk_token: 1,
            }

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for token, index in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        return normalized.lower() if self.lowercase else normalized

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(self.normalize(text))

    def fit(
        self,
        texts: Iterable[str],
        *,
        min_frequency: int = 1,
        max_vocab_size: int | None = None,
    ) -> None:
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        if max_vocab_size is not None and max_vocab_size < 2:
            raise ValueError("max_vocab_size must be at least 2")

        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        candidates = [
            (token, count)
            for token, count in counter.items()
            if count >= min_frequency
            and token not in {self.pad_token, self.unk_token}
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))

        if max_vocab_size is not None:
            candidates = candidates[: max_vocab_size - 2]

        self.token_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
        }
        for token, _ in candidates:
            self.token_to_id[token] = len(self.token_to_id)

    def encode(self, text: str, *, max_length: int | None = None) -> list[int]:
        token_ids = [
            self.token_to_id.get(token, self.unk_id)
            for token in self.tokenize(text)
        ]

        if max_length is None:
            return token_ids
        if max_length < 0:
            raise ValueError("max_length must be non-negative")

        token_ids = token_ids[:max_length]
        if len(token_ids) < max_length:
            token_ids.extend([self.pad_id] * (max_length - len(token_ids)))
        return token_ids

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        reverse = self.id_to_token
        tokens: list[str] = []
        special_tokens = {self.pad_token, self.unk_token}

        for token_id in token_ids:
            token = reverse.get(token_id, self.unk_token)
            if skip_special_tokens and token in special_tokens:
                continue
            tokens.append(token)

        return " ".join(tokens)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lowercase": self.lowercase,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "token_to_id": self.token_to_id,
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            lowercase=payload["lowercase"],
            pad_token=payload["pad_token"],
            unk_token=payload["unk_token"],
            token_to_id={
                str(token): int(index)
                for token, index in payload["token_to_id"].items()
            },
        )
