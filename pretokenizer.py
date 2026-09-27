from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

_TOKEN_PATTERN = re.compile(
    r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]",
    re.UNICODE,
)


@dataclass(slots=True)
class PretokenizedSpan:
    text: str
    start: int
    end: int


@dataclass(slots=True)
class Pretokenizer:
    lowercase: bool = False
    strip_accents: bool = False
    collapse_whitespace: bool = True

    def normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        if self.strip_accents:
            normalized = "".join(
                ch
                for ch in unicodedata.normalize("NFD", normalized)
                if unicodedata.category(ch) != "Mn"
            )
        if self.lowercase:
            normalized = normalized.lower()
        if self.collapse_whitespace:
            normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def split(self, text: str) -> list[str]:
        normalized = self.normalize(text)
        return _TOKEN_PATTERN.findall(normalized)

    def split_with_offsets(self, text: str) -> list[PretokenizedSpan]:
        normalized = self.normalize(text)
        return [
            PretokenizedSpan(match.group(0), match.start(), match.end())
            for match in _TOKEN_PATTERN.finditer(normalized)
        ]

    def batch(self, texts: Iterable[str]) -> list[list[str]]:
        return [self.split(text) for text in texts]
