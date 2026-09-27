from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from text_utils import normalize_text, split_unicode_tokens


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
        normalized = normalize_text(
            text,
            lowercase=self.lowercase,
            strip_accents=self.strip_accents,
        )
        if self.collapse_whitespace:
            normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def split(self, text: str) -> list[str]:
        return split_unicode_tokens(self.normalize(text))

    def split_with_offsets(self, text: str) -> list[PretokenizedSpan]:
        normalized = self.normalize(text)
        tokens = split_unicode_tokens(normalized)
        spans: list[PretokenizedSpan] = []
        cursor = 0
        for token in tokens:
            start = normalized.find(token, cursor)
            if start < 0:
                continue
            end = start + len(token)
            spans.append(PretokenizedSpan(token, start, end))
            cursor = end
        return spans

    def batch(self, texts: Iterable[str]) -> list[list[str]]:
        return [self.split(text) for text in texts]
