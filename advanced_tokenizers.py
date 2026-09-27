from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from text_utils import normalize_text, split_unicode_tokens


def basic_words(text: str, lowercase: bool = False) -> list[str]:
    return split_unicode_tokens(normalize_text(text, lowercase=lowercase))


@dataclass(slots=True)
class BPETokenizer:
    lowercase: bool = False
    vocab_size: int = 200
    end_of_word: str = "</w>"
    merges: list[tuple[str, str]] = field(default_factory=list)
    token_to_id: dict[str, int] = field(default_factory=dict)

    def train(self, texts: Iterable[str]) -> None:
        corpus: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            for word in basic_words(text, self.lowercase):
                corpus[tuple(word) + (self.end_of_word,)] += 1

        symbols = {symbol for word in corpus for symbol in word}
        self.merges = []
        while len(symbols) < self.vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for word, freq in corpus.items():
                for left, right in zip(word, word[1:]):
                    pair_counts[(left, right)] += freq
            if not pair_counts:
                break
            best_pair, _ = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
            merged_symbol = "".join(best_pair)
            self.merges.append(best_pair)

            new_corpus: Counter[tuple[str, ...]] = Counter()
            for word, freq in corpus.items():
                out: list[str] = []
                i = 0
                while i < len(word):
                    if i + 1 < len(word) and (word[i], word[i + 1]) == best_pair:
                        out.append(merged_symbol)
                        i += 2
                    else:
                        out.append(word[i])
                        i += 1
                new_corpus[tuple(out)] += freq
            corpus = new_corpus
            symbols.add(merged_symbol)

        counts: Counter[str] = Counter()
        for word, freq in corpus.items():
            for token in word:
                counts[token] += freq
        ordered = ["<PAD>", "<UNK>"] + [
            token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if token not in {"<PAD>", "<UNK>"}
        ]
        self.token_to_id = {token: i for i, token in enumerate(ordered[: self.vocab_size])}

    def tokenize_word(self, word: str) -> list[str]:
        symbols = list(word) + [self.end_of_word]
        for pair in self.merges:
            merged_symbol = "".join(pair)
            out: list[str] = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                    out.append(merged_symbol)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return symbols

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for word in basic_words(text, self.lowercase):
            out.extend(self.tokenize_word(word))
        return out

    def encode(self, text: str, *, max_length: int | None = None) -> list[int]:
        unk = self.token_to_id.get("<UNK>", 1)
        ids = [self.token_to_id.get(token, unk) for token in self.tokenize(text)]
        return ids if max_length is None else ids[:max_length]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "type": "bpe", "lowercase": self.lowercase, "vocab_size": self.vocab_size,
            "end_of_word": self.end_of_word, "merges": self.merges, "token_to_id": self.token_to_id
        }, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(slots=True)
class WordPieceTokenizer:
    """Educational lightweight WordPiece-style tokenizer.

    This is intentionally simpler than the likelihood-based algorithm used by
    production WordPiece implementations.
    """

    lowercase: bool = False
    vocab_size: int = 200
    prefix: str = "##"
    token_to_id: dict[str, int] = field(default_factory=dict)

    def train(self, texts: Iterable[str]) -> None:
        word_counts: Counter[str] = Counter()
        for text in texts:
            word_counts.update(basic_words(text, self.lowercase))

        pieces: Counter[str] = Counter()
        for word, freq in word_counts.items():
            if not word:
                continue
            pieces[word[0]] += freq
            for ch in word[1:]:
                pieces[self.prefix + ch] += freq

        candidates = Counter(pieces)
        for word, freq in word_counts.items():
            if len(word) >= 2:
                candidates[word] += freq * len(word)

        ordered = ["<PAD>", "<UNK>"] + [
            token for token, _ in sorted(candidates.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
            if token not in {"<PAD>", "<UNK>"}
        ]
        self.token_to_id = {token: i for i, token in enumerate(ordered[: self.vocab_size])}

    def tokenize_word(self, word: str) -> list[str]:
        if word in self.token_to_id:
            return [word]
        pieces: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            current = None
            while end > start:
                piece = word[start:end]
                candidate = piece if start == 0 else self.prefix + piece
                if candidate in self.token_to_id:
                    current = candidate
                    break
                end -= 1
            if current is None:
                return ["<UNK>"]
            pieces.append(current)
            start = end
        return pieces

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for word in basic_words(text, self.lowercase):
            out.extend(self.tokenize_word(word))
        return out

    def encode(self, text: str, *, max_length: int | None = None) -> list[int]:
        unk = self.token_to_id.get("<UNK>", 1)
        ids = [self.token_to_id.get(token, unk) for token in self.tokenize(text)]
        return ids if max_length is None else ids[:max_length]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "type": "wordpiece", "lowercase": self.lowercase, "vocab_size": self.vocab_size,
            "prefix": self.prefix, "token_to_id": self.token_to_id
        }, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(slots=True)
class SentencePieceStyleTokenizer:
    lowercase: bool = False
    vocab_size: int = 200
    boundary: str = "▁"
    token_to_id: dict[str, int] = field(default_factory=dict)

    def train(self, texts: Iterable[str]) -> None:
        counts: Counter[str] = Counter()
        required_chars: Counter[str] = Counter()
        for text in texts:
            normalized = normalize_text(text, lowercase=self.lowercase)
            for word in normalized.split():
                marked = self.boundary + word
                required_chars.update(marked)
                for n in range(1, min(len(marked), 8) + 1):
                    for i in range(len(marked) - n + 1):
                        counts[marked[i:i+n]] += 1

        required = [token for token, _ in sorted(required_chars.items(), key=lambda item: (-item[1], item[0]))]
        if len(required) + 2 > self.vocab_size:
            raise ValueError(
                f"vocab_size={self.vocab_size} is too small for required character coverage "
                f"({len(required)} characters + 2 special tokens)"
            )

        scored = sorted(
            counts.items(),
            key=lambda item: (-(item[1] * math.log2(len(item[0]) + 1)), -len(item[0]), item[0]),
        )
        ordered = ["<PAD>", "<UNK>"] + required
        for token, _ in scored:
            if token not in ordered:
                ordered.append(token)
            if len(ordered) >= self.vocab_size:
                break
        self.token_to_id = {token: i for i, token in enumerate(ordered)}

    def tokenize(self, text: str) -> list[str]:
        normalized = normalize_text(text, lowercase=self.lowercase)
        tokens: list[str] = []
        for word in normalized.split():
            marked = self.boundary + word
            i = 0
            while i < len(marked):
                match = None
                for end in range(len(marked), i, -1):
                    piece = marked[i:end]
                    if piece in self.token_to_id:
                        match = piece
                        break
                if match is None:
                    tokens.append("<UNK>")
                    i += 1
                else:
                    tokens.append(match)
                    i += len(match)
        return tokens

    def encode(self, text: str, *, max_length: int | None = None) -> list[int]:
        unk = self.token_to_id.get("<UNK>", 1)
        ids = [self.token_to_id.get(token, unk) for token in self.tokenize(text)]
        return ids if max_length is None else ids[:max_length]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "type": "sentencepiece-style", "lowercase": self.lowercase, "vocab_size": self.vocab_size,
            "boundary": self.boundary, "token_to_id": self.token_to_id
        }, ensure_ascii=False, indent=2), encoding="utf-8")


def train_tokenizer(kind: str, texts: Iterable[str], *, vocab_size: int = 200, lowercase: bool = False):
    kind = kind.lower()
    if kind == "bpe":
        tokenizer = BPETokenizer(lowercase=lowercase, vocab_size=vocab_size)
    elif kind == "wordpiece":
        tokenizer = WordPieceTokenizer(lowercase=lowercase, vocab_size=vocab_size)
    elif kind in {"sentencepiece", "sentencepiece-style", "sp"}:
        tokenizer = SentencePieceStyleTokenizer(lowercase=lowercase, vocab_size=vocab_size)
    else:
        raise ValueError(f"unsupported tokenizer kind: {kind}")
    tokenizer.train(texts)
    return tokenizer
