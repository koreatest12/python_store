from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_WORD_RE = re.compile(r"[가-힣A-Za-z0-9]+|[^\w\s]", re.UNICODE)


def normalize_text(text: str, lowercase: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.lower() if lowercase else text


def basic_words(text: str, lowercase: bool = False) -> list[str]:
    return _WORD_RE.findall(normalize_text(text, lowercase=lowercase))


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
                if len(word) == 1 and not word.isalnum() and not ("가" <= word <= "힣"):
                    corpus[(word, self.end_of_word)] += 1
                else:
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
                merged: list[str] = []
                i = 0
                while i < len(word):
                    if i + 1 < len(word) and (word[i], word[i + 1]) == best_pair:
                        merged.append(merged_symbol)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                new_corpus[tuple(merged)] += freq
            corpus = new_corpus
            symbols.add(merged_symbol)

        final_tokens = Counter()
        for word, freq in corpus.items():
            for token in word:
                final_tokens[token] += freq
        ordered = ["<PAD>", "<UNK>"] + [
            token for token, _ in sorted(final_tokens.items(), key=lambda item: (-item[1], item[0]))
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
        tokens: list[str] = []
        for word in basic_words(text, self.lowercase):
            tokens.extend(self.tokenize_word(word))
        return tokens

    def encode(self, text: str) -> list[int]:
        unk = self.token_to_id.get("<UNK>", 1)
        return [self.token_to_id.get(token, unk) for token in self.tokenize(text)]

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "bpe",
            "lowercase": self.lowercase,
            "vocab_size": self.vocab_size,
            "end_of_word": self.end_of_word,
            "merges": self.merges,
            "token_to_id": self.token_to_id,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(slots=True)
class WordPieceTokenizer:
    lowercase: bool = False
    vocab_size: int = 200
    prefix: str = "##"
    token_to_id: dict[str, int] = field(default_factory=dict)

    def train(self, texts: Iterable[str]) -> None:
        word_counts: Counter[str] = Counter()
        for text in texts:
            for word in basic_words(text, self.lowercase):
                word_counts[word] += 1

        piece_counts: Counter[str] = Counter()
        for word, freq in word_counts.items():
            if not word:
                continue
            piece_counts[word[0]] += freq
            for ch in word[1:]:
                piece_counts[self.prefix + ch] += freq

        candidates: Counter[str] = Counter(piece_counts)
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

    def encode(self, text: str) -> list[int]:
        unk = self.token_to_id.get("<UNK>", 1)
        return [self.token_to_id.get(token, unk) for token in self.tokenize(text)]

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "wordpiece",
            "lowercase": self.lowercase,
            "vocab_size": self.vocab_size,
            "prefix": self.prefix,
            "token_to_id": self.token_to_id,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(slots=True)
class SentencePieceStyleTokenizer:
    lowercase: bool = False
    vocab_size: int = 200
    boundary: str = "▁"
    token_to_id: dict[str, int] = field(default_factory=dict)

    def train(self, texts: Iterable[str]) -> None:
        counts: Counter[str] = Counter()
        for text in texts:
            normalized = normalize_text(text, self.lowercase)
            for word in normalized.split():
                marked = self.boundary + word
                for n in range(1, min(len(marked), 8) + 1):
                    for i in range(0, len(marked) - n + 1):
                        counts[marked[i:i+n]] += 1

        scored = sorted(
            counts.items(),
            key=lambda item: (-(item[1] * math.log2(len(item[0]) + 1)), -len(item[0]), item[0]),
        )
        ordered = ["<PAD>", "<UNK>"] + [
            token for token, _ in scored if token not in {"<PAD>", "<UNK>"}
        ]
        self.token_to_id = {token: i for i, token in enumerate(ordered[: self.vocab_size])}

    def tokenize(self, text: str) -> list[str]:
        normalized = normalize_text(text, self.lowercase)
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

    def encode(self, text: str) -> list[int]:
        unk = self.token_to_id.get("<UNK>", 1)
        return [self.token_to_id.get(token, unk) for token in self.tokenize(text)]

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "sentencepiece-style",
            "lowercase": self.lowercase,
            "vocab_size": self.vocab_size,
            "boundary": self.boundary,
            "token_to_id": self.token_to_id,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_tokenizer(
    kind: str,
    texts: Iterable[str],
    *,
    vocab_size: int = 200,
    lowercase: bool = False,
):
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
