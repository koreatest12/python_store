from __future__ import annotations

import unicodedata


def normalize_text(text: str, *, lowercase: bool = False, strip_accents: bool = False) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    if strip_accents:
        normalized = "".join(
            ch
            for ch in unicodedata.normalize("NFD", normalized)
            if unicodedata.category(ch) != "Mn"
        )
        normalized = unicodedata.normalize("NFC", normalized)
    return normalized.lower() if lowercase else normalized


def split_unicode_tokens(text: str) -> list[str]:
    """Split text without silently dropping non-whitespace characters.

    Letters, numbers, underscore and combining marks are grouped as word-like
    tokens. Other non-whitespace characters are emitted individually.
    """
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for ch in text:
        if ch.isspace():
            flush()
            continue

        category = unicodedata.category(ch)
        is_word = (
            category.startswith("L")
            or category.startswith("N")
            or category.startswith("M")
            or ch == "_"
        )

        if is_word:
            current.append(ch)
        else:
            flush()
            tokens.append(ch)

    flush()
    return tokens
