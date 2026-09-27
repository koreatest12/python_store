from __future__ import annotations

import re
import unicodedata

_DECIMAL_OR_NUMBER = re.compile(r"\d+(?:[.,]\d+)*", re.UNICODE)


def normalize_text(text: str, *, lowercase: bool = False, strip_accents: bool = False) -> str:
    # Preserve Korean compatibility jamo (e.g. ㅋㅋㅋ, ㅎㅎ, ㅜㅜ) across NFKC,
    # because NFKC otherwise converts them to canonical choseong/jungseong codepoints.
    protected: dict[str, str] = {}
    chars: list[str] = []
    next_private = 0xE000

    for ch in text:
        code = ord(ch)
        if 0x3130 <= code <= 0x318F:
            placeholder = chr(next_private)
            next_private += 1
            protected[placeholder] = ch
            chars.append(placeholder)
        else:
            chars.append(ch)

    normalized = unicodedata.normalize("NFKC", "".join(chars))
    for placeholder, original in protected.items():
        normalized = normalized.replace(placeholder, original)

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

    - Unicode letters/numbers/marks and underscore are grouped.
    - Decimal-like numbers such as 3.14 and 1,000 are preserved.
    - Other non-whitespace characters (including emoji) are emitted individually.
    """
    tokens: list[str] = []
    i = 0

    while i < len(text):
        ch = text[i]

        if ch.isspace():
            i += 1
            continue

        if ch.isdigit():
            match = _DECIMAL_OR_NUMBER.match(text, i)
            if match is not None:
                tokens.append(match.group(0))
                i = match.end()
                continue

        category = unicodedata.category(ch)
        if (
            category.startswith("L")
            or category.startswith("N")
            or category.startswith("M")
            or ch == "_"
        ):
            start = i
            i += 1
            while i < len(text):
                current = text[i]
                current_category = unicodedata.category(current)
                if (
                    current_category.startswith("L")
                    or current_category.startswith("N")
                    or current_category.startswith("M")
                    or current == "_"
                ):
                    i += 1
                else:
                    break
            tokens.append(text[start:i])
            continue

        tokens.append(ch)
        i += 1

    return tokens
