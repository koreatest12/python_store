from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tokenizer import SimpleTokenizer


class SimpleTokenizerTests(unittest.TestCase):
    def test_tokenize_korean_english_numbers_and_punctuation(self) -> None:
        tokenizer = SimpleTokenizer()
        self.assertEqual(
            tokenizer.tokenize("안녕하세요 Python 3.14!"),
            ["안녕하세요", "Python", "3.14", "!"],
        )

    def test_fit_and_encode(self) -> None:
        tokenizer = SimpleTokenizer(lowercase=True)
        tokenizer.fit(["Hello world", "hello Python"])

        encoded = tokenizer.encode("HELLO unknown")
        self.assertEqual(encoded[0], tokenizer.token_to_id["hello"])
        self.assertEqual(encoded[1], tokenizer.unk_id)

    def test_padding_and_truncation(self) -> None:
        tokenizer = SimpleTokenizer()
        tokenizer.fit(["가 나 다"])

        padded = tokenizer.encode("가", max_length=3)
        self.assertEqual(len(padded), 3)
        self.assertEqual(padded[-1], tokenizer.pad_id)

        truncated = tokenizer.encode("가 나 다", max_length=2)
        self.assertEqual(len(truncated), 2)

    def test_save_and_load(self) -> None:
        tokenizer = SimpleTokenizer(lowercase=True)
        tokenizer.fit(["Token 저장 테스트"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = SimpleTokenizer.load(path)

        self.assertEqual(restored.lowercase, tokenizer.lowercase)
        self.assertEqual(restored.token_to_id, tokenizer.token_to_id)


if __name__ == "__main__":
    unittest.main()
