from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from advanced_tokenizers import (
    BPETokenizer,
    SentencePieceStyleTokenizer,
    WordPieceTokenizer,
    train_tokenizer,
)


CORPUS = [
    "안녕하세요 파이썬 토크나이저",
    "파이썬 토크나이저 학습",
    "hello tokenizer world",
    "hello python world",
]


class AdvancedTokenizerTests(unittest.TestCase):
    def test_bpe_training_and_encoding(self) -> None:
        tokenizer = BPETokenizer(vocab_size=64, lowercase=True)
        tokenizer.train(CORPUS)
        self.assertTrue(tokenizer.merges)
        self.assertGreater(len(tokenizer.token_to_id), 2)
        self.assertTrue(tokenizer.tokenize("hello python"))
        self.assertTrue(tokenizer.encode("hello python"))

    def test_wordpiece_training_and_encoding(self) -> None:
        tokenizer = WordPieceTokenizer(vocab_size=64, lowercase=True)
        tokenizer.train(CORPUS)
        self.assertGreater(len(tokenizer.token_to_id), 2)
        tokens = tokenizer.tokenize("hello")
        self.assertTrue(tokens)
        self.assertTrue(tokenizer.encode("hello"))

    def test_sentencepiece_style_training(self) -> None:
        tokenizer = SentencePieceStyleTokenizer(vocab_size=64, lowercase=True)
        tokenizer.train(CORPUS)
        self.assertGreater(len(tokenizer.token_to_id), 2)
        tokens = tokenizer.tokenize("hello world")
        self.assertTrue(tokens)
        self.assertTrue(any(token.startswith("▁") for token in tokens if token != "<UNK>"))

    def test_factory(self) -> None:
        for kind in ("bpe", "wordpiece", "sentencepiece"):
            tokenizer = train_tokenizer(kind, CORPUS, vocab_size=48)
            self.assertGreater(len(tokenizer.token_to_id), 2)

    def test_models_can_be_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for kind in ("bpe", "wordpiece", "sentencepiece"):
                tokenizer = train_tokenizer(kind, CORPUS, vocab_size=48)
                path = Path(directory) / f"{kind}.json"
                tokenizer.save(path)
                self.assertTrue(path.exists())
                self.assertIn("token_to_id", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
