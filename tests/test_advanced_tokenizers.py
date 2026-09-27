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

    def test_wordpiece_training_and_encoding(self) -> None:
        tokenizer = WordPieceTokenizer(vocab_size=64, lowercase=True)
        tokenizer.train(CORPUS)
        self.assertTrue(tokenizer.encode("hello"))

    def test_sentencepiece_character_coverage(self) -> None:
        tokenizer = SentencePieceStyleTokenizer(vocab_size=64, lowercase=True)
        tokenizer.train(CORPUS)
        tokens = tokenizer.tokenize("안녕하세요 python")
        self.assertTrue(tokens)
        self.assertNotIn("<UNK>", tokens)

    def test_unicode_input_is_preserved(self) -> None:
        tokenizer = BPETokenizer(vocab_size=128)
        tokenizer.train(["ㅋㅋㅋ café 東京 snake_case 😀"])
        tokens = tokenizer.tokenize("ㅋㅋㅋ café 東京 snake_case 😀")
        self.assertTrue(tokens)
        self.assertNotEqual(tokens, [])

    def test_factory(self) -> None:
        for kind in ("bpe", "wordpiece", "sentencepiece"):
            tokenizer = train_tokenizer(kind, CORPUS, vocab_size=64)
            self.assertGreater(len(tokenizer.token_to_id), 2)

    def test_models_can_be_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for kind in ("bpe", "wordpiece", "sentencepiece"):
                tokenizer = train_tokenizer(kind, CORPUS, vocab_size=64)
                path = Path(directory) / f"{kind}.json"
                tokenizer.save(path)
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
