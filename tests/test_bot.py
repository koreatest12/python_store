from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bot import DataTokenBot


class DataTokenBotTests(unittest.TestCase):
    def test_prepare_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.jsonl"
            source.write_text(
                '{"text":"안녕하세요","label":"ko"}\n{"text":"hello","label":"en"}\n',
                encoding="utf-8",
            )
            bot = DataTokenBot(Path(directory) / "workspace")
            report = bot.prepare_dataset(source, source_format="jsonl")
            self.assertEqual(report.status, "ok")
            self.assertEqual(report.details["records"], 2)

    def test_train_and_register(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            corpus.write_text("hello world\nhello python\n", encoding="utf-8")
            bot = DataTokenBot(Path(directory) / "workspace")
            report = bot.train_and_register(
                corpus,
                kind="bpe",
                name="bot-test",
                version="1.0.0",
                vocab_size=32,
            )
            self.assertEqual(report.status, "ok")
            self.assertTrue(report.details["sha256"])


if __name__ == "__main__":
    unittest.main()
