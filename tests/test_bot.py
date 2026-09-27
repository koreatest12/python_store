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

    def test_cycle_runs_all_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            jsonl = base / "dataset.jsonl"
            csv = base / "dataset.csv"
            corpus = base / "corpus.txt"

            jsonl.write_text(
                '{"text":"안녕하세요","label":"ko"}\n{"text":"hello","label":"en"}\n',
                encoding="utf-8",
            )
            csv.write_text(
                'text,label\n"안녕하세요",ko\n"hello",en\n',
                encoding="utf-8",
            )
            corpus.write_text(
                "안녕하세요 토크나이저\nhello tokenizer\n",
                encoding="utf-8",
            )

            bot = DataTokenBot(base / "workspace")
            report = bot.cycle(
                jsonl_dataset=jsonl,
                csv_dataset=csv,
                corpus=corpus,
                vocab_size=64,
            )

            self.assertEqual(report.status, "ok")
            self.assertEqual(report.details["stage_count"], 10)
            self.assertEqual(report.details["failed_stage_count"], 0)
            self.assertTrue((base / "workspace" / "reports" / "latest-cycle.json").exists())

    def test_generate_tiny_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bot = DataTokenBot(Path(directory) / "workspace")
            report = bot.generate_tiny_code()
            self.assertEqual(report.status, "ok")
            self.assertEqual(report.details["generated"], 6)
            for file in report.details["files"]:
                self.assertTrue(Path(file).exists())

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
