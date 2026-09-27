from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset import DatasetRecord, TextDataset, build_tokenized_dataset
from pretokenizer import Pretokenizer
from tokenizer import SimpleTokenizer


class PretokenizerDatasetTests(unittest.TestCase):
    def test_pretokenizer_split_and_offsets(self) -> None:
        pre = Pretokenizer(lowercase=True)
        tokens = pre.split("안녕하세요 Python 3.14!")
        self.assertEqual(tokens, ["안녕하세요", "python", "3.14", "!"])

        spans = pre.split_with_offsets("Hello world!")
        self.assertEqual([span.text for span in spans], ["hello", "world", "!"])
        self.assertEqual(spans[0].start, 0)

    def test_unicode_characters_are_not_silently_dropped(self) -> None:
        text = "ㅋㅋㅋ 좋아요 café naïve 東京 snake_case 😀"
        tokens = Pretokenizer().split(text)
        joined = "".join(tokens)
        for expected in ["ㅋㅋㅋ", "좋아요", "café", "naïve", "東京", "snake_case", "😀"]:
            self.assertIn(expected, tokens)
        self.assertNotIn(" ", joined)

    def test_dataset_split_batches_and_jsonl(self) -> None:
        records = [DatasetRecord(text=f"sample {i}", label=str(i % 2)) for i in range(10)]
        dataset = TextDataset(records)
        split = dataset.split(seed=1)
        self.assertEqual(len(split.train) + len(split.validation) + len(split.test), 10)

        batches = list(dataset.batches(4))
        self.assertEqual([len(batch) for batch in batches], [4, 4, 2])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            dataset.to_jsonl(path)
            restored = TextDataset.from_jsonl(path)
            self.assertEqual(len(restored), len(dataset))

    def test_csv_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            path.write_text("text,label\nhello,en\n안녕,ko\n", encoding="utf-8")
            dataset = TextDataset.from_csv(path)
            self.assertEqual(dataset[1].label, "ko")

    def test_build_tokenized_dataset(self) -> None:
        dataset = TextDataset([DatasetRecord(text="안녕하세요 Python")])
        tokenizer = SimpleTokenizer(lowercase=True)
        tokenizer.fit(dataset.texts())

        rows = build_tokenized_dataset(dataset, tokenizer, max_length=6)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["input_ids"]), 6)


if __name__ == "__main__":
    unittest.main()
