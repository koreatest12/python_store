from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from token_repository import TokenRepository


class TokenRepositoryTests(unittest.TestCase):
    def test_add_verify_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            source = Path(directory) / "model.json"
            source.write_text(
                json.dumps({"token_to_id": {"<PAD>": 0, "<UNK>": 1}}),
                encoding="utf-8",
            )

            repository = TokenRepository(root)
            artifact = repository.add(
                source,
                name="demo-model",
                model_type="bpe",
                version="1.0.0",
            )

            self.assertTrue(Path(artifact.file).exists())
            self.assertTrue(repository.verify())
            self.assertEqual(len(repository.list()), 1)

            archive = repository.export_download_bundle()
            self.assertTrue(archive.exists())
            self.assertGreater(archive.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
