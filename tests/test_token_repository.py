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

            self.assertTrue((repository.root / artifact.file).exists())
            self.assertTrue(repository.verify())
            self.assertEqual(len(repository.list()), 1)
            self.assertFalse(Path(artifact.file).is_absolute())

            archive = repository.export_download_bundle()
            self.assertTrue(archive.exists())
            self.assertGreater(archive.stat().st_size, 0)

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            source = Path(directory) / "model.json"
            source.write_text("{}", encoding="utf-8")
            repository = TokenRepository(root)

            attacks = [
                {"name": "../escape", "version": "1.0.0"},
                {"name": "safe", "version": "../../../escaped"},
                {"name": "..", "version": "1.0.0"},
            ]
            for attack in attacks:
                with self.subTest(attack=attack):
                    with self.assertRaises(ValueError):
                        repository.add(
                            source,
                            model_type="bpe",
                            name=attack["name"],
                            version=attack["version"],
                        )

    def test_manifest_paths_are_root_relative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            source = Path(directory) / "model.json"
            source.write_text("{}", encoding="utf-8")
            repository = TokenRepository(root)
            repository.add(source, name="demo", model_type="bpe", version="1.0.0")

            original_cwd = Path.cwd()
            try:
                # verify() should not depend on the current working directory
                import os
                os.chdir(directory)
                self.assertTrue(repository.verify())
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
