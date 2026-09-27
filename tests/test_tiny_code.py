from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from tiny_code import TinyCodeGenerator


class TinyCodeGeneratorTests(unittest.TestCase):
    def test_generate_all_templates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generator = TinyCodeGenerator(directory)
            artifacts = generator.generate_all()

            self.assertEqual(len(artifacts), 6)
            for artifact in artifacts:
                path = Path(artifact.path)
                self.assertTrue(path.exists())
                ast.parse(path.read_text(encoding="utf-8"))
                self.assertTrue(artifact.valid_python)

    def test_invalid_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generator = TinyCodeGenerator(directory)
            with self.assertRaises(ValueError):
                generator.generate("tokenize", name="../escape")


if __name__ == "__main__":
    unittest.main()
