from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

SUPPORTED_MODEL_TYPES = {"bpe", "wordpiece", "sentencepiece", "simple"}


@dataclass(slots=True)
class TokenArtifact:
    name: str
    model_type: str
    version: str
    file: str
    sha256: str
    size_bytes: int
    created_at: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenRepository:
    def __init__(self, root: str | Path = "token_repository") -> None:
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.downloads_dir = self.root / "downloads"
        self.manifest_path = self.root / "manifest.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"schema_version": 1, "artifacts": []}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _save_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(
        self,
        source: str | Path,
        *,
        name: str,
        model_type: str,
        version: str = "1.0.0",
    ) -> TokenArtifact:
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"unsupported model type: {model_type}")

        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        safe_name = name.replace(" ", "-")
        target_dir = self.models_dir / model_type / safe_name / version
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source_path.name
        shutil.copy2(source_path, target)

        artifact = TokenArtifact(
            name=name,
            model_type=model_type,
            version=version,
            file=target.as_posix(),
            sha256=sha256_file(target),
            size_bytes=target.stat().st_size,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        manifest = self._load_manifest()
        manifest["artifacts"] = [
            item
            for item in manifest["artifacts"]
            if not (
                item["name"] == name
                and item["model_type"] == model_type
                and item["version"] == version
            )
        ]
        manifest["artifacts"].append(asdict(artifact))
        manifest["artifacts"].sort(
            key=lambda item: (item["model_type"], item["name"], item["version"])
        )
        self._save_manifest(manifest)
        return artifact

    def list(self) -> list[dict]:
        return self._load_manifest()["artifacts"]

    def verify(self) -> bool:
        valid = True
        for item in self.list():
            path = Path(item["file"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                valid = False
        return valid

    def export_download_bundle(self, output: str | Path | None = None) -> Path:
        output_path = Path(output) if output else self.downloads_dir / "token-models"
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        manifest = self._load_manifest()
        for item in manifest["artifacts"]:
            source = Path(item["file"])
            destination = (
                output_path
                / item["model_type"]
                / item["name"].replace(" ", "-")
                / item["version"]
                / source.name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        (output_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        checksums = []
        for file in sorted(output_path.rglob("*")):
            if file.is_file() and file.name != "SHA256SUMS.txt":
                checksums.append(
                    f"{sha256_file(file)}  {file.relative_to(output_path).as_posix()}"
                )
        (output_path / "SHA256SUMS.txt").write_text(
            "\n".join(checksums) + "\n",
            encoding="utf-8",
        )

        archive_base = self.downloads_dir / "python-token-repository"
        archive_path = Path(
            shutil.make_archive(str(archive_base), "zip", root_dir=output_path)
        )
        return archive_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage downloadable tokenizer models")
    parser.add_argument("--root", default="token_repository")
    sub = parser.add_subparsers(dest="command", required=True)

    add_parser = sub.add_parser("add")
    add_parser.add_argument("source")
    add_parser.add_argument("--name", required=True)
    add_parser.add_argument(
        "--type",
        dest="model_type",
        required=True,
        choices=sorted(SUPPORTED_MODEL_TYPES),
    )
    add_parser.add_argument("--version", default="1.0.0")

    sub.add_parser("list")
    sub.add_parser("verify")

    export_parser = sub.add_parser("export")
    export_parser.add_argument("--output")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = TokenRepository(args.root)

    if args.command == "add":
        artifact = repository.add(
            args.source,
            name=args.name,
            model_type=args.model_type,
            version=args.version,
        )
        print(json.dumps(asdict(artifact), ensure_ascii=False, indent=2))
    elif args.command == "list":
        print(json.dumps(repository.list(), ensure_ascii=False, indent=2))
    elif args.command == "verify":
        if not repository.verify():
            raise SystemExit("Token repository verification failed")
        print("Token repository verification succeeded")
    elif args.command == "export":
        archive = repository.export_download_bundle(args.output)
        print(f"Created download archive: {archive}")


if __name__ == "__main__":
    main()
