from __future__ import annotations

import platform
import sys


def main() -> None:
    print("python_store")
    print(f"Python: {platform.python_version()}")
    print(f"Executable: {sys.executable}")


if __name__ == "__main__":
    main()
