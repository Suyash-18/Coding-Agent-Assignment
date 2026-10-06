import shutil
from pathlib import Path

SRC = Path("src/sample_project")
DEST = Path("demo_repo")


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(SRC, DEST, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    print(f"Fresh copy ready at {DEST}/")


if __name__ == "__main__":
    main()