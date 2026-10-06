import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "sample_project"
DEST = ROOT / "demo" / "sample_project"  # must keep the name "sample_project" (see PLAN.md)


def main() -> None:
    if DEST.parent.exists():
        shutil.rmtree(DEST.parent)
    DEST.parent.mkdir(parents=True)
    shutil.copytree(SRC, DEST, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    print(f"Fresh copy ready at {DEST.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()