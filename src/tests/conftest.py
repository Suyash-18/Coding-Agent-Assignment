import shutil
from pathlib import Path

import pytest

SAMPLE = Path(__file__).resolve().parent.parent / "sample_project"


@pytest.fixture
def make_repo():
    """Returns a function that creates files under a root folder."""

    def _make(root: Path, files: dict[str, str]) -> Path:
        for rel, content in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return root

    return _make


@pytest.fixture
def sample_repo(tmp_path) -> Path:
    """A throwaway copy of sample_project that tests may freely modify."""
    dest = tmp_path / "sample_repo"
    shutil.copytree(
        SAMPLE, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache")
    )
    return dest