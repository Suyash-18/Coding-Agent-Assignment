from pathlib import Path

import pytest


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