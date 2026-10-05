from pathlib import Path

import pytest

from coding_agent.errors import PathViolation, ToolError
from coding_agent.tools import apply_changes, build_diff, cleanup_temp, copy_to_temp


class TestBuildDiff:
    def test_shows_changes(self):
        diff = build_diff("a\nb\n", "a\nc\n", "f.py")
        assert "--- a/f.py" in diff
        assert "+++ b/f.py" in diff
        assert "-b" in diff and "+c" in diff

    def test_identical_is_empty(self):
        assert build_diff("same\n", "same\n", "f.py") == ""

    def test_new_file(self):
        diff = build_diff("", "x\n", "new.py")
        assert "--- /dev/null" in diff
        assert "+x" in diff

    def test_missing_trailing_newline(self):
        assert "No newline at end of file" in build_diff("a", "b", "f.py")


class TestCopyToTemp:
    def test_excludes_secrets_and_junk(self, tmp_path, make_repo):
        repo = make_repo(tmp_path / "repo", {
            "app.py": "1", ".env": "S", ".venv/lib.py": "x",
            "__pycache__/a.pyc": "x", ".git/HEAD": "x",
        })
        copy = copy_to_temp(repo)
        try:
            names = {p.relative_to(copy).as_posix() for p in copy.rglob("*") if p.is_file()}
            assert names == {"app.py"}
        finally:
            cleanup_temp(copy)

    def test_original_untouched_by_changes(self, tmp_path, make_repo):
        repo = make_repo(tmp_path / "repo", {"app.py": "original"})
        copy = copy_to_temp(repo)
        try:
            (copy / "app.py").write_text("changed")
            assert (repo / "app.py").read_text() == "original"
        finally:
            cleanup_temp(copy)

    def test_cleanup_refuses_foreign_folder(self, tmp_path):
        with pytest.raises(ToolError):
            cleanup_temp(tmp_path)
        assert tmp_path.exists()


class TestApplyChanges:
    def test_writes_and_creates_dirs(self, tmp_path, make_repo):
        repo = make_repo(tmp_path / "repo", {"x.py": "0"})
        written = apply_changes(repo, {"x.py": "1", "pkg/new.py": "2"})
        assert written == ["x.py", "pkg/new.py"]
        assert (repo / "x.py").read_text() == "1"
        assert (repo / "pkg" / "new.py").read_text() == "2"

    def test_bad_path_writes_nothing(self, tmp_path, make_repo):
        repo = make_repo(tmp_path / "repo", {"x.py": "0"})
        with pytest.raises(PathViolation):
            apply_changes(repo, {"ok.py": "1", "../evil.py": "2"})
        assert not (repo / "ok.py").exists()
        assert not (tmp_path / "evil.py").exists()

    def test_blocks_protected_file(self, tmp_path, make_repo):
        repo = make_repo(tmp_path / "repo", {"x.py": "0"})
        with pytest.raises(PathViolation):
            apply_changes(repo, {".env": "KEY=1"})

    def test_rejects_non_text(self, tmp_path, make_repo):
        repo = make_repo(tmp_path / "repo", {"x.py": "0"})
        with pytest.raises(ToolError):
            apply_changes(repo, {"x.py": 123})  # type: ignore[dict-item]