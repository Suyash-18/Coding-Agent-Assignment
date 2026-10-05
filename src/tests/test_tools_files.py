import pytest

from coding_agent.errors import PathViolation, ToolError
from coding_agent.tools import list_files, read_file, safe_path


class TestSafePath:
    def test_allows_normal_path(self, tmp_path, make_repo):
        make_repo(tmp_path, {"a.py": "x"})
        assert safe_path(tmp_path, "a.py") == (tmp_path / "a.py").resolve()

    def test_blocks_parent_traversal(self, tmp_path):
        with pytest.raises(PathViolation):
            safe_path(tmp_path, "../outside.txt")

    def test_blocks_nested_traversal(self, tmp_path):
        with pytest.raises(PathViolation):
            safe_path(tmp_path, "sub/../../outside.txt")

    def test_blocks_absolute_path(self, tmp_path):
        with pytest.raises(PathViolation):
            safe_path(tmp_path, str(tmp_path.parent / "x.txt"))

    def test_blocks_null_byte(self, tmp_path):
        with pytest.raises(PathViolation):
            safe_path(tmp_path, "a\0b")

    def test_blocks_symlink_escape(self, tmp_path, make_repo):
        outside = make_repo(tmp_path / "outside", {"secret.txt": "s"})
        repo = tmp_path / "repo"
        repo.mkdir()
        try:
            (repo / "link").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported here")
        with pytest.raises(PathViolation):
            safe_path(repo, "link/secret.txt")


class TestListFiles:
    def test_sorted_posix_paths(self, tmp_path, make_repo):
        make_repo(tmp_path, {"b.py": "1", "pkg/a.py": "2"})
        assert [f.path for f in list_files(tmp_path)] == ["b.py", "pkg/a.py"]

    def test_skips_ignored_and_protected(self, tmp_path, make_repo):
        make_repo(tmp_path, {
            "keep.py": "1",
            ".env": "SECRET=1",
            ".env.local": "x",
            ".git/config": "x",
            "__pycache__/m.pyc": "x",
            "node_modules/pkg/i.js": "x",
            "logo.png": "x",
            "server.key": "x",
        })
        assert [f.path for f in list_files(tmp_path)] == ["keep.py"]

    def test_skips_oversized(self, tmp_path, make_repo):
        make_repo(tmp_path, {"small.py": "x", "big.py": "x" * 200})
        assert [f.path for f in list_files(tmp_path, max_bytes=100)] == ["small.py"]

    def test_not_a_directory(self, tmp_path):
        with pytest.raises(ToolError):
            list_files(tmp_path / "nope")


class TestReadFile:
    def test_reads_content(self, tmp_path, make_repo):
        make_repo(tmp_path, {"a.py": "print('hi')\n"})
        assert read_file(tmp_path, "a.py") == "print('hi')\n"

    def test_blocks_env_file(self, tmp_path, make_repo):
        make_repo(tmp_path, {".env": "SECRET=1"})
        with pytest.raises(PathViolation):
            read_file(tmp_path, ".env")

    def test_blocks_traversal(self, tmp_path):
        with pytest.raises(PathViolation):
            read_file(tmp_path, "../x.txt")

    def test_missing_file(self, tmp_path):
        with pytest.raises(ToolError, match="not found"):
            read_file(tmp_path, "nope.py")

    def test_too_large(self, tmp_path, make_repo):
        make_repo(tmp_path, {"big.py": "x" * 200})
        with pytest.raises(ToolError, match="too large"):
            read_file(tmp_path, "big.py", max_bytes=100)

    def test_binary_content(self, tmp_path):
        (tmp_path / "data.txt").write_bytes(b"\xff\xfe\x00")
        with pytest.raises(ToolError, match="UTF-8"):
            read_file(tmp_path, "data.txt")