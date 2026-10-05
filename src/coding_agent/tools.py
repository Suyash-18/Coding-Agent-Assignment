"""Pure, LLM-free tools the agent uses to inspect and change a repo."""

from __future__ import annotations

import difflib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from coding_agent.errors import PathViolation, ToolError

IGNORED_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "runs", ".idea", ".vscode",
}
BLOCKED_SUFFIXES = {
    ".pem", ".key", ".pyc", ".pyo", ".db", ".sqlite", ".sqlite3",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".exe", ".dll", ".so",
}
BLOCKED_NAMES = {"id_rsa", "id_ed25519"}
MAX_FILE_BYTES = 100_000
MAX_TEST_OUTPUT_CHARS = 4_000


@dataclass(frozen=True)
class FileInfo:
    path: str  # posix-style, relative to the repo root
    size: int


def is_protected(rel_path: str | Path) -> bool:
    """True for files/folders the agent must never read or write."""
    path = Path(rel_path)
    if any(part in IGNORED_DIRS for part in path.parts):
        return True
    name = path.name
    return (
        name.startswith(".env")
        or name in BLOCKED_NAMES
        or path.suffix.lower() in BLOCKED_SUFFIXES
    )


def safe_path(repo: str | Path, rel_path: str | Path) -> Path:
    """Resolve rel_path inside repo, or raise PathViolation if it escapes."""
    root = Path(repo).resolve()
    if "\0" in str(rel_path):
        raise PathViolation(f"Invalid path: {rel_path!r}")
    try:
        candidate = Path(rel_path)
        if candidate.is_absolute() or candidate.drive:
            raise PathViolation(f"Absolute paths are not allowed: {rel_path}")
        target = (root / candidate).resolve()  # also resolves symlinks
    except (ValueError, OSError) as exc:
        raise PathViolation(f"Invalid path: {rel_path!r}") from exc
    if target != root and not target.is_relative_to(root):
        raise PathViolation(f"Path escapes the repository: {rel_path}")
    return target


def list_files(repo: str | Path, max_bytes: int = MAX_FILE_BYTES) -> list[FileInfo]:
    """List readable text-like files, skipping ignored, protected, and huge ones."""
    root = Path(repo).resolve()
    if not root.is_dir():
        raise ToolError(f"Not a directory: {repo}")

    found: list[FileInfo] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in IGNORED_DIRS and not (Path(dirpath) / d).is_symlink()
        )
        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = full.relative_to(root)
            if full.is_symlink() or is_protected(rel):
                continue
            size = full.stat().st_size
            if size > max_bytes:
                continue
            found.append(FileInfo(rel.as_posix(), size))
    return sorted(found, key=lambda f: f.path)


def read_file(
    repo: str | Path, rel_path: str | Path, max_bytes: int = MAX_FILE_BYTES
) -> str:
    """Read a UTF-8 text file from the repo."""
    root = Path(repo).resolve()
    target = safe_path(root, rel_path)
    if is_protected(target.relative_to(root)):
        raise PathViolation(f"Access to protected file is blocked: {rel_path}")
    if not target.is_file():
        raise ToolError(f"File not found: {rel_path}")
    if target.stat().st_size > max_bytes:
        raise ToolError(f"File too large to read (limit {max_bytes} bytes): {rel_path}")
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"File is not valid UTF-8 text: {rel_path}") from exc

def build_diff(old: str, new: str, path: str) -> str:
    """Unified diff between old and new content (empty string if identical)."""
    lines = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}" if old else "/dev/null",
        tofile=f"b/{path}",
    )
    out = []
    for line in lines:
        out.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(out)


def _copy_ignore(directory: str, names: list[str]) -> list[str]:
    skipped = []
    for name in names:
        full = Path(directory) / name
        if full.is_symlink() or name in IGNORED_DIRS or is_protected(name):
            skipped.append(name)
    return skipped


def copy_to_temp(repo: str | Path) -> Path:
    """Copy the repo (minus secrets, venvs, caches) to a fresh temp folder."""
    root = Path(repo).resolve()
    if not root.is_dir():
        raise ToolError(f"Not a directory: {repo}")
    tmp = Path(tempfile.mkdtemp(prefix="coding_agent_"))
    shutil.copytree(root, tmp, dirs_exist_ok=True, ignore=_copy_ignore)
    return tmp


def cleanup_temp(path: str | Path) -> None:
    """Delete a temp copy. Refuses to delete anything not made by copy_to_temp."""
    target = Path(path)
    if not target.name.startswith("coding_agent_"):
        raise ToolError(f"Refusing to delete a folder we did not create: {path}")
    shutil.rmtree(target, ignore_errors=True)


def apply_changes(repo: str | Path, changes: dict[str, str]) -> list[str]:
    """Write {relative_path: new_content}. Validates everything before writing."""
    root = Path(repo).resolve()
    planned: list[tuple[str, Path, str]] = []
    for rel, content in changes.items():
        if not isinstance(content, str):
            raise ToolError(f"Content for {rel} must be text")
        target = safe_path(root, rel)
        if target == root or target.is_dir():
            raise ToolError(f"Not a file path: {rel}")
        if is_protected(target.relative_to(root)):
            raise PathViolation(f"Writing to protected file is blocked: {rel}")
        planned.append((rel, target, content))

    written = []
    for rel, target, content in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(rel)
    return written

@dataclass(frozen=True)
class TestResult:
    __test__ = False  # stop pytest from trying to collect this class

    passed: bool
    returncode: int
    output: str
    timed_out: bool = False


_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _clean_env() -> dict[str, str]:
    """Environment for the test subprocess: no API keys or secrets."""
    env = {
        k: v for k, v in os.environ.items()
        if not any(marker in k.upper() for marker in _SECRET_MARKERS)
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _to_text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data


def _tail(text: str, limit: int = MAX_TEST_OUTPUT_CHARS) -> str:
    return text if len(text) <= limit else "...[truncated]\n" + text[-limit:]


def run_tests(root: str | Path, timeout: int = 60) -> TestResult:
    """Run pytest in root with a timeout and without access to our secrets."""
    folder = Path(root)
    if not folder.is_dir():
        raise ToolError(f"Not a directory: {root}")
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider"]
    try:
        proc = subprocess.run(
            cmd, cwd=folder, env=_clean_env(), capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        partial = _to_text(exc.stdout) + _to_text(exc.stderr)
        return TestResult(False, -1, _tail(partial + f"\nTimed out after {timeout}s"), True)
    return TestResult(proc.returncode == 0, proc.returncode, _tail(proc.stdout + proc.stderr))