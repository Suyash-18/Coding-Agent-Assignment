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