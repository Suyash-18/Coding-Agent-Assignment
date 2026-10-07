from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from coding_agent.api.errors import ApiError
from coding_agent.api.settings import Settings
from coding_agent.errors import PathViolation, ToolError
from coding_agent.tools import build_diff, list_files, read_file

MAX_DIFF_CHARS = 200_000


@dataclass(frozen=True)
class RepoInfo:
    id: str
    label: str
    description: str
    path: Path
    writable: bool
    pristine: Path | None = None

    @property
    def ready(self) -> bool:
        return self.path.is_dir()


def registry(settings: Settings) -> dict[str, RepoInfo]:
    """The only repos the API will ever touch. Clients pick an id, never a path."""
    return {
        "demo": RepoInfo(
            "demo", "Demo repo (agent target)",
            "Disposable copy of the sample project. Approved changes are written here.",
            settings.demo_path, writable=True, pristine=settings.sample_path,
        ),
        "sample": RepoInfo(
            "sample", "Sample project (pristine)",
            "The original sample project. Read-only reference for comparisons.",
            settings.sample_path, writable=False,
        ),
    }


def get_repo(settings: Settings, repo_id: str) -> RepoInfo:
    info = registry(settings).get(repo_id)
    if info is None:
        raise ApiError(404, "unknown_repo", f"Unknown repo '{repo_id}'.", available=sorted(registry(settings)))
    return info


def require_ready(info: RepoInfo) -> None:
    if not info.ready:
        hint = " Call POST /repos/demo/reset to create it." if info.id == "demo" else ""
        raise ApiError(409, "repo_not_ready", f"Repo '{info.id}' does not exist yet.{hint}")


def list_repo_files(info: RepoInfo) -> list[dict]:
    require_ready(info)
    return [{"path": f.path, "size": f.size} for f in list_files(info.path)]


def read_repo_file(info: RepoInfo, rel_path: str) -> dict:
    require_ready(info)
    try:
        text = read_file(info.path, rel_path)
    except PathViolation as exc:
        raise ApiError(403, "path_blocked", str(exc)) from exc
    except ToolError as exc:
        msg = str(exc)
        if "not found" in msg.lower():
            raise ApiError(404, "file_not_found", msg) from exc
        if "too large" in msg.lower():
            raise ApiError(413, "file_too_large", msg) from exc
        raise ApiError(415, "not_text", msg) from exc
    return {"repo_id": info.id, "path": rel_path, "size": len(text.encode("utf-8")), "content": text}


def _read_or_empty(root: Path, rel: str) -> str:
    try:
        return read_file(root, rel)
    except ToolError:
        return ""


def diff_against_pristine(info: RepoInfo) -> dict:
    """What has changed in a repo compared with the pristine sample it was copied from."""
    if info.pristine is None:
        raise ApiError(400, "no_baseline", f"Repo '{info.id}' has no pristine baseline to compare with.")
    require_ready(info)
    before = {f.path for f in list_files(info.pristine)}
    after = {f.path for f in list_files(info.path)}
    files, parts = [], []
    for rel in sorted(before | after):
        diff = build_diff(_read_or_empty(info.pristine, rel), _read_or_empty(info.path, rel), rel)
        if not diff:
            continue
        status = "added" if rel not in before else "deleted" if rel not in after else "modified"
        files.append({"path": rel, "status": status})
        parts.append(diff)
    text = "".join(parts)
    return {
        "repo_id": info.id,
        "changed": bool(files),
        "files": files,
        "diff": text[:MAX_DIFF_CHARS],
        "truncated": len(text) > MAX_DIFF_CHARS,
    }


def reset_demo(settings: Settings) -> dict:
    """Replace demo/sample_project with a fresh copy of the pristine sample."""
    dest, sample = settings.demo_path, settings.sample_path
    if not sample.is_dir():
        raise ApiError(500, "sample_missing", "The pristine sample project was not found on the server.")
    container = dest.parent
    if container.resolve() != (settings.root / "demo").resolve():  # never delete anything else
        raise ApiError(500, "unsafe_reset", "Refusing to reset an unexpected folder.")
    if container.exists():
        shutil.rmtree(container)
    container.mkdir(parents=True)
    shutil.copytree(sample, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".env*"))
    count = len(list_files(dest))
    return {"repo_id": "demo", "path": "demo/sample_project", "files": count}
