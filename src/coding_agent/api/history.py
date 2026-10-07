from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from coding_agent.api.errors import ApiError
from coding_agent.api.settings import Settings

_NAME = re.compile(r"^[\w.\-]+\.jsonl$")
MAX_LISTED = 100


def _require_dir(settings: Settings):
    if settings.runs_dir is None:
        raise ApiError(404, "run_logs_disabled", "Run logs are disabled (AGENT_RUNS_DIR=off).")
    return settings.runs_dir


def list_history(settings: Settings) -> list[dict]:
    folder = _require_dir(settings)
    if not folder.is_dir():
        return []
    files = sorted((p for p in folder.glob("*.jsonl") if p.is_file()), key=lambda p: p.name, reverse=True)
    return [
        {
            "name": p.name,
            "size": p.stat().st_size,
            "modified": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
        }
        for p in files[:MAX_LISTED]
    ]


def read_history(settings: Settings, name: str, limit: int = 500) -> dict:
    folder = _require_dir(settings)
    if not _NAME.match(name):
        raise ApiError(400, "bad_log_name", "Log names look like <time>_<id>.jsonl.")
    path = (folder / name).resolve()
    if path.parent != folder.resolve() or not path.is_file():
        raise ApiError(404, "log_not_found", f"No run log named '{name}'.")
    events, truncated = [], False
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if len(events) >= limit:
                truncated = True
                break
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {"name": name, "events": events, "truncated": truncated}
