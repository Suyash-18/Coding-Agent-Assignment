"""Per-run JSON-lines logs.

One file per run under the runs directory (see config.runs_dir): one JSON object per
line with a timestamp, elapsed milliseconds, the event type and its data. Everything is
passed through guardrails.redact first, and a logging failure never breaks a run.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coding_agent import config
from coding_agent.guardrails import redact


class RunLog:
    def __init__(self, run_id: str, directory: Path | None):
        self.run_id = run_id
        self._started = time.monotonic()
        self.path: Path | None = None
        if directory is not None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.path = Path(directory) / f"{stamp}_{run_id[:8]}.jsonl"

    @classmethod
    def create(cls, run_id: str) -> RunLog:
        return cls(run_id, config.runs_dir())

    def write(self, record_type: str, **fields: Any) -> None:
        if self.path is None:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "elapsed_ms": int((time.monotonic() - self._started) * 1000),
            "run_id": self.run_id,
            "type": record_type,
            **fields,
        }
        try:
            line = redact(json.dumps(record, ensure_ascii=False, default=str))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            self.path = None  # disk problem: stop logging, keep the run alive

    def event(self, ev) -> None:
        self.write(ev.type, node=ev.node, data=ev.data)