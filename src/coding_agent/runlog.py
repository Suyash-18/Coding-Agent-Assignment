"""Per-run logs stored in MongoDB Atlas.

One document per event in the `run_logs` collection: run_id, seq, ts, elapsed_ms,
type and its data. Everything is passed through guardrails.redact first, and a
logging failure never breaks a run.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

from pymongo.errors import PyMongoError

from coding_agent import mongo
from coding_agent.guardrails import redact


class RunLog:
    def __init__(self, run_id: str, enabled: bool = True):
        self.run_id = run_id
        self._started = time.monotonic()
        self._seq = itertools.count(1)
        self._lock = threading.Lock()
        self.enabled = enabled
        self.written = False

    @classmethod
    def create(cls, run_id: str) -> RunLog:
        return cls(run_id, enabled=mongo.enabled())

    def write(self, record_type: str, **fields: Any) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                payload = {
                    "run_id": self.run_id,
                    "seq": next(self._seq),
                    "elapsed_ms": int((time.monotonic() - self._started) * 1000),
                    "type": record_type,
                    **fields,
                }
            raw = redact(json.dumps(payload, ensure_ascii=False, default=str))
            try:
                doc = json.loads(raw)
            except ValueError:  # redaction broke the JSON: keep the text anyway
                doc = {"run_id": self.run_id, "type": record_type, "raw": raw}
            doc["ts"] = datetime.now(timezone.utc)
            mongo.get_collection().insert_one(doc)
            self.written = True
        except (PyMongoError, KeyError):
            self.enabled = False  # DB problem: stop logging, keep the run alive

    def event(self, ev) -> None:
        self.write(ev.type, node=ev.node, data=ev.data)