from __future__ import annotations

import re
from typing import Any

from pymongo.errors import PyMongoError

from coding_agent import mongo
from coding_agent.api.errors import ApiError
from coding_agent.api.settings import Settings

_NAME = re.compile(r"^[\w\-]{1,64}$")
MAX_LISTED = 100


def _require_db() -> None:
    if not mongo.enabled():
        raise ApiError(404, "run_logs_disabled",
                       "Run logs are disabled (AGENT_RUN_LOGS=off or no MongoDB URI set).")


def list_history(settings: Settings) -> list[dict]:
    _require_db()
    pipeline = [
        {"$group": {
            "_id": "$run_id",
            "started": {"$min": "$ts"},
            "modified": {"$max": "$ts"},
            "events": {"$sum": 1},
            "task": {"$max": {"$cond": [{"$eq": ["$type", "run_start"]}, "$task", None]}},
            "model": {"$max": {"$cond": [{"$eq": ["$type", "run_start"]}, "$model", None]}},
        }},
        {"$sort": {"started": -1}},
        {"$limit": MAX_LISTED},
    ]
    try:
        rows = list(mongo.get_collection().aggregate(pipeline))
    except PyMongoError as exc:
        raise ApiError(503, "log_store_unavailable", f"Could not reach the log database: {exc}") from exc
    return [
        {
            "name": r["_id"],  # the run id; pass it to GET /history/{name}
            "events": r["events"],
            "task": r.get("task"),
            "model": r.get("model"),
            "started": r["started"].isoformat(),
            "modified": r["modified"].isoformat(),
        }
        for r in rows
    ]


def read_history(settings: Settings, name: str, limit: int = 500) -> dict:
    _require_db()
    if not _NAME.match(name):
        raise ApiError(400, "bad_log_name", "Log names are run ids.")
    try:
        cursor = (mongo.get_collection()
                  .find({"run_id": name}, {"_id": 0})
                  .sort("seq", 1)
                  .limit(limit + 1))
        docs: list[dict[str, Any]] = list(cursor)
    except PyMongoError as exc:
        raise ApiError(503, "log_store_unavailable", f"Could not reach the log database: {exc}") from exc
    if not docs:
        raise ApiError(404, "log_not_found", f"No run log named '{name}'.")
    truncated = len(docs) > limit
    events = docs[:limit]
    for ev in events:
        ev["ts"] = ev["ts"].isoformat()
    return {"name": name, "events": events, "truncated": truncated}