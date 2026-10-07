"""MongoDB Atlas connection for run logs."""
from __future__ import annotations

import os
import threading

from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv


load_dotenv()



DB_NAME = "coding_agent"        # <- set your database name here
COLLECTION = "run_logs"
URI_ENV = "MONGODB_URI"         # <- env var that holds your Atlas URL

_client: MongoClient | None = None
_indexed = False
_lock = threading.Lock()


def enabled() -> bool:
    """Logs are on when the URI is set and AGENT_RUN_LOGS is not 'off'."""
    flag = os.getenv("AGENT_RUN_LOGS", "on").strip().lower()
    return flag not in ("off", "0", "false") and bool(os.getenv(URI_ENV))


def get_collection() -> Collection:
    global _client, _indexed
    with _lock:
        if _client is None:
            _client = MongoClient(
                os.environ[URI_ENV],
                serverSelectionTimeoutMS=3000,   # fail fast: logging must not stall a run
                connectTimeoutMS=3000,
                tz_aware=True,
            )
        coll = _client[DB_NAME][COLLECTION]
        if not _indexed:
            coll.create_index([("run_id", 1), ("seq", 1)])
            coll.create_index([("ts", -1)])
            _indexed = True
        return coll