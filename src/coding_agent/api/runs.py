from __future__ import annotations

import json
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

from coding_agent.api.errors import ApiError
from coding_agent.api.repos import registry
from coding_agent.api.safety import redact_obj
from coding_agent.api.settings import Settings
from coding_agent.runner import resume_agent, stream_agent

TERMINAL = {"completed", "failed"}


class RateLimiter:
    """Sliding-window limits: per client per hour, and a global per-day cap."""

    def __init__(self, per_hour: int, per_day: int, clock: Callable[[], float] = time.time):
        self.per_hour, self.per_day, self.clock = per_hour, per_day, clock
        self._by_client: dict[str, deque[float]] = {}
        self._all: deque[float] = deque()
        self._lock = threading.Lock()

    @staticmethod
    def _prune(window: deque[float], now: float, span: float) -> None:
        while window and now - window[0] >= span:
            window.popleft()

    def check(self, client: str) -> int | None:
        """Seconds until the client may try again, or None if it may go ahead now."""
        with self._lock:
            now = self.clock()
            mine = self._by_client.setdefault(client, deque())
            self._prune(mine, now, 3600)
            self._prune(self._all, now, 86400)
            if self.per_hour > 0 and len(mine) >= self.per_hour:
                return max(1, math.ceil(mine[0] + 3600 - now))
            if self.per_day > 0 and len(self._all) >= self.per_day:
                return max(1, math.ceil(self._all[0] + 86400 - now))
            return None

    def record(self, client: str) -> None:
        with self._lock:
            now = self.clock()
            self._by_client.setdefault(client, deque()).append(now)
            self._all.append(now)


class Run:
    def __init__(self, run_id: str, task: str, repo_id: str, model: str, client: str, now: float):
        self.id, self.task, self.repo_id, self.model, self.client = run_id, task, repo_id, model, client
        self.created_at = self.updated_at = now
        self.state = "running"  # running | awaiting_plan | awaiting_apply | completed | failed
        self.events: list[dict[str, Any]] = []
        self.cond = threading.Condition()
        self.cancel_requested = False
        self.pending: dict | None = None
        self.plan: dict | None = None
        self.selected_files: list[str] = []
        self.diff = ""
        self.files: list[str] = []
        self.tests: list[dict] = []
        self.explanation = ""
        self.done: dict | None = None
        self.error: dict | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    @property
    def pending_stage(self) -> str | None:
        return self.state.removeprefix("awaiting_") if self.state.startswith("awaiting_") else None

    def append(self, etype: str, node: str, data: dict, now: float) -> dict:
        data = redact_obj(data)
        with self.cond:
            env = {"seq": len(self.events) + 1, "type": etype, "node": node or "", "data": data}
            self.events.append(env)
            self.updated_at = now
            self._derive(etype, node, data)
            self.cond.notify_all()
            return env

    def _derive(self, etype: str, node: str, data: dict) -> None:
        if etype == "plan":
            self.plan = data.get("plan")
        elif etype == "diff":
            self.diff = data.get("diff", "")
            self.files = list(data.get("files", []))
        elif etype == "test":
            entry = {k: data.get(k) for k in ("attempt", "passed", "will_retry", "no_tests")}
            entry["output"] = (data.get("output") or "")[-4000:]
            self.tests.append(entry)
        elif etype == "node_done" and node in ("select_files", "expand_files"):
            self.selected_files = list(data.get("relevant_files", []))
        elif etype == "interrupt":
            stage = data.get("stage")
            self.pending = data
            self.state = "awaiting_apply" if stage == "apply" else "awaiting_plan"
        elif etype == "done":
            self.done, self.pending = data, None
            self.explanation = data.get("explanation", "") or self.explanation
            self.state = "completed"
        elif etype == "error":
            self.error, self.pending = data, None
            self.state = "failed"


def sse_format(env: dict) -> str:
    return f"id: {env['seq']}\nevent: {env['type']}\ndata: {json.dumps(env, default=str)}\n\n"


class RunManager:
    def __init__(self, settings: Settings, llm_factory: Callable[[str], Any] | None = None,
                 clock: Callable[[], float] = time.time):
        self.settings, self.llm_factory, self.clock = settings, llm_factory, clock
        self.limiter = RateLimiter(settings.max_runs_per_hour, settings.max_runs_per_day, clock)
        self._runs: dict[str, Run] = {}
        self._claims: dict[str, str] = {}  # repo_id -> owner (run id or "job:<id>")
        self._lock = threading.RLock()

    # ---- repo claims (one writer at a time) --------------------------------
    def claim_repo(self, repo_id: str, owner: str) -> None:
        with self._lock:
            self._expire_stale()
            current = self._claims.get(repo_id)
            if current and current != owner:
                raise ApiError(409, "repo_busy", f"Repo '{repo_id}' is busy ({self._describe(current)}).", active=current)
            self._claims[repo_id] = owner

    def release_repo(self, repo_id: str, owner: str) -> None:
        with self._lock:
            if self._claims.get(repo_id) == owner:
                del self._claims[repo_id]

    @staticmethod
    def _describe(owner: str) -> str:
        return f"script {owner[4:]}" if owner.startswith("job:") else f"run {owner}"

    def active_count(self) -> int:
        with self._lock:
            self._expire_stale()
            return sum(1 for r in self._runs.values() if not r.terminal)

    # ---- lifecycle --------------------------------------------------------
    def create(self, task: str, repo_id: str, model: str | None, client: str) -> Run:
        info = registry(self.settings).get(repo_id)
        if info is None:
            raise ApiError(404, "unknown_repo", f"Unknown repo '{repo_id}'.", available=sorted(registry(self.settings)))
        if not info.writable:
            raise ApiError(403, "repo_read_only", f"Repo '{repo_id}' is read-only. Use repo_id 'demo'.")
        model = model or self.settings.default_model
        if model not in self.settings.allowed_models:
            raise ApiError(422, "model_not_allowed", f"Model '{model}' is not allowed.", allowed=list(self.settings.allowed_models))
        if not info.ready:
            raise ApiError(409, "repo_not_ready", "The demo repo does not exist yet. Call POST /repos/demo/reset first.")
        retry = self.limiter.check(client)
        if retry is not None:
            raise ApiError(429, "rate_limited", f"Too many runs. Try again in {retry}s.", headers={"Retry-After": str(retry)}, retry_after=retry)

        run_id = uuid.uuid4().hex
        self.claim_repo(info.id, run_id)
        try:
            llm = self.llm_factory(model) if self.llm_factory else None
        except Exception as exc:
            self.release_repo(info.id, run_id)
            raise ApiError(503, "llm_unavailable", f"Could not start the model: {exc}") from exc
        self.limiter.record(client)

        run = Run(run_id, task, info.id, model, client, self.clock())
        with self._lock:
            self._runs[run_id] = run
            self._evict()
        self._spawn(run, lambda: stream_agent(task, str(info.path), model=model, thread_id=run_id, llm=llm))
        return run

    def get(self, run_id: str) -> Run:
        with self._lock:
            self._expire_stale()
            run = self._runs.get(run_id)
        if run is None:
            raise ApiError(404, "unknown_run", f"No run with id '{run_id}'.")
        return run

    def list(self) -> list[Run]:
        with self._lock:
            self._expire_stale()
            return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def decide(self, run_id: str, stage: str, action: str, feedback: str | None) -> Run:
        run = self.get(run_id)
        payload: dict[str, Any] = {"approved": action == "approve"}
        if stage == "plan" and action == "approve" and feedback and feedback.strip():
            payload["feedback"] = feedback.strip()
        with run.cond:
            if run.state != f"awaiting_{stage}":
                raise ApiError(409, "wrong_state", f"Run is '{run.state}', not waiting for a {stage} decision.", state=run.state)
            run.state = "running"
            run.pending = None
            run.updated_at = self.clock()
        self._spawn(run, lambda: resume_agent(run.id, payload))
        return run

    def cancel(self, run_id: str) -> Run:
        run = self.get(run_id)
        with run.cond:
            if run.terminal:
                raise ApiError(409, "already_finished", f"Run already finished ({run.state}).", state=run.state)
            waiting = run.pending_stage is not None
            if not waiting:
                run.cancel_requested = True  # best effort: takes effect at the next event boundary
        if waiting:
            self._finish(run, "cancelled")
        return run

    # ---- streaming --------------------------------------------------------
    def stream(self, run: Run, after: int = 0, follow: bool = True) -> Iterator[str]:
        cursor = max(0, after)
        while True:
            with run.cond:
                if follow:
                    while cursor >= len(run.events) and not run.terminal:
                        if not run.cond.wait(timeout=self.settings.heartbeat_s):
                            break
                batch = run.events[cursor:]
                terminal = run.terminal
            if not batch:
                if not follow or terminal:
                    return
                yield ": keep-alive\n\n"
                continue
            for env in batch:
                yield sse_format(env)
            cursor += len(batch)
            if not follow:
                return

    # ---- internals --------------------------------------------------------
    def _spawn(self, run: Run, events_factory: Callable[[], Iterator]) -> None:
        def work() -> None:
            try:
                for ev in events_factory():
                    if run.cancel_requested:
                        break
                    run.append(ev.type, ev.node, dict(ev.data), self.clock())
            except Exception as exc:  # last line of defence for the worker thread
                run.append("error", "", {"message": f"Unexpected error: {type(exc).__name__}: {exc}", "kind": type(exc).__name__}, self.clock())
            finally:
                self._after_stream(run)

        threading.Thread(target=work, daemon=True, name=f"run-{run.id[:8]}").start()

    def _after_stream(self, run: Run) -> None:
        if run.cancel_requested and not run.terminal:
            self._finish(run, "cancelled")
            return
        if run.state == "running":
            run.append("error", "", {"message": "The run ended unexpectedly.", "kind": "RunInterrupted"}, self.clock())
        if run.terminal:
            self.release_repo(run.repo_id, run.id)

    def _finish(self, run: Run, status: str) -> None:
        last = run.tests[-1] if run.tests else {}
        run.append("done", "", {
            "status": status, "applied": False, "test_passed": bool(last.get("passed")),
            "attempts": len(run.tests), "plan": run.plan, "diff": run.diff,
            "changed_files": run.files, "explanation": "",
        }, self.clock())
        self.release_repo(run.repo_id, run.id)

    def _expire_stale(self) -> None:
        now = self.clock()
        for run in list(self._runs.values()):
            if run.pending_stage and now - run.updated_at > self.settings.decision_ttl_s:
                self._finish(run, "expired")

    def _evict(self) -> None:
        extra = len(self._runs) - self.settings.max_runs_kept
        if extra <= 0:
            return
        for run in sorted((r for r in self._runs.values() if r.terminal), key=lambda r: r.created_at)[:extra]:
            del self._runs[run.id]
