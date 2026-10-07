"""Allow-listed internal scripts, run as background jobs. Clients can never pass raw commands."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from coding_agent.api.errors import ApiError
from coding_agent.api.repos import reset_demo
from coding_agent.api.runs import RunManager
from coding_agent.api.safety import precheck_text, redact_text
from coding_agent.api.schemas import ScriptRequest
from coding_agent.api.settings import Settings

MAX_OUTPUT_CHARS = 20_000
MAX_JOBS_KEPT = 50

Runner = Callable[[list[str], Path, dict, int], tuple[int, str, str, bool]]


@dataclass(frozen=True)
class ScriptSpec:
    name: str
    description: str
    module: str | None = None  # python -m <module>; None means "run in-process"
    claims_demo: bool = False
    uses_llm: bool = False
    params: tuple[str, ...] = ()


SCRIPTS: dict[str, ScriptSpec] = {s.name: s for s in (
    ScriptSpec("smoke", "Call the model once to check the API key and model name.",
               module="scripts.smoke", uses_llm=True),
    ScriptSpec("reset_demo", "Replace the demo repo with a fresh copy of the sample project.",
               claims_demo=True),
    ScriptSpec("check_demo", "Run the demo repo's tests in the same isolated runner the agent uses.",
               module="scripts.check_demo", claims_demo=True),
    ScriptSpec("dev_run", "Run the agent end to end on the demo repo (auto-answers the approvals).",
               module="scripts.dev_run", claims_demo=True, uses_llm=True,
               params=("task", "model", "apply", "reject")),
)}


def _text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data


def subprocess_runner(argv: list[str], cwd: Path, env: dict, timeout: int) -> tuple[int, str, str, bool]:
    try:
        proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as exc:
        return -1, _text(exc.stdout), _text(exc.stderr) + f"\nTimed out after {timeout}s", True


def _clip(text: str) -> str:
    text = redact_text(text)
    return text if len(text) <= MAX_OUTPUT_CHARS else "...[truncated]\n" + text[-MAX_OUTPUT_CHARS:]


class Job:
    def __init__(self, job_id: str, name: str, args: list[str], now: float):
        self.id, self.name, self.args = job_id, name, args
        self.state = "running"  # running | succeeded | failed | timeout
        self.exit_code: int | None = None
        self.stdout = self.stderr = ""
        self.started_at, self.finished_at = now, None
        self.done = threading.Event()

    def view(self) -> dict:
        end = self.finished_at or time.time()
        return {
            "job_id": self.id, "script": self.name, "state": self.state, "exit_code": self.exit_code,
            "args": self.args, "stdout": self.stdout, "stderr": self.stderr,
            "duration_s": round(end - self.started_at, 2),
        }


class JobManager:
    def __init__(self, settings: Settings, manager: RunManager, runner: Runner | None = None,
                 clock: Callable[[], float] = time.time):
        self.settings, self.manager, self.runner, self.clock = settings, manager, runner or subprocess_runner, clock
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def available(self, spec: ScriptSpec) -> bool:
        if spec.module is None:
            return True
        return (self.settings.root / "src" / "scripts" / f"{spec.name}.py").is_file()

    def specs(self) -> list[dict]:
        return [
            {"name": s.name, "description": s.description, "available": self.available(s),
             "uses_llm": s.uses_llm, "mutates_demo": s.claims_demo, "params": list(s.params)}
            for s in SCRIPTS.values()
        ]

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)
        return [j.view() for j in jobs]

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise ApiError(404, "unknown_job", f"No script job with id '{job_id}'.")
        return job

    def _argv(self, spec: ScriptSpec, req: ScriptRequest) -> list[str]:
        if spec.module is None:
            return []
        argv = [sys.executable, "-m", spec.module]
        if spec.name == "dev_run":
            if not req.task:
                raise ApiError(422, "task_required", "dev_run needs a 'task'.")
            if req.apply and req.reject:
                raise ApiError(422, "conflicting_flags", "Choose either 'apply' or 'reject', not both.")
            precheck_text(req.task)
            model = req.model or self.settings.default_model
            if model not in self.settings.allowed_models:
                raise ApiError(422, "model_not_allowed", f"Model '{model}' is not allowed.", allowed=list(self.settings.allowed_models))
            argv += ["--repo", str(self.settings.demo_path), "--task", req.task, "--model", model]
            argv += ["--apply"] if req.apply else []
            argv += ["--reject"] if req.reject else []
        return argv

    def start(self, name: str, req: ScriptRequest, client: str) -> Job:
        spec = SCRIPTS.get(name)
        if spec is None:
            raise ApiError(404, "unknown_script", f"Unknown script '{name}'.", available=sorted(SCRIPTS))
        if not self.available(spec):
            raise ApiError(404, "script_unavailable", f"scripts/{name}.py is not present on this server.")
        argv = self._argv(spec, req)

        with self._lock:
            if any(not j.done.is_set() for j in self._jobs.values()):
                raise ApiError(409, "job_running", "Another script job is still running.")
            job = Job(uuid.uuid4().hex, name, argv[3:] if argv else [], self.clock())
            self._jobs[job.id] = job
            self._evict()
        owner = f"job:{job.id}"
        try:
            if spec.claims_demo:
                self.manager.claim_repo("demo", owner)
            if spec.uses_llm:
                retry = self.manager.limiter.check(client)
                if retry is not None:
                    raise ApiError(429, "rate_limited", f"Too many runs. Try again in {retry}s.", headers={"Retry-After": str(retry)}, retry_after=retry)
                self.manager.limiter.record(client)
        except ApiError:
            with self._lock:
                self._jobs.pop(job.id, None)
            self.manager.release_repo("demo", owner)
            raise

        threading.Thread(target=self._work, args=(job, spec, argv, owner), daemon=True, name=f"job-{job.id[:8]}").start()
        return job

    def _work(self, job: Job, spec: ScriptSpec, argv: list[str], owner: str) -> None:
        code, out, err, timed_out = 1, "", "", False
        try:
            if spec.module is None:
                info = reset_demo(self.settings)
                code, out = 0, f"Fresh copy ready at {info['path']}/ ({info['files']} files)\n"
            else:
                env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
                code, out, err, timed_out = self.runner(argv, self.settings.root, env, self.settings.script_timeout_s)
        except ApiError as exc:
            err = exc.message
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        finally:
            job.stdout, job.stderr, job.exit_code = _clip(out), _clip(err), code
            job.state = "timeout" if timed_out else "succeeded" if code == 0 else "failed"
            job.finished_at = self.clock()
            self.manager.release_repo("demo", owner)
            job.done.set()

    def _evict(self) -> None:
        extra = len(self._jobs) - MAX_JOBS_KEPT
        for job in sorted((j for j in self._jobs.values() if j.done.is_set()), key=lambda j: j.started_at)[:max(extra, 0)]:
            del self._jobs[job.id]
