from __future__ import annotations

import time
from collections.abc import Callable
from importlib import metadata
from typing import Any

from fastapi import FastAPI, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from coding_agent.api import history as history_api
from coding_agent.api import repos as repos_api
from coding_agent.api.errors import ApiError
from coding_agent.api.jobs import JobManager
from coding_agent.api.runs import Run, RunManager
from coding_agent.api.safety import precheck_text
from coding_agent.api.schemas import DecisionRequest, RunCreate, ScriptRequest
from coding_agent.api.settings import Settings
from coding_agent import mongo

DESCRIPTION = """
HTTP API for the coding agent. Typical flow:

1. `POST /runs` with a task, then `GET /runs/{id}/events` (Server-Sent Events).
2. When an `interrupt` event arrives, answer it with `POST /runs/{id}/decision`
   (stage `plan`, then stage `apply`).
3. Read `GET /runs/{id}/result` or `GET /runs/{id}/patch`.
"""


def _version() -> str:
    for name in ("coding-agent", "coding_agent"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return "dev"


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def run_summary(run: Run, detail: bool = False) -> dict[str, Any]:
    out = {
        "run_id": run.id, "state": run.state, "repo_id": run.repo_id, "model": run.model, "task": run.task,
        "pending_stage": run.pending_stage, "events": len(run.events),
        "created_at": run.created_at, "updated_at": run.updated_at,
    }
    if detail:
        out["pending"] = run.pending
    return out


def run_result(run: Run) -> dict[str, Any]:
    done = run.done or {}
    last_test = run.tests[-1] if run.tests else {}
    return {
        "run_id": run.id, "state": run.state, "ready": run.terminal,
        "status": done.get("status"), "applied": bool(done.get("applied", False)),
        "task": run.task, "repo_id": run.repo_id, "model": run.model,
        "plan": run.plan, "selected_files": run.selected_files, "changed_files": run.files,
        "diff": run.diff, "explanation": run.explanation,
        "test_passed": done.get("test_passed", last_test.get("passed")),
        "attempts": done.get("attempts", len(run.tests)),
        "tests": run.tests, "error": run.error,
    }


def create_app(
    settings: Settings | None = None,
    *,
    llm_factory: Callable[[str], Any] | None = None,
    script_runner: Callable | None = None,
    clock: Callable[[], float] = time.time,
) -> FastAPI:
    settings = settings or Settings.from_env()
    manager = RunManager(settings, llm_factory=llm_factory, clock=clock)
    jobs = JobManager(settings, manager, runner=script_runner, clock=clock)
    started = clock()

    app = FastAPI(title="Coding Agent API", version=_version(), description=DESCRIPTION)
    app.state.settings, app.state.manager, app.state.jobs = settings, manager, jobs
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        detail = {"code": exc.code, "message": exc.message, **exc.extra}
        return JSONResponse({"detail": detail}, status_code=exc.status, headers=exc.headers)

    @app.exception_handler(404)
    async def _not_found(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            {
                "detail": {
                    "code": "not_found",
                    "message": "The requested resource was not found.",
                }
            },
            status_code=404,
        )

    # ---- system ------------------------------------------------------------
    @app.get("/health", tags=["system"])
    def health() -> dict[str, Any]:
        import os
        return {
            "status": "ok", "version": _version(), "uptime_s": round(clock() - started, 1),
            "default_model": settings.default_model,
            "llm_key_configured": bool(os.getenv("GROQ_API_KEY")),
            "repos": {rid: info.ready for rid, info in repos_api.registry(settings).items()},
            "active_runs": manager.active_count(),
            "run_logs": mongo.enabled(),
        }
    @app.get("/health-corn", tags=["system"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
        }

    @app.get("/models", tags=["system"])
    def models() -> dict[str, Any]:
        return {"default": settings.default_model, "allowed": list(settings.allowed_models)}

    # ---- repos --------------------------------------------------------------
    @app.get("/repos", tags=["repos"])
    def list_repos() -> list[dict]:
        return [
            {"id": r.id, "label": r.label, "description": r.description, "writable": r.writable,
             "ready": r.ready, "path": r.path.relative_to(settings.root).as_posix()}
            for r in repos_api.registry(settings).values()
        ]

    @app.get("/repos/{repo_id}/files", tags=["repos"])
    def repo_files(repo_id: str) -> dict:
        info = repos_api.get_repo(settings, repo_id)
        return {"repo_id": repo_id, "files": repos_api.list_repo_files(info)}

    @app.get("/repos/{repo_id}/files/{path:path}", tags=["repos"])
    def repo_file(repo_id: str, path: str) -> dict:
        return repos_api.read_repo_file(repos_api.get_repo(settings, repo_id), path)

    @app.get("/repos/{repo_id}/diff", tags=["repos"])
    def repo_diff(repo_id: str) -> dict:
        """Everything that differs between the demo repo and the pristine sample."""
        return repos_api.diff_against_pristine(repos_api.get_repo(settings, repo_id))

    @app.post("/repos/{repo_id}/reset", tags=["repos"])
    def repo_reset(repo_id: str) -> dict:
        info = repos_api.get_repo(settings, repo_id)
        if info.pristine is None:
            raise ApiError(403, "not_resettable", f"Repo '{repo_id}' cannot be reset.")
        owner = "reset"
        manager.claim_repo(info.id, owner)
        try:
            return repos_api.reset_demo(settings)
        finally:
            manager.release_repo(info.id, owner)

    # ---- runs ---------------------------------------------------------------
    @app.post("/runs", status_code=201, tags=["runs"])
    def create_run(body: RunCreate, request: Request, response: Response) -> dict:
        precheck_text(body.task)
        run = manager.create(body.task.strip(), body.repo_id, body.model, _client_ip(request))
        response.headers["Location"] = f"/runs/{run.id}"
        return {**run_summary(run), "events_url": f"/runs/{run.id}/events", "result_url": f"/runs/{run.id}/result"}

    @app.get("/runs", tags=["runs"])
    def list_runs() -> list[dict]:
        return [run_summary(r) for r in manager.list()]

    @app.get("/runs/{run_id}", tags=["runs"])
    def get_run(run_id: str) -> dict:
        return run_summary(manager.get(run_id), detail=True)

    @app.get("/runs/{run_id}/events", tags=["runs"])
    def run_events(
        run_id: str,
        after: int = Query(0, ge=0, description="Only events with seq greater than this."),
        follow: bool = Query(True, description="Keep the stream open until the run finishes."),
        last_event_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        """Server-Sent Events. Stays open across approval pauses; ends when the run finishes."""
        run = manager.get(run_id)
        if last_event_id and last_event_id.isdigit():
            after = max(after, int(last_event_id))
        return StreamingResponse(
            manager.stream(run, after=after, follow=follow),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/runs/{run_id}/decision", tags=["runs"])
    def decide(run_id: str, body: DecisionRequest) -> dict:
        if body.feedback:
            precheck_text(body.feedback, field="feedback")
        return run_summary(manager.decide(run_id, body.stage, body.action, body.feedback))

    @app.get("/runs/{run_id}/result", tags=["runs"])
    def result(run_id: str) -> dict:
        return run_result(manager.get(run_id))

    @app.get("/runs/{run_id}/patch", tags=["runs"])
    def patch(run_id: str) -> Response:
        run = manager.get(run_id)
        if not run.diff:
            raise ApiError(404, "no_diff_yet", "This run has not produced a diff yet.")
        return Response(run.diff, media_type="text/x-diff",
                        headers={"Content-Disposition": f'attachment; filename="{run.id[:8]}.patch"'})

    @app.post("/runs/{run_id}/cancel", tags=["runs"])
    def cancel(run_id: str) -> dict:
        return run_summary(manager.cancel(run_id))

    # ---- scripts -------------------------------------------------------------
    @app.get("/scripts", tags=["scripts"])
    def list_scripts() -> list[dict]:
        return jobs.specs()

    @app.get("/scripts/jobs", tags=["scripts"])
    def list_jobs() -> list[dict]:
        return jobs.list()

    @app.get("/scripts/jobs/{job_id}", tags=["scripts"])
    def get_job(job_id: str) -> dict:
        return jobs.get(job_id).view()

    @app.post("/scripts/{name}", status_code=202, tags=["scripts"])
    def run_script(name: str, request: Request, body: ScriptRequest | None = None,
                   wait: float = Query(0, ge=0, le=120, description="Seconds to wait for the result.")) -> dict:
        job = jobs.start(name, body or ScriptRequest(), _client_ip(request))
        if wait:
            job.done.wait(timeout=wait)
        return job.view()

    # ---- run logs --------------------------------------------------------------
    @app.get("/history", tags=["history"])
    def list_logs() -> list[dict]:
        return history_api.list_history(settings)

    @app.get("/history/{name}", tags=["history"])
    def read_log(name: str, limit: int = Query(500, ge=1, le=5000)) -> dict:
        return history_api.read_history(settings, name, limit)

    return app
