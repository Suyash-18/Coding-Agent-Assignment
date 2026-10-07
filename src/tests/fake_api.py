"""A stateful fake of the backend, built from the routes in the pasted API file.

Used through httpx.MockTransport, so UI tests never touch a network. It models the same
approval flow as the real agent: events up to a plan interrupt, then up to an apply
interrupt, then the final events.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

RUN_ID = "abcd1234ef567890"
PLAN = {
    "summary": "Add field constraints and a test.",
    "steps": [{"file": "models.py", "action": "Add Field constraints"},
              {"file": "tests/test_users.py", "action": "Test invalid age"}],
    "assumptions": ["Age must be between 0 and 130"],
}
DIFF = "--- a/models.py\n+++ b/models.py\n@@ -1 +1 @@\n-age: int\n+age: int = Field(ge=0)\n"
FILES = ["models.py", "tests/test_users.py"]


class FakeApi:
    def __init__(self, tests_pass: bool = True, block: bool = False):
        self.tests_pass, self.block = tests_pass, block
        self.created: list[dict] = []
        self.decisions: list[dict] = []
        self.served = 0          # highest event seq already streamed
        self.phase = 0           # 0: before plan decision, 1: before apply decision, 2: after
        self.pending: str | None = None
        self.finished: str | None = None   # None | applied | rejected | declined | cancelled
        self.events = self._script()

    # ---- scripted agent events -------------------------------------------------
    def _script(self) -> list[tuple[str, str, dict]]:
        ok = self.tests_pass
        return [
            ("start", "", {"thread_id": RUN_ID}),
            ("node_done", "scan_repo", {"file_count": 7}),
            ("node_done", "select_files", {"relevant_files": FILES, "reason": "models + tests"}),
            ("node_done", "read_files", {"files_read": FILES}),
            ("plan", "make_plan", {"plan": PLAN}),
            ("interrupt", "", {"stage": "plan", "plan": PLAN, "files": FILES}),
            ("node_done", "plan_approval", {"approved": True}),
            ("node_done", "generate_changes", {"files": FILES}),
            ("test", "run_tests", {"passed": ok, "attempt": 1, "will_retry": False,
                                   "no_tests": False, "output": "1 failed" if not ok else "8 passed"}),
            ("diff", "build_diff", {"diff": DIFF, "files": FILES}),
            ("node_done", "explain", {"explanation": "Added constraints."}),
            ("interrupt", "", {"stage": "apply", "diff": DIFF, "files": FILES,
                               "test_passed": ok, "no_tests": False, "attempts": 1}),
            ("node_done", "apply_approval", {"approved": True}),
            ("node_done", "apply_changes", {"files": FILES}),
            ("done", "", {"status": "applied", "applied": True, "test_passed": ok, "attempts": 1}),
        ]

    LIMITS = (6, 12, 15)

    # ---- request handling ----------------------------------------------------------
    def __call__(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        body = json.loads(request.content) if request.content else None

        if path == "/health":
            return self.ok({"status": "ok", "version": "0.1.0", "uptime_s": 5.0,
                            "default_model": "openai/gpt-oss-120b", "llm_key_configured": True,
                            "repos": {"demo": True}, "active_runs": 0, "run_logs": True})
        if path == "/models":
            return self.ok({"default": "openai/gpt-oss-120b",
                            "allowed": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]})
        if path == "/repos":
            return self.ok([{"id": "demo", "label": "Demo sample project", "writable": True,
                             "description": "A small FastAPI user manager.", "ready": True,
                             "path": "demo/sample_project"}])
        if path == "/repos/demo/files":
            return self.ok({"repo_id": "demo", "files": [{"path": "models.py"}, {"path": "utils.py"}]})
        if path.startswith("/repos/demo/files/"):
            return self.ok({"path": path.removeprefix("/repos/demo/files/"), "content": "x = 1\n"})
        if path == "/repos/demo/diff":
            return self.ok({"diff": ""})
        if path == "/repos/demo/reset" and method == "POST":
            return self.ok({"status": "reset", "files": 7})
        if path == "/history":
            return self.ok([{"name": "20261001T000000Z_abcd1234.jsonl", "size": 120}])
        if path.startswith("/history/"):
            return self.ok({"name": path.rsplit("/", 1)[-1], "records": [{"type": "start"}]})
        if path == "/scripts" and method == "GET":
            return self.ok([{"name": "smoke", "description": "Call the model once"}])
        if path == "/scripts/jobs":
            return self.ok([{"id": "job1", "name": "smoke", "status": "done"}])
        if path.startswith("/scripts/jobs/"):
            return self.ok({"id": "job1", "status": "done", "output": "ok"})
        if path == "/scripts/smoke" and method == "POST":
            return self.ok({"id": "job1", "name": "smoke", "status": "done"}, 202)

        if path == "/runs" and method == "POST":
            if self.block:
                return self.error(400, "input_blocked", "Blocked by input guard (destructive request).")
            self.created.append(body)
            return self.ok({**self.summary(), "events_url": f"/runs/{RUN_ID}/events",
                            "result_url": f"/runs/{RUN_ID}/result"}, 201)
        if path == f"/runs/{RUN_ID}":
            return self.ok(self.summary(detail=True))
        if path == f"/runs/{RUN_ID}/result":
            return self.ok(self.result())
        if path == f"/runs/{RUN_ID}/events":
            return self.stream(int(request.url.params.get("after", 0)))
        if path == f"/runs/{RUN_ID}/decision" and method == "POST":
            return self.decide(body)
        if path == f"/runs/{RUN_ID}/cancel" and method == "POST":
            self.finished, self.pending = "cancelled", None
            return self.ok(self.summary())
        if path == f"/runs/{RUN_ID}/patch":
            return httpx.Response(200, content=DIFF.encode(), headers={"content-type": "text/x-diff"})
        return self.error(404, "not_found", "The requested resource was not found.")

    # ---- pieces -----------------------------------------------------------------------
    @staticmethod
    def ok(payload: Any, status: int = 200) -> httpx.Response:
        return httpx.Response(status, json=payload)

    @staticmethod
    def error(status: int, code: str, message: str) -> httpx.Response:
        return httpx.Response(status, json={"detail": {"code": code, "message": message}})

    def stream(self, after: int) -> httpx.Response:
        limit = 15 if self.finished and self.finished != "cancelled" else self.LIMITS[self.phase]
        lines = [": keep-alive", ""]
        for seq, (kind, node, data) in enumerate(self.events, start=1):
            if after < seq <= limit:
                lines += [f"id: {seq}", f"event: {kind}",
                          "data: " + json.dumps({"type": kind, "node": node, "data": data}), ""]
        self.served = max(self.served, limit if limit > after else self.served)
        if limit == 6:
            self.pending = "plan"
        elif limit == 12:
            self.pending = "apply"
        elif limit == 15:
            self.pending, self.finished = None, self.finished or "applied"
        return httpx.Response(200, content="\n".join(lines).encode(),
                              headers={"content-type": "text/event-stream"})

    def decide(self, body: dict) -> httpx.Response:
        self.decisions.append(body)
        stage, action = body["stage"], body["action"]
        if stage != self.pending:
            return self.error(409, "wrong_stage", f"The run is not waiting for a {stage} decision.")
        self.pending = None
        if action == "reject":
            self.finished = "rejected" if stage == "plan" else "declined"
        else:
            self.phase = 1 if stage == "plan" else 2
        return self.ok(self.summary())

    def summary(self, detail: bool = False) -> dict:
        out = {"run_id": RUN_ID, "state": self.state(), "repo_id": "demo",
               "model": "openai/gpt-oss-120b", "task": "t", "pending_stage": self.pending,
               "events": self.served, "created_at": 1.0, "updated_at": 2.0}
        if detail:
            out["pending"] = ({"stage": "plan", "plan": PLAN, "files": FILES} if self.pending == "plan"
                              else {"stage": "apply", "diff": DIFF} if self.pending else None)
        return out

    def state(self) -> str:
        return self.finished or ("awaiting_decision" if self.pending else "running")

    def result(self) -> dict:
        s, ok = self.served, self.tests_pass
        status = None if self.finished in (None, "cancelled") else self.finished
        return {
            "run_id": RUN_ID, "state": self.state(), "ready": self.finished is not None,
            "status": status, "applied": self.finished == "applied", "task": "t",
            "repo_id": "demo", "model": "openai/gpt-oss-120b",
            "plan": PLAN if s >= 5 else None, "selected_files": FILES if s >= 3 else [],
            "changed_files": FILES if s >= 10 else [], "diff": DIFF if s >= 10 else "",
            "explanation": "Added constraints." if s >= 11 else "",
            "test_passed": (ok if s >= 9 else None), "attempts": 1 if s >= 9 else 0,
            "tests": ([{"passed": ok, "attempt": 1, "will_retry": False, "no_tests": False,
                        "output": "" if ok else "FAILED test_x"}] if s >= 9 else []),
            "error": None,
        }
