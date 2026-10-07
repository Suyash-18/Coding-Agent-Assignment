"""HTTP client for the coding-agent backend.

Everything the UI assumes about the wire format lives in this file:

* ACTIONS / STAGES: the strings POST /runs/{id}/decision expects.
* parse_sse(): how Server-Sent Events are turned into event dicts
  {"type", "node", "data", "seq"}. It accepts both "event: <type>" frames and frames whose
  JSON already carries a "type" field.

If the backend disagrees with an assumption, this is the only module to change.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from typing import Any

import httpx

DEFAULT_API_URL = os.getenv("CODING_AGENT_API_URL", "http://localhost:8000")
ACTIONS = {"approve": "approve", "reject": "reject"}
STAGES = ("plan", "apply")

# Tests set this to an httpx transport so the real network is never touched.
DEFAULT_TRANSPORT: httpx.BaseTransport | None = None


class UiApiError(Exception):
    """The backend answered with an error. Carries what the UI needs to explain it."""

    def __init__(self, status: int, code: str, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after

    def __str__(self) -> str:
        wait = f" (retry in {self.retry_after:.0f}s)" if self.retry_after else ""
        return f"{self.message}{wait}"


class ApiUnavailable(Exception):
    """The backend could not be reached at all."""


def _error_from(response: httpx.Response) -> UiApiError:
    code, message = "http_error", f"The server answered HTTP {response.status_code}."
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, dict):
        code = str(detail.get("code", code))
        message = str(detail.get("message", message))
    elif isinstance(detail, list):  # FastAPI request-validation errors
        code = "validation_error"
        message = "; ".join(
            f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: {item.get('msg', '')}".strip(": ")
            for item in detail if isinstance(item, dict)
        ) or message
    elif isinstance(detail, str):
        message = detail
    retry = response.headers.get("retry-after")
    try:
        retry_after = float(retry) if retry else None
    except ValueError:
        retry_after = None
    return UiApiError(response.status_code, code, message, retry_after)


# ---- Server-Sent Events -----------------------------------------------------
def _build_event(name: str | None, raw: str, event_id: str | None) -> dict[str, Any]:
    try:
        payload: Any = json.loads(raw)
    except ValueError:
        payload = {"message": raw}
    if isinstance(payload, dict) and "type" in payload:
        event = {"type": str(payload["type"]), "node": payload.get("node") or "",
                 "data": payload.get("data") or {}}
    elif isinstance(payload, dict):
        event = {"type": name or "message", "node": payload.get("node") or "",
                 "data": payload.get("data", payload)}
    else:
        event = {"type": name or "message", "node": "", "data": {"value": payload}}
    seq = event_id if event_id is not None else (payload.get("seq") if isinstance(payload, dict) else None)
    event["seq"] = int(seq) if str(seq).isdigit() else None
    return event


def parse_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Turn SSE text lines into event dicts. Comments (keep-alives) are ignored."""
    name: str | None = None
    event_id: str | None = None
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                yield _build_event(name, "\n".join(data), event_id)
            name, event_id, data = None, None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            name = value
        elif field == "id":
            event_id = value
        elif field == "data":
            data.append(value)
    if data:  # stream ended without a trailing blank line
        yield _build_event(name, "\n".join(data), event_id)


# ---- client -----------------------------------------------------------------
class ApiClient:
    def __init__(self, base_url: str = DEFAULT_API_URL, *, timeout: float = 20.0,
                 transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout,
                                  transport=transport or DEFAULT_TRANSPORT)

    # -- plumbing
    def _call(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiUnavailable(f"The backend at {self.base_url} did not answer in time.") from exc
        except httpx.TransportError as exc:
            raise ApiUnavailable(f"Could not reach the backend at {self.base_url}.") from exc
        if response.status_code >= 400:
            raise _error_from(response)
        return response

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._call(method, path, **kwargs).json()

    # -- system
    def health(self) -> dict:
        return self._json("GET", "/health")

    def models(self) -> dict:
        return self._json("GET", "/models")

    # -- repos
    def repos(self) -> list[dict]:
        return self._json("GET", "/repos")

    def repo_files(self, repo_id: str) -> dict:
        return self._json("GET", f"/repos/{repo_id}/files")

    def repo_file(self, repo_id: str, path: str) -> dict:
        return self._json("GET", f"/repos/{repo_id}/files/{path}")

    def repo_diff(self, repo_id: str) -> dict:
        return self._json("GET", f"/repos/{repo_id}/diff")

    def reset_repo(self, repo_id: str) -> dict:
        return self._json("POST", f"/repos/{repo_id}/reset")

    # -- runs
    def create_run(self, task: str, repo_id: str, model: str | None = None) -> dict:
        body: dict[str, Any] = {"task": task, "repo_id": repo_id}
        if model:
            body["model"] = model
        return self._json("POST", "/runs", json=body)

    def runs(self) -> list[dict]:
        return self._json("GET", "/runs")

    def run(self, run_id: str) -> dict:
        return self._json("GET", f"/runs/{run_id}")

    def result(self, run_id: str) -> dict:
        return self._json("GET", f"/runs/{run_id}/result")

    def patch(self, run_id: str) -> bytes:
        return self._call("GET", f"/runs/{run_id}/patch").content

    def decide(self, run_id: str, stage: str, action: str, feedback: str | None = None) -> dict:
        body: dict[str, Any] = {"stage": stage, "action": ACTIONS[action]}
        if feedback:
            body["feedback"] = feedback
        return self._json("POST", f"/runs/{run_id}/decision", json=body)

    def cancel(self, run_id: str) -> dict:
        return self._json("POST", f"/runs/{run_id}/cancel")

    def stream_events(self, run_id: str, after: int = 0, follow: bool = True) -> Iterator[dict]:
        """Yield parsed events. A read timeout ends the stream quietly; the caller re-checks."""
        timeout = httpx.Timeout(10.0, read=120.0)
        try:
            with self._http.stream(
                "GET", f"/runs/{run_id}/events",
                params={"after": after, "follow": str(follow).lower()}, timeout=timeout,
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise _error_from(response)
                yield from parse_sse(response.iter_lines())
        except httpx.ReadTimeout:
            return
        except httpx.TransportError as exc:
            raise ApiUnavailable(f"Lost the connection to {self.base_url}.") from exc

    # -- scripts
    def scripts(self) -> list[dict]:
        return self._json("GET", "/scripts")

    def jobs(self) -> list[dict]:
        return self._json("GET", "/scripts/jobs")

    def job(self, job_id: str) -> dict:
        return self._json("GET", f"/scripts/jobs/{job_id}")

    def run_script(self, name: str, body: dict | None = None, wait: float = 0) -> dict:
        return self._json("POST", f"/scripts/{name}", json=body or {}, params={"wait": wait})

    # -- history
    def history(self) -> list[dict]:
        return self._json("GET", "/history")

    def history_item(self, name: str, limit: int = 500) -> dict:
        return self._json("GET", f"/history/{name}", params={"limit": limit})


def make_client(base_url: str | None = None) -> ApiClient:
    return ApiClient(base_url or DEFAULT_API_URL)
