"""Streamlit front end for the coding agent.

Run it with:  uv run streamlit run src/coding_agent/ui/app.py

Pages: Run agent, Repositories, History, Scripts. Everything goes through the backend's
HTTP API (see client.py); there is no second agent engine here.

Live-run flow (Streamlit re-executes this script on every interaction, so state lives in
st.session_state): stream events into a timeline until an `interrupt` or terminal event,
then show the approval controls. A click POSTs the decision and reruns the script, which
resumes the stream from the last event it saw.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import streamlit as st

from coding_agent.ui.client import (
    DEFAULT_API_URL,
    ApiClient,
    ApiUnavailable,
    UiApiError,
    make_client,
)
from coding_agent.ui.timeline import STOP_TYPES, timeline_markdown

st.set_page_config(page_title="Coding Agent", page_icon="🛠️", layout="wide")

PAGES = ["Run agent", "Repositories", "History", "Scripts", "About"]
EXAMPLES = {
    "Add validation": "Add input validation to the create-user API and write a test",
    "Add /health": "Add a /health route in route.py with a test",
    "Rename function": "Rename normalize_email to normalize_email_address everywhere",
    "Add docstrings": "Add docstrings to all route functions",
}
MAX_STREAM_ATTEMPTS = 600  # bound on reconnects while a run is working
API_ERRORS = (UiApiError, ApiUnavailable)
# The two models offered in the sidebar (same names as MODEL_NAME / MODEL_NAME2 in .env).
MODEL_OPTIONS = list(dict.fromkeys(
    m for m in (
        os.getenv("MODEL_NAME", "openai/gpt-oss-120b").strip("'\" "),
        os.getenv("MODEL_NAME2", "qwen/qwen3.8-27b").strip("'\" "),
    ) if m
))


# ---- small helpers -----------------------------------------------------------
def show_problem(exc: Exception) -> None:
    st.error(str(exc))
    code = getattr(exc, "code", None)
    if code:
        st.caption(f"error code: `{code}`")


def error_text(err: Any) -> str:
    if isinstance(err, dict):
        text = str(err.get("message", "The run failed."))
        return f"{text}\n\n{err['hint']}" if err.get("hint") else text
    return str(err)


def item_name(item: Any, *keys: str) -> str:
    """Name of a list entry whose exact shape we do not control (str or dict)."""
    if isinstance(item, dict):
        for key in (*keys, "name", "id", "path"):
            if item.get(key):
                return str(item[key])
        return json.dumps(item, sort_keys=True)[:60]
    return str(item)


def reset_run_state() -> None:
    for key in ("run_id", "events", "last_seq", "run_meta"):
        st.session_state.pop(key, None)


# ---- sidebar -------------------------------------------------------------------
def sidebar() -> tuple[ApiClient, dict | None, str]:
    st.sidebar.title("🛠️ Coding Agent")
    client = make_client(DEFAULT_API_URL)  # set with CODING_AGENT_API_URL in the environment/.env
    st.sidebar.caption(f"API: `{DEFAULT_API_URL}`")
    health: dict | None = None
    try:
        health = client.health()
    except API_ERRORS as exc:
        st.sidebar.error(str(exc))
    if health:
        st.sidebar.success(f"Connected · v{health.get('version', '?')}")
        if not health.get("llm_key_configured", True):
            st.sidebar.warning("The backend has no GROQ_API_KEY: runs will fail.")
        st.sidebar.caption(
            f"default model: `{health.get('default_model', '?')}` · "
            f"active runs: {health.get('active_runs', 0)}"
        )
    default_model = (health or {}).get("default_model")
    st.sidebar.selectbox(
        "Model", MODEL_OPTIONS, key="sidebar_model",
        index=MODEL_OPTIONS.index(default_model) if default_model in MODEL_OPTIONS else 0,
    )
    page = st.sidebar.radio("Page", PAGES, key="page")
    return client, health, page


# ---- run page -----------------------------------------------------------------
def start_form(client: ApiClient, health: dict) -> None:
    st.header("Run the agent")
    try:
        repos = client.repos()
        models = client.models()
    except API_ERRORS as exc:
        show_problem(exc)
        return

    ready = [r for r in repos if r.get("ready")]
    for repo in repos:
        if not repo.get("ready"):
            st.caption(f"Repository `{repo.get('id')}` is not ready and was left out.")

    st.markdown("**Examples**")
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        col.button(label, key=f"example_{label}", use_container_width=True,
                   on_click=lambda t=text: st.session_state.update(task_text=t))

    task = st.text_area("What should the agent change?", key="task_text", height=110,
                        placeholder="Describe a code change inside the repository…")
    left, right = st.columns(2)
    with left:
        by_id = {r["id"]: r for r in ready}
        repo_id = st.selectbox("Repository", list(by_id), key="repo_choice",
                               format_func=lambda i: by_id[i].get("label", i)) if by_id else None
        if repo_id:
            info = by_id[repo_id]
            note = "" if info.get("writable", True) else " · read-only"
            st.caption(f"{info.get('description', '')}{note}")
    with right:
        model = st.session_state.get("sidebar_model") or models.get("default")
        st.markdown("**Model**")
        st.caption(f"`{model}` · change it in the sidebar")

    if not ready:
        st.warning("No repository is ready. Check the Repositories page.")
    if st.button("Start run", type="primary", key="start_run",
                 disabled=not (task or "").strip() or not repo_id):
        try:
            created = client.create_run(task.strip(), repo_id, model)
        except API_ERRORS as exc:
            show_problem(exc)
            return
        st.session_state.update(
            run_id=created["run_id"], events=[], last_seq=0,
            run_meta={"task": task.strip(), "repo_id": repo_id, "model": model},
        )
        st.rerun()


def snapshot(client: ApiClient, run_id: str) -> tuple[dict, dict]:
    return client.run(run_id), client.result(run_id)


def advance(client: ApiClient, run_id: str, live) -> tuple[dict, dict] | None:
    """Stream events until a decision is needed or the run ends. None on a hard error."""
    state = st.session_state
    for _ in range(MAX_STREAM_ATTEMPTS):
        try:
            for event in client.stream_events(run_id, after=state.last_seq):
                state.events.append(event)
                seq = event.get("seq")
                state.last_seq = seq if seq is not None else len(state.events)
                live.markdown(timeline_markdown(state.events))
                if event["type"] in STOP_TYPES:
                    break
            summary, result = snapshot(client, run_id)
        except API_ERRORS as exc:
            show_problem(exc)
            return None
        if summary.get("pending_stage") or result.get("ready"):
            return summary, result
        time.sleep(0.5)  # stream closed early: look again instead of spinning
    st.warning("The run is still working. Use the page again in a moment.")
    return None


def render_result(result: dict, events: list[dict], run_id: str) -> None:
    err = result.get("error")
    if err:
        st.error(error_text(err))
    elif result.get("ready"):
        status = result.get("status")
        if result.get("applied"):
            st.success(f"Changes applied to `{result.get('repo_id')}`.")
        elif status == "rejected":
            st.info("Plan rejected. No files were changed.")
        elif status == "declined":
            st.info("Changes were not applied. No files were changed.")

    if result.get("selected_files"):
        st.caption("Files the agent looked at: " + ", ".join(f"`{f}`" for f in result["selected_files"]))

    plan = result.get("plan")
    if plan:
        with st.expander("Plan", expanded=not result.get("diff")):
            st.markdown(plan.get("summary", ""))
            if plan.get("steps"):
                st.table([{"File": s.get("file"), "Action": s.get("action")} for s in plan["steps"]])
            for note in plan.get("assumptions", []):
                st.caption(f"Assumption: {note}")

    tests = result.get("tests") or []
    if tests:
        with st.expander("Test results", expanded=not result.get("diff")):
            for t in tests:
                verdict = ":green[passed]" if t.get("passed") else (
                    ":orange[no tests collected]" if t.get("no_tests") else ":red[failed]")
                retry = " · retried" if t.get("will_retry") else ""
                st.markdown(f"**Attempt {t.get('attempt')}**: {verdict}{retry}")
                if not t.get("passed") and t.get("output"):
                    st.code(str(t["output"])[-3000:], language="text")

    diff = result.get("diff") or ""
    if diff:
        with st.expander("Proposed diff", expanded=True):
            st.code(diff, language="diff")
            st.download_button("Download patch", data=diff, file_name=f"{run_id[:8]}.patch",
                               mime="text/x-diff", key="download_patch")

    if result.get("explanation"):
        with st.expander("Explanation", expanded=True):
            st.markdown(result["explanation"])

    with st.expander("Timeline", expanded=False):
        st.markdown(timeline_markdown(events))


def decide(client: ApiClient, run_id: str, stage: str, action: str, feedback: str | None = None) -> None:
    try:
        client.decide(run_id, stage, action, feedback)
    except API_ERRORS as exc:
        show_problem(exc)
        return
    st.rerun()


def approval_panel(client: ApiClient, run_id: str, stage: str, result: dict) -> None:
    st.divider()
    if stage == "plan":
        st.subheader("Approve this plan?")
        st.caption("Nothing is written to the repository at this step.")
        feedback = st.text_area("Optional note for the model", key=f"feedback_{run_id}")
        a, b = st.columns(2)
        if a.button("Approve plan", type="primary", key="approve_plan", use_container_width=True):
            decide(client, run_id, "plan", "approve", feedback.strip() or None)
        if b.button("Reject plan", key="reject_plan", use_container_width=True):
            decide(client, run_id, "plan", "reject")
        return

    st.subheader("Apply these changes?")
    verified = bool(result.get("test_passed"))
    if verified:
        st.success(f"Tests passed after {result.get('attempts', '?')} attempt(s).")
        acknowledged = True
    else:
        st.warning("Tests did **not** pass: this change is UNVERIFIED.")
        acknowledged = st.checkbox("I understand and still want to apply it", key="ack_unverified")
    try:
        repo = next((r for r in client.repos() if r.get("id") == result.get("repo_id")), None)
    except API_ERRORS:
        repo = None
    if repo and not repo.get("writable", True):
        st.info("This repository is read-only, so applying may be refused. You can still download the patch.")
    a, b = st.columns(2)
    if a.button("Apply changes", type="primary", key="apply_changes",
                disabled=not acknowledged, use_container_width=True):
        decide(client, run_id, "apply", "approve")
    if b.button("Decline", key="decline_changes", use_container_width=True):
        decide(client, run_id, "apply", "reject")


def run_view(client: ApiClient) -> None:
    state = st.session_state
    run_id = state.run_id
    try:
        summary, result = snapshot(client, run_id)
    except API_ERRORS as exc:
        show_problem(exc)
        if st.button("Forget this run", key="forget_run"):
            reset_run_state()
            st.rerun()
        return

    meta = state.get("run_meta", {})
    st.header("Agent run")
    st.markdown(f"> {meta.get('task') or summary.get('task', '')}")
    st.caption(f"run `{run_id[:8]}` · repo `{summary.get('repo_id')}` · model `{summary.get('model')}` "
               f"· state **{summary.get('state')}**")

    if not result.get("ready"):
        if st.button("Cancel run", key="cancel_run"):
            try:
                client.cancel(run_id)
            except API_ERRORS as exc:
                show_problem(exc)
            else:
                st.rerun()

    live = st.empty()
    if not result.get("ready") and not summary.get("pending_stage"):
        fresh = advance(client, run_id, live)
        if fresh is None:
            return
        summary, result = fresh
    live.empty()

    render_result(result, state.get("events", []), run_id)

    stage = summary.get("pending_stage")
    if stage and not result.get("ready"):
        approval_panel(client, run_id, stage, result)
    elif result.get("ready"):
        st.divider()
        if st.button("Start another run", type="primary", key="new_run"):
            reset_run_state()
            st.rerun()


# ---- repositories page ----------------------------------------------------------
def repos_page(client: ApiClient) -> None:
    st.header("Repositories")
    try:
        repos = client.repos()
    except API_ERRORS as exc:
        show_problem(exc)
        return
    st.dataframe([{k: r.get(k) for k in ("id", "label", "ready", "writable", "path", "description")}
                  for r in repos], use_container_width=True, hide_index=True)
    if not repos:
        return
    repo_id = st.selectbox("Repository", [r["id"] for r in repos], key="repos_choice")
    files_tab, diff_tab, reset_tab = st.tabs(["Files", "Changes vs pristine", "Reset"])

    with files_tab:
        try:
            files = client.repo_files(repo_id).get("files", [])
        except API_ERRORS as exc:
            show_problem(exc)
            files = []
        paths = [item_name(f, "path") for f in files]
        if paths:
            path = st.selectbox("File", paths, key="repo_file_choice")
            try:
                payload = client.repo_file(repo_id, path)
            except API_ERRORS as exc:
                show_problem(exc)
            else:
                content = payload.get("content") if isinstance(payload, dict) else None
                if isinstance(content, str):
                    st.code(content, language="python" if path.endswith(".py") else "text")
                else:
                    st.json(payload)
        else:
            st.caption("No files to show.")

    with diff_tab:
        try:
            payload = client.repo_diff(repo_id)
        except API_ERRORS as exc:
            show_problem(exc)
        else:
            diff = payload.get("diff") if isinstance(payload, dict) else None
            if isinstance(diff, str):
                st.code(diff, language="diff") if diff.strip() else st.caption("No differences.")
            else:
                st.json(payload)

    with reset_tab:
        st.caption("Restores the demo repository from the pristine sample project.")
        sure = st.checkbox("I want to discard every change in this repository", key="reset_sure")
        if st.button("Reset repository", key="reset_repo", disabled=not sure):
            try:
                outcome = client.reset_repo(repo_id)
            except API_ERRORS as exc:
                show_problem(exc)
            else:
                st.success("Repository reset.")
                st.json(outcome)


# ---- history page -----------------------------------------------------------------
def _events_of(data: Any) -> list[dict]:
    events = data.get("events") if isinstance(data, dict) else data
    return [e for e in events if isinstance(e, dict)] if isinstance(events, list) else []


@st.cache_data(ttl=30, show_spinner=False)
def _run_summary(_client: ApiClient, base_url: str, name: str) -> dict:
    """Prompt/model/repo of one run log, read from its run_start event."""
    try:
        events = _events_of(_client.history_item(name, 5000))
    except API_ERRORS:
        return {}
    start = next((e for e in events if e.get("type") == "run_start"), {})
    return {"prompt": start.get("task", ""), "model": start.get("model", ""),
            "repo": start.get("repo", "")}


def _collect_run(events: list[dict]) -> dict:
    """Pull the user-facing facts out of a raw run log."""
    run: dict[str, Any] = {"tests": [], "steps": []}
    for e in events:
        kind, node, data = e.get("type", ""), e.get("node", ""), e.get("data") or {}
        if kind == "run_start":
            run.update(task=e.get("task", ""), model=e.get("model", ""), repo=e.get("repo", ""))
        elif kind == "plan":
            run["plan"] = data.get("plan") or {}
        elif kind == "test":
            run["tests"].append(data)
        elif kind == "diff":
            run["diff"] = data.get("diff", "")
        elif kind == "error":
            run["error"] = data
        elif kind == "done":
            run["done"] = data
        elif kind == "node_done":
            if node == "select_files":
                run["selected"], run["reason"] = data.get("relevant_files") or [], data.get("reason")
            elif node == "read_files":
                run["read"] = data.get("files_read") or []
            elif node == "explain":
                run["explanation"] = data.get("explanation", "")
            elif node == "apply_changes":
                run["applied_files"] = data.get("files") or []
        # one timeline line per step an average user cares about
        line = _timeline_line(kind, node, data, e)
        if line:
            run["steps"].append(line)
    done = run.get("done") or {}
    if done.get("explanation"):
        run["explanation"] = done["explanation"]
    if done.get("diff") and not run.get("diff"):
        run["diff"] = done["diff"]
    return run


def _timeline_line(kind: str, node: str, data: dict, event: dict) -> str | None:
    secs = event.get("elapsed_ms")
    when = f" · {secs / 1000:.1f}s" if isinstance(secs, (int, float)) else ""
    if kind == "run_start":
        return f"▶️ Task received{when}"
    if kind == "plan":
        n = len((data.get("plan") or {}).get("steps", []))
        return f"📝 Plan ready ({n} step{'s' if n != 1 else ''}){when}"
    if kind == "test":
        verdict = "passed" if data.get("passed") else "no tests collected" if data.get(
            "no_tests") else "failed"
        icon = "✅" if data.get("passed") else "⚠️" if data.get("no_tests") else "❌"
        return f"{icon} Tests {verdict} (attempt {data.get('attempt', '?')}){when}"
    if kind == "diff":
        n = len(data.get("files") or [])
        return f"🧾 Diff ready ({n} file{'s' if n != 1 else ''}){when}"
    if kind == "error":
        return f"🛑 Error: {data.get('message', 'unknown error')}{when}"
    if kind == "done":
        return f"🏁 Finished ({data.get('status', 'done')}){when}"
    if kind == "node_done":
        files = lambda v: ", ".join(str(x) for x in v) if v else "none"  # noqa: E731
        return {
            "select_files": f"✔️ Selected files: {files(data.get('relevant_files'))}{when}",
            "read_files": f"✔️ Read files: {files(data.get('files_read'))}{when}",
            "explain": f"✔️ Wrote the explanation{when}",
            "apply_changes": f"✔️ Applied: {files(data.get('files'))}{when}",
        }.get(node)
    return None


def _run_view(events: list[dict], name: str) -> None:
    run = _collect_run(events)
    done = run.get("done") or {}
    repo = str(run.get("repo", "")).replace("\\", "/").rstrip("/").split("/")[-1]
    state = done.get("status") or ("error" if run.get("error") else "unfinished")
    st.caption(f"run `{name.split('_')[-1].removesuffix('.jsonl')}` · repo `{repo or '?'}` · "
               f"model `{run.get('model') or '?'}` · state **{state}**")

    if run.get("error"):
        st.error(error_text(run["error"]))
    elif done.get("applied"):
        st.success(f"Changes applied to `{repo}`.")
    elif state == "rejected":
        st.info("Plan rejected. No files were changed.")
    elif state == "declined":
        st.info("Changes were not applied. No files were changed.")
    elif state == "unfinished":
        st.info("This run did not finish.")

    if run.get("task"):
        st.markdown(f"**Task:** {run['task']}")
    if run.get("selected"):
        st.caption("Files the agent looked at: " + ", ".join(f"`{f}`" for f in run["selected"]))
        if run.get("reason"):
            st.caption(run["reason"])
    if run.get("read"):
        st.caption("Files read: " + ", ".join(f"`{f}`" for f in run["read"]))

    plan = run.get("plan")
    if plan:
        with st.expander("Plan", expanded=not run.get("diff")):
            st.markdown(plan.get("summary", ""))
            if plan.get("steps"):
                st.table([{"File": s.get("file"), "Action": s.get("action")} for s in plan["steps"]])
            for note in plan.get("assumptions") or []:
                st.caption(f"Assumption: {note}")

    if run["tests"]:
        with st.expander("Test results", expanded=not run.get("diff")):
            for t in run["tests"]:
                verdict = ":green[passed]" if t.get("passed") else (
                    ":orange[no tests collected]" if t.get("no_tests") else ":red[failed]")
                retry = " · retried" if t.get("will_retry") else ""
                st.markdown(f"**Attempt {t.get('attempt')}**: {verdict}{retry}")
                if not t.get("passed") and t.get("output"):
                    st.code(str(t["output"])[-3000:], language="text")

    if run.get("diff"):
        with st.expander("Proposed diff", expanded=True):
            st.code(run["diff"], language="diff")
            st.download_button("Download patch", data=run["diff"],
                               file_name=f"{name.split('_')[-1].removesuffix('.jsonl')}.patch",
                               mime="text/x-diff", key=f"history_patch_{name}")

    if run.get("explanation"):
        with st.expander("Explanation", expanded=True):
            st.markdown(run["explanation"])

    if run.get("applied_files"):
        st.caption("Applied to: " + ", ".join(f"`{f}`" for f in run["applied_files"]))

    with st.expander("Timeline", expanded=False):
        st.markdown("\n\n".join(run["steps"]) or "_No events._")


def history_page(client: ApiClient) -> None:
    st.header("Run history")
    try:
        items = client.history()
    except API_ERRORS as exc:
        show_problem(exc)
        return
    if not items:
        st.caption("No run logs yet.")
        return

    def modified_of(item: Any) -> str:
        return str(item.get("modified", "")) if isinstance(item, dict) else ""

    items = sorted(items, key=modified_of, reverse=True)
    rows, labels = [], {}
    for pos, item in enumerate(items):
        name = item_name(item)
        info = _run_summary(client, client.base_url, name) if pos < 30 else {}
        prompt = info.get("prompt") or "(prompt unavailable)"
        rows.append({"prompt": prompt, "model": info.get("model", ""),
                     "modified": modified_of(item)[:19].replace("T", " ")})
        labels[name] = f"{prompt[:90]} · {modified_of(item)[11:19]}"
    st.dataframe(rows, use_container_width=True, hide_index=True,
                 column_config={"prompt": st.column_config.TextColumn("prompt", width="large")})

    name = st.selectbox("Run log", list(labels), format_func=labels.get, key="history_choice")
    try:
        data = client.history_item(name, 5000)
    except API_ERRORS as exc:
        show_problem(exc)
        return
    events = _events_of(data)
    if not events:
        st.caption("This log has no events.")
        return
    _run_view(events, name)


# ---- scripts page -----------------------------------------------------------------
_BOOL_PARAMS = {"apply", "dry_run", "yes", "auto_apply"}


def _param_names(spec: Any) -> list[str]:
    params = spec.get("params") if isinstance(spec, dict) else None
    return [str(p.get("name")) if isinstance(p, dict) else str(p) for p in params or []]


def _param_value(raw: str) -> Any:
    """'5' -> 5, 'true' -> True, anything else stays text."""
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _params_form(script: str, params: list[str]) -> dict[str, Any]:
    """One input per parameter the script declares; returns the request body."""
    body: dict[str, Any] = {}
    if not params:
        st.caption("This script takes no parameters.")
        return body
    for p in params:
        key = f"script_{script}_{p}"
        if p == "task":
            text = st.text_area("Task", key=key, height=90,
                                placeholder="What should the agent change in the demo repo?")
            if text.strip():
                body[p] = text.strip()
        elif p == "model":
            current = st.session_state.get("sidebar_model")
            body[p] = st.selectbox("Model", MODEL_OPTIONS, key=key,
                                   index=MODEL_OPTIONS.index(current) if current in MODEL_OPTIONS else 0)
        elif p in _BOOL_PARAMS:
            body[p] = st.checkbox(p.replace("_", " ").capitalize(), key=key)
        else:
            text = st.text_input(p, key=key)
            if text.strip():
                body[p] = _param_value(text.strip())
    return body


def _render_job(job: dict) -> None:
    """A finished (or running) script job in plain language."""
    state = str(job.get("state", "?"))
    script = job.get("script", "?")
    secs = job.get("duration_s")
    took = f" in {secs:.1f}s" if isinstance(secs, (int, float)) else ""
    if state == "succeeded":
        st.success(f"`{script}` succeeded{took}.")
    elif state in ("running", "queued", "pending"):
        st.info(f"`{script}` is {state}. Press Refresh in a moment.")
    else:
        st.error(f"`{script}` {state} (exit code {job.get('exit_code', '?')}){took}.")
    if job.get("args"):
        st.caption("Arguments: " + " ".join(f"`{a}`" for a in job["args"]))
    out = (job.get("stdout") or "").strip("\n")
    err = (job.get("stderr") or "").strip("\n")
    if out:
        with st.expander("Output", expanded=True):
            st.code(out, language="text")
    if err:
        with st.expander("Errors", expanded=state != "succeeded"):
            st.code(err, language="text")
    if not out and not err and state == "succeeded":
        st.caption("The script printed nothing.")
    st.caption(f"job `{str(job.get('job_id', ''))[:8]}`")


def scripts_page(client: ApiClient) -> None:
    st.header("Scripts")
    try:
        specs = client.scripts()
    except API_ERRORS as exc:
        show_problem(exc)
        return
    if specs and isinstance(specs[0], dict):
        st.dataframe(specs, use_container_width=True, hide_index=True)
    names = [item_name(s) for s in specs]
    if names:
        name = st.selectbox("Script", names, key="script_choice")
        spec = next((s for s in specs if item_name(s) == name), {})
        body = _params_form(name, _param_names(spec))
        wait = st.slider("Seconds to wait for the result", 0, 120, 30, key="script_wait")
        if st.button("Run script", type="primary", key="run_script"):
            try:
                st.session_state.last_job = client.run_script(name, body, float(wait))
            except API_ERRORS as exc:
                show_problem(exc)
    if st.session_state.get("last_job"):
        st.subheader("Latest job")
        _render_job(st.session_state.last_job)
    st.subheader("All jobs")
    try:
        jobs = client.jobs()
    except API_ERRORS as exc:
        show_problem(exc)
        return
    if not jobs:
        st.caption("No jobs yet.")
        return
    cols = ("job_id", "script", "state", "exit_code", "duration_s")
    st.dataframe([{k: j.get(k) for k in cols} if isinstance(j, dict) else j for j in jobs],
                 use_container_width=True, hide_index=True)
    ids = [item_name(j, "job_id") for j in jobs]
    job_id = st.selectbox("Job", ids, key="job_choice")
    st.button("Refresh job", key="refresh_job")
    try:
        _render_job(client.job(job_id))
    except API_ERRORS as exc:
        show_problem(exc)


# ---- about page -------------------------------------------------------------------
def about_page() -> None:
    st.header("About")
    st.markdown(
        "This app is the front end for the **Coding Agent**. You describe a code change in plain "
        "language; the agent finds the relevant files, proposes a plan, writes the change, runs "
        "the tests and shows you a diff. **Nothing is written to a repository until you approve "
        "it.** Everything here talks to the backend API, so the backend must be running."
    )

    st.subheader("Getting around")
    st.markdown(
        "- **Sidebar, top:** the backend's API URL and its connection status. The URL comes from "
        "`CODING_AGENT_API_URL` in the environment. If it is "
        "wrong or the backend is stopped, you'll see an error and the pages stay hidden.\n"
        "- **Sidebar, Model:** the model used for new runs and for scripts that take a model.\n"
        "- **Sidebar, Page:** switch between the pages below. Your run in progress is kept while "
        "you look at other pages, so you can come back to it."
    )

    st.subheader("Pages")
    with st.expander("Run agent", expanded=True):
        st.markdown(
            "Start and follow a change from request to result.\n\n"
            "1. Click an example or type what the agent should change.\n"
            "2. Pick the repository (usually the demo repo). The model comes from the sidebar.\n"
            "3. Click **Start run**. A live timeline shows each step as it happens.\n"
            "4. **Approve or reject the plan.** You can add an optional note for the model. "
            "Nothing is written at this step.\n"
            "5. The agent generates the change and runs the tests. You then see the plan, test "
            "results, the proposed diff (with a patch download) and an explanation.\n"
            "6. **Apply or decline the changes.** If the tests did not pass, the change is marked "
            "unverified and you must tick a box to apply it anyway.\n"
            "7. Click **Start another run** to begin again. **Cancel run** stops a run that is "
            "still working."
        )
    with st.expander("Repositories"):
        st.markdown(
            "Look at the repositories the agent can work on. Select one, then use the tabs:\n\n"
            "- **Files:** browse the files and read their contents.\n"
            "- **Changes vs pristine:** everything that differs from the original sample project, "
            "which is handy after applying a change.\n"
            "- **Reset:** discard all changes and restore the demo repo from the pristine sample "
            "(you must tick the confirmation box first)."
        )
    with st.expander("History"):
        st.markdown(
            "Every run is saved as a log. The table lists the **prompt** of each run with its "
            "model and time, newest first.\n\n"
            "Choose a run in **Run log** to see it the way it looked when it finished: the task, "
            "files the agent looked at, plan, test results, diff, explanation and what was "
            "applied. The **Timeline** at the bottom lists the steps with how long each took."
        )
    with st.expander("Scripts"):
        st.markdown(
            "Small maintenance jobs that run on the backend:\n\n"
            "- **smoke:** one call to the model to check the API key and model name.\n"
            "- **reset_demo:** replace the demo repo with a fresh copy of the sample project.\n"
            "- **check_demo:** run the demo repo's tests.\n"
            "- **dev_run:** run the agent end to end on the demo repo, answering the approvals "
            "automatically.\n\n"
            "Select a script, fill in its parameters if it has any, choose how long to wait and "
            "click **Run script**. Results appear under **Latest job**; **All jobs** lets you "
            "reopen earlier ones. A job that is still running can be checked with **Refresh job**."
        )

    st.subheader("Good to know")
    st.markdown(
        "- Only one run can change a repository at a time. If it is busy, wait for the other run "
        "or script to finish.\n"
        "- The backend limits how many runs you can start per hour and per day.\n"
        "- If a model call fails with a rate-limit message, wait the time it mentions and try again.\n"
        "- Changes are applied only to the demo repo, so you can always reset it from "
        "**Repositories** or **Scripts**."
    )


# ---- main -------------------------------------------------------------------------
def main() -> None:
    client, health, page = sidebar()
    if page == "About":
        about_page()
        return
    if health is None:
        st.title("Coding Agent")
        st.error("The backend is not reachable. Start it, then check CODING_AGENT_API_URL in your environment (.env).")
        return
    if page == "Run agent":
        run_view(client) if st.session_state.get("run_id") else start_form(client, health)
    elif page == "Repositories":
        repos_page(client)
    elif page == "History":
        history_page(client)
    else:
        scripts_page(client)


main()