"""Turn agent events into short human-readable timeline lines (no Streamlit here)."""

from __future__ import annotations

from typing import Any

STOP_TYPES = frozenset({"interrupt", "done", "error"})

_ICONS = {"start": "▶️", "plan": "📝", "test_ok": "✅", "test_fail": "❌", "test_warn": "⚠️",
          "diff": "🧾", "interrupt": "⏸️", "error": "🛑", "done": "🏁", "step": "✔️"}


def _files(value: Any) -> str:
    return ", ".join(str(v) for v in value) if value else "none"


def describe(event: dict[str, Any]) -> tuple[str, str] | None:
    """Return (icon, text) for an event, or None for events not worth showing."""
    kind, node, data = event.get("type", ""), event.get("node", ""), event.get("data") or {}

    if kind == "start":
        return _ICONS["start"], "Run started"
    if kind == "plan":
        steps = len((data.get("plan") or {}).get("steps", []))
        return _ICONS["plan"], f"Plan ready ({steps} step{'s' if steps != 1 else ''})"
    if kind == "test":
        attempt = data.get("attempt", "?")
        if data.get("passed"):
            return _ICONS["test_ok"], f"Tests passed (attempt {attempt})"
        if data.get("no_tests"):
            return _ICONS["test_warn"], f"No tests were collected (attempt {attempt})"
        suffix = ", retrying with the failure output" if data.get("will_retry") else ""
        return _ICONS["test_fail"], f"Tests failed (attempt {attempt}){suffix}"
    if kind == "diff":
        n = len(data.get("files") or [])
        return _ICONS["diff"], f"Diff ready ({n} file{'s' if n != 1 else ''})"
    if kind == "interrupt":
        stage = data.get("stage", "?")
        return _ICONS["interrupt"], f"Waiting for your decision ({stage})"
    if kind == "error":
        return _ICONS["error"], f"Error: {data.get('message', 'unknown error')}"
    if kind == "done":
        return _ICONS["done"], f"Finished ({data.get('status', 'done')})"
    if kind == "node_done":
        text = {
            "scan_repo": f"Scanned repository ({data.get('file_count', '?')} files)",
            "select_files": f"Selected files: {_files(data.get('relevant_files'))}",
            "read_files": f"Read {len(data.get('files_read') or [])} file(s)",
            "expand_files": "Added the extra files the planner asked for",
            "plan_approval": "Plan approved" if data.get("approved") else "Plan rejected",
            "generate_changes": f"Generated changes for: {_files(data.get('files'))}",
            "explain": "Wrote the explanation",
            "apply_approval": "Apply approved" if data.get("approved") else "Apply declined",
            "apply_changes": f"Applied: {_files(data.get('files'))}",
        }.get(node)
        if node == "make_plan" and data.get("needs_files"):
            text = f"Planner asked for more files: {_files(data['needs_files'])}"
        return (_ICONS["step"], text) if text else None
    label = f"{kind}: {node}" if node else str(kind)
    return _ICONS["step"], label


def timeline_markdown(events: list[dict[str, Any]]) -> str:
    lines = [f"{icon} {text}" for e in events if (d := describe(e))
             for icon, text in [d]]
    return "\n\n".join(lines) if lines else "_Waiting for the first event…_"
