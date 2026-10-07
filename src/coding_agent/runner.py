import traceback
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from coding_agent.config import DEFAULT_MODEL
from coding_agent.errors import AgentError
from coding_agent.graph import build_graph
from coding_agent.llm import get_llm
from coding_agent.runlog import RunLog

_CHECKPOINTER = MemorySaver()
_GRAPHS: dict[str, Any] = {}
_LOGS: dict[str, RunLog] = {}


def run_log_id(thread_id: str) -> str | None:
    """The run_id of a run's log in MongoDB, once something has been written."""
    log = _LOGS.get(thread_id)
    return log.run_id if log and log.written else None


@dataclass
class AgentEvent:
    type: str  # start | node_done | plan | test | diff | interrupt | error | done
    node: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SUMMARIES = {
    "scan_repo": lambda u: {"file_count": len(u["file_tree"])},
    "select_files": lambda u: {
        "relevant_files": u["relevant_files"], "reason": u["selection_reason"],
    },
    "read_files": lambda u: {"files_read": list(u["file_contents"])},
    "plan_approval": lambda u: {"approved": u.get("plan_approved", False)},
    "generate_changes": lambda u: {"files": list(u["changes"])},
    "explain": lambda u: {"explanation": u["explanation"]},
    "expand_files": lambda u: {"relevant_files": u["relevant_files"]},
    "apply_approval": lambda u: {"approved": u.get("apply_approved", False)},
    "apply_changes": lambda u: {"files": u["applied_files"]},
}


def _translate(node: str, update: dict[str, Any]) -> AgentEvent | None:
    if node in ("input_guard", "output_guard"):
        return None  # silent on success; violations surface as an `error` event
    if node == "make_plan":
        if "plan" in update:
            return AgentEvent("plan", node, {"plan": update["plan"]})
        return AgentEvent("node_done", node, {"needs_files": update["needs_files"]})
    if node == "run_tests":
        return AgentEvent("test", node, {
            "passed": update["test_passed"],
            "attempt": update["attempts"],
            "will_retry": update["will_retry"],
            "no_tests": update["no_tests"],
            "output": update["test_output"],
        })
    if node == "build_diff":
        return AgentEvent("diff", node, {"diff": update["diff"], "files": update["changed_files"]})
    summarize = _SUMMARIES.get(node, lambda u: {})
    return AgentEvent("node_done", node, summarize(update))


def _error(exc: Exception) -> AgentEvent:
    if isinstance(exc, AgentError):
        message = str(exc)
    else:
        message = f"Unexpected error: {type(exc).__name__}: {exc}"
    hint = getattr(exc, "hint", "") if isinstance(exc, AgentError) else ""
    return AgentEvent(
        "error", data={"message": message, "kind": type(exc).__name__, "hint": hint}
    )


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _drive(graph, graph_input, thread_id: str) -> Iterator[AgentEvent]:
    log = _LOGS.get(thread_id)
    for ev in _drive_events(graph, graph_input, thread_id, log):
        if log:
            log.event(ev)
        yield ev


def _drive_events(graph, graph_input, thread_id: str, log: RunLog | None) -> Iterator[AgentEvent]:
    config = _config(thread_id)
    try:
        for chunk in graph.stream(graph_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    yield AgentEvent("interrupt", data=update[0].value)
                    return
                event = _translate(node, update or {})
                if event is not None:
                    yield event
        final = graph.get_state(config).values
        yield AgentEvent("done", data={
            "status": final.get("status", "proposed"),
            "applied": final.get("applied", False),
            "test_passed": final.get("test_passed", False),
            "attempts": final.get("attempts", 0),
            "plan": final.get("plan"),
            "diff": final.get("diff", ""),
            "changed_files": final.get("changed_files", []),
            "explanation": final.get("explanation", ""),
        })
    except Exception as exc:  # boundary: never leak a traceback to the UI
        if log:
            log.write("traceback", traceback=traceback.format_exc())  # kept in the log only
        yield _error(exc)


def stream_agent(
    task: str,
    repo_path: str,
    model: str | None = None,
    thread_id: str | None = None,
    llm=None,
) -> Iterator[AgentEvent]:
    """Start a run. Stops at the plan-approval interrupt; continue with resume_agent."""
    thread_id = thread_id or uuid.uuid4().hex
    log = _LOGS[thread_id] = RunLog.create(thread_id)
    start = AgentEvent("start", data={"thread_id": thread_id})
    log.event(start)
    yield start
    model_name = model or DEFAULT_MODEL
    log.write("run_start", task=task, repo=str(repo_path), model=model_name)
    try:
        graph = build_graph(llm or get_llm(model_name), _CHECKPOINTER)
    except Exception as exc:
        log.write("traceback", traceback=traceback.format_exc())
        failure = _error(exc)
        log.event(failure)
        yield failure
        return
    _GRAPHS[thread_id] = graph
    yield from _drive(
        graph, {"task": task, "repo_path": str(repo_path), "model": model_name}, thread_id
    )


def resume_agent(thread_id: str, decision: dict[str, Any]) -> Iterator[AgentEvent]:
    """Continue a paused run. decision example: {"approved": True, "feedback": "..."}."""
    graph = _GRAPHS.get(thread_id)
    if graph is None:
        yield AgentEvent("error", data={"message": "Unknown or expired run.", "kind": "AgentError"})
        return
    yield from _drive(graph, Command(resume=decision), thread_id)