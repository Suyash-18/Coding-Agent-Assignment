from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from coding_agent import prompts
from coding_agent.errors import AgentError, PathViolation
from coding_agent.llm import clean_text, invoke_structured
from coding_agent.schemas import ChangeSet, FileSelection, Plan
from coding_agent.state import AgentState
from coding_agent.tools import (
    apply_changes,
    build_diff,
    cleanup_temp,
    copy_to_temp,
    is_protected,
    list_files,
    read_file,
    run_tests,
    safe_path,
)

MAX_SELECTED_FILES = 6
MAX_EXPANSIONS = 1
MAX_EXTRA_FILES = 3
MAX_CONTEXT_CHARS = 20_000
MAX_DIFF_CHARS_IN_PROMPT = 12_000
MAX_RETRIES = 2
TEST_TIMEOUT = 60
PYTEST_NO_TESTS = 5  # pytest's exit code when nothing was collected

def _clean_rel(path: str) -> str:
    return path.strip().replace("\\", "/").removeprefix("./")


class AgentNodes:
    """Graph nodes. LLM-backed: select_files, make_plan, generate_changes, explain."""

    def __init__(self, llm):
        self.llm = llm

    # ---- code-only nodes -------------------------------------------------
    def scan_repo(self, state: AgentState) -> dict[str, Any]:
        files = list_files(state["repo_path"])
        if not files:
            raise AgentError("No readable files were found in this repository.")
        return {"file_tree": [f.path for f in files]}

    def read_files(self, state: AgentState) -> dict[str, Any]:
        contents = {p: read_file(state["repo_path"], p) for p in state["relevant_files"]}
        total = sum(len(text) for text in contents.values())
        if total > MAX_CONTEXT_CHARS:
            raise AgentError(
                f"The selected files are too large ({total:,} characters, "
                f"limit {MAX_CONTEXT_CHARS:,}). Try a more specific task."
            )
        return {"file_contents": contents}

    def expand_files(self, state: AgentState) -> dict[str, Any]:
        current = list(state["relevant_files"])
        merged = current + [p for p in state["needs_files"] if p not in current]
        return {
            "relevant_files": merged,
            "needs_files": [],
            "expansions": state.get("expansions", 0) + 1,
        }

    def route_after_draft(self, state: AgentState) -> str:
        return "expand" if state.get("needs_files") else "approve"

    def plan_approval(self, state: AgentState) -> dict[str, Any]:
        # Keep this node side-effect free: on resume it re-runs from the top.
        decision = interrupt(
            {"stage": "plan", "plan": state["plan"], "files": state["relevant_files"]}
        )
        if isinstance(decision, dict):
            approved = bool(decision.get("approved"))
            feedback = str(decision.get("feedback", "")).strip()
        else:
            approved, feedback = bool(decision), ""
        if not approved:
            return {"plan_approved": False, "status": "rejected"}
        return {"plan_approved": True, "feedback": feedback}

    def route_after_plan(self, state: AgentState) -> str:
        return "generate" if state.get("plan_approved") else "stop"

    def build_diff_node(self, state: AgentState) -> dict[str, Any]:
        parts = [
            build_diff(self._existing(state, rel), new, rel)
            for rel, new in state["changes"].items()
        ]
        return {"diff": "".join(parts), "changed_files": list(state["changes"])}

       # ---- validation (code only) -----------------------------------------
    def run_tests_node(self, state: AgentState) -> dict[str, Any]:
        """Apply the proposed changes to a temp copy and run the tests there."""
        work = copy_to_temp(state["repo_path"])
        attempts = state.get("attempts", 0) + 1
        try:
            apply_changes(work, state["changes"])
            result = run_tests(work, timeout=TEST_TIMEOUT)
        finally:
            cleanup_temp(work)

        no_tests = result.returncode == PYTEST_NO_TESTS
        return {
            "attempts": attempts,
            "test_passed": result.passed,
            "test_output": result.output,
            "no_tests": no_tests,
            "will_retry": (not result.passed) and (not no_tests) and attempts <= MAX_RETRIES,
        }

    def route_after_tests(self, state: AgentState) -> str:
        return "retry" if state.get("will_retry") else "finish"

    # ---- apply approval --------------------------------------------------
    def apply_approval(self, state: AgentState) -> dict[str, Any]:
        # Side-effect free (re-runs on resume). Writing happens in apply_changes_node.
        decision = interrupt({
            "stage": "apply",
            "diff": state["diff"],
            "files": state["changed_files"],
            "test_passed": state.get("test_passed", False),
            "no_tests": state.get("no_tests", False),
            "attempts": state.get("attempts", 0),
        })
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        if approved:
            return {"apply_approved": True}
        return {"apply_approved": False, "status": "declined"}

    def route_after_apply(self, state: AgentState) -> str:
        return "apply" if state.get("apply_approved") else "stop"

    def apply_changes_node(self, state: AgentState) -> dict[str, Any]:
        written = apply_changes(state["repo_path"], state["changes"])
        return {"applied": True, "status": "applied", "applied_files": written}

    # ---- LLM nodes -------------------------------------------------------
    def select_files(self, state: AgentState) -> dict[str, Any]:
        tree = state["file_tree"]
        result = invoke_structured(
            self.llm, FileSelection, prompts.select_files_messages(state["task"], tree)
        )
        known = set(tree)
        chosen: list[str] = []
        for raw in result.relevant_files:  # drops hallucinated paths
            path = _clean_rel(raw)
            if path in known and path not in chosen:
                chosen.append(path)
        if not chosen:
            raise AgentError(
                "The agent could not match your task to any file in this repo. "
                "Try describing the task more specifically."
            )
        return {
            "relevant_files": chosen[:MAX_SELECTED_FILES],
            "selection_reason": result.reasoning,
        }

    def make_plan(self, state: AgentState) -> dict[str, Any]:
        contents = state["file_contents"]
        unread = [p for p in state["file_tree"] if p not in contents]
        plan = invoke_structured(
            self.llm, Plan, prompts.plan_messages(state["task"], contents, unread)
        )
        can_expand = state.get("expansions", 0) < MAX_EXPANSIONS
        wanted = self._valid_requests(plan.needs_files, unread) if can_expand else []
        if wanted:  # discard this draft; read the requested files and plan again
            return {"needs_files": wanted}
        if not plan.steps:
            raise AgentError(
                "The model could not produce a plan for this task. Try rephrasing it."
            )
        return {"plan": plan.model_dump(exclude={"needs_files"}), "needs_files": []}

    def generate_changes(self, state: AgentState) -> dict[str, Any]:
        repo = state["repo_path"]
        root = Path(repo).resolve()
        is_retry = state.get("attempts", 0) > 0 and not state.get("test_passed", False)
        previous = dict(state.get("changes", {})) if is_retry else {}

        result = invoke_structured(
            self.llm, ChangeSet,
            prompts.generate_messages(
                state["task"], state["plan"], state["file_contents"],
                state.get("feedback", ""),
                previous=previous,
                test_output=state.get("test_output", "") if is_retry else "",
            ),
        )

        changes = dict(previous)  # retries merge into the previous attempt
        for item in result.changes:
            rel = _clean_rel(item.path)
            target = safe_path(repo, rel)
            if is_protected(target.relative_to(root)):
                raise PathViolation(f"The model tried to change a protected file: {rel}")
            if item.content == self._existing(state, rel):
                changes.pop(rel, None)  # same as the original: no change (or a revert)
            else:
                changes[rel] = item.content
        if not changes:
            raise AgentError(
                "The model did not propose any changes. Try rephrasing the task."
            )
        return {"changes": changes}

    def explain(self, state: AgentState) -> dict[str, Any]:
        diff = state["diff"][:MAX_DIFF_CHARS_IN_PROMPT]
        reply = self.llm.invoke(prompts.explain_messages(state["task"], state["plan"], diff))
        text = clean_text(reply.content) or state["plan"]["summary"]
        return {"explanation": text, "status": "proposed"}

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _valid_requests(requested: list[str], unread: list[str]) -> list[str]:
        allowed = set(unread)
        picked: list[str] = []
        for raw in requested:
            path = _clean_rel(raw)
            if path in allowed and path not in picked:
                picked.append(path)
        return picked[:MAX_EXTRA_FILES]

    def _existing(self, state: AgentState, rel: str) -> str:
        """Current content of a file: what the model saw, else disk, else empty."""
        seen = state.get("file_contents", {})
        if rel in seen:
            return seen[rel]
        if safe_path(state["repo_path"], rel).is_file():
            return read_file(state["repo_path"], rel)
        return ""