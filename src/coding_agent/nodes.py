from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from coding_agent import prompts
from coding_agent.errors import AgentError, PathViolation
from coding_agent.llm import clean_text, structured
from coding_agent.schemas import ChangeSet, FileSelection, Plan
from coding_agent.state import AgentState
from coding_agent.tools import build_diff, is_protected, list_files, read_file, safe_path

MAX_SELECTED_FILES = 6
MAX_CONTEXT_CHARS = 20_000
MAX_DIFF_CHARS_IN_PROMPT = 12_000


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

    # ---- LLM nodes -------------------------------------------------------
    def select_files(self, state: AgentState) -> dict[str, Any]:
        tree = state["file_tree"]
        result = structured(self.llm, FileSelection).invoke(
            prompts.select_files_messages(state["task"], tree)
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
        plan = structured(self.llm, Plan).invoke(
            prompts.plan_messages(state["task"], state["file_contents"])
        )
        return {"plan": plan.model_dump()}

    def generate_changes(self, state: AgentState) -> dict[str, Any]:
        repo = state["repo_path"]
        root = Path(repo).resolve()
        result = structured(self.llm, ChangeSet).invoke(
            prompts.generate_messages(
                state["task"], state["plan"], state["file_contents"],
                state.get("feedback", ""),
            )
        )
        changes: dict[str, str] = {}
        for item in result.changes:
            rel = _clean_rel(item.path)
            target = safe_path(repo, rel)
            if is_protected(target.relative_to(root)):
                raise PathViolation(f"The model tried to change a protected file: {rel}")
            if item.content != self._existing(state, rel):
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
        return {"explanation": text, "status": "completed"}

    # ---- helpers ---------------------------------------------------------
    def _existing(self, state: AgentState, rel: str) -> str:
        """Current content of a file: what the model saw, else disk, else empty."""
        seen = state.get("file_contents", {})
        if rel in seen:
            return seen[rel]
        if safe_path(state["repo_path"], rel).is_file():
            return read_file(state["repo_path"], rel)
        return ""