from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # inputs
    task: str
    repo_path: str
    model: str
    # discovery
    file_tree: list[str]
    relevant_files: list[str]
    selection_reason: str
    file_contents: dict[str, str]
    needs_files: list[str]
    expansions: int
    # planning
    plan: dict[str, Any]
    plan_approved: bool
    feedback: str
    # changes
    changes: dict[str, str]
    changed_files: list[str]
    diff: str
    explanation: str
    # outcome
    status: str  # "proposed" | "rejected"  (Phase 4 adds "applied")
    applied: bool