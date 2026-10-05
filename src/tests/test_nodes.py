from pathlib import Path

import pytest

from coding_agent import nodes as nodes_module
from coding_agent.errors import AgentError, PathViolation
from coding_agent.nodes import AgentNodes
from coding_agent.schemas import ChangeSet, FileChange, FileSelection, Plan, PlanStep
from tests.fakes import FakeLLM

SAMPLE = Path(__file__).resolve().parent.parent / "sample_project"


def make(**kwargs) -> AgentNodes:
    return AgentNodes(FakeLLM(**kwargs))


def select(files):
    return FileSelection(relevant_files=files, reasoning="because")

print({"repo_path": str(SAMPLE)})

class TestScanRepo:
    def test_lists_sample_project(self):
        tree = make().scan_repo({"repo_path": str(SAMPLE)})["file_tree"]
        assert "routes.py" in tree and "tests/test_users.py" in tree
        assert not any(".env" in p or "__pycache__" in p for p in tree)

    def test_empty_repo(self, tmp_path):
        with pytest.raises(AgentError):
            make().scan_repo({"repo_path": str(tmp_path)})


class TestSelectFiles:
    TREE = ["a.py", "b.py", "tests/test_a.py"]

    def test_drops_unknown_and_dedupes_and_normalizes(self):
        n = make(structured={FileSelection: [select(["./a.py", "a.py", "ghost.py", "tests\\test_a.py"])]})
        out = n.select_files({"task": "t", "file_tree": self.TREE})
        assert out["relevant_files"] == ["a.py", "tests/test_a.py"]

    def test_caps_file_count(self):
        tree = [f"f{i}.py" for i in range(10)]
        n = make(structured={FileSelection: [select(tree)]})
        out = n.select_files({"task": "t", "file_tree": tree})
        assert len(out["relevant_files"]) == nodes_module.MAX_SELECTED_FILES

    def test_no_valid_files_raises(self):
        n = make(structured={FileSelection: [select(["ghost.py"])]})
        with pytest.raises(AgentError, match="could not match"):
            n.select_files({"task": "t", "file_tree": self.TREE})


class TestReadFiles:
    def test_reads_selected(self):
        out = make().read_files({"repo_path": str(SAMPLE), "relevant_files": ["utils.py"]})
        assert "normalize_email" in out["file_contents"]["utils.py"]

    def test_too_large(self, monkeypatch):
        monkeypatch.setattr(nodes_module, "MAX_CONTEXT_CHARS", 10)
        with pytest.raises(AgentError, match="too large"):
            make().read_files({"repo_path": str(SAMPLE), "relevant_files": ["routes.py"]})


class TestMakePlan:
    TREE = ["a.py", "b.py", "c.py"]

    def state(self, **extra):
        return {"task": "t", "file_tree": self.TREE, "file_contents": {"a.py": "1"}, **extra}

    def plan(self, needs=(), steps=True):
        return Plan(
            summary="s",
            steps=[PlanStep(file="a.py", action="x")] if steps else [],
            assumptions=[],
            needs_files=list(needs),
        )

    def test_returns_plan_without_needs_field(self):
        out = make(structured={Plan: [self.plan()]}).make_plan(self.state())
        assert out["plan"]["summary"] == "s"
        assert "needs_files" not in out["plan"]
        assert out["needs_files"] == []

    def test_valid_request_triggers_expansion(self):
        n = make(structured={Plan: [self.plan(needs=["./b.py", "ghost.py", "a.py"])]})
        assert n.make_plan(self.state()) == {"needs_files": ["b.py"]}  # unknown + already-read dropped

    def test_request_is_capped(self):
        tree = [f"f{i}.py" for i in range(8)]
        state = {"task": "t", "file_tree": tree, "file_contents": {"f0.py": "1"}}
        out = make(structured={Plan: [self.plan(needs=tree[1:])]}).make_plan(state)
        assert len(out["needs_files"]) == nodes_module.MAX_EXTRA_FILES

    def test_no_second_expansion(self):
        out = make(structured={Plan: [self.plan(needs=["b.py"])]}).make_plan(self.state(expansions=1))
        assert "plan" in out and out["needs_files"] == []

    def test_empty_plan_without_valid_request_raises(self):
        n = make(structured={Plan: [self.plan(needs=["ghost.py"], steps=False)]})
        with pytest.raises(AgentError, match="could not"):
            n.make_plan(self.state())


def test_expand_files_merges_and_counts():
    state = {"relevant_files": ["a.py"], "needs_files": ["b.py", "a.py"], "expansions": 0}
    assert make().expand_files(state) == {
        "relevant_files": ["a.py", "b.py"], "needs_files": [], "expansions": 1,
    }

class TestGenerateChanges:
    PLAN = {"summary": "s", "steps": [], "assumptions": []}

    def run(self, items, contents=None):
        llm = FakeLLM(structured={ChangeSet: [ChangeSet(changes=items)]})
        state = {
            "task": "t", "repo_path": str(SAMPLE), "plan": self.PLAN,
            "file_contents": contents or {"utils.py": "old\n"},
        }
        return AgentNodes(llm).generate_changes(state)

    def test_keeps_changed_drops_unchanged(self):
        out = self.run([FileChange(path="utils.py", content="new\n")])
        assert out["changes"] == {"utils.py": "new\n"}

    def test_all_unchanged_raises(self):
        with pytest.raises(AgentError, match="did not propose"):
            self.run([FileChange(path="utils.py", content="old\n")])

    def test_allows_new_file(self):
        out = self.run([FileChange(path="tests/test_new.py", content="x\n")])
        assert "tests/test_new.py" in out["changes"]

    @pytest.mark.parametrize("bad", ["../evil.py", ".env", "/etc/passwd"])
    def test_blocks_dangerous_paths(self, bad):
        with pytest.raises(PathViolation):
            self.run([FileChange(path=bad, content="x")])


def test_build_diff_node_handles_new_file():
    state = {"repo_path": str(SAMPLE), "file_contents": {}, "changes": {"tests/brand_new.py": "x\n"}}
    out = make().build_diff_node(state)
    assert "--- /dev/null" in out["diff"]
    assert out["changed_files"] == ["tests/brand_new.py"]


def test_explain_strips_reasoning():
    n = make(text=["<think>hmm</think>Added validation."])
    state = {"task": "t", "plan": {"summary": "s", "steps": [], "assumptions": []}, "diff": "d"}
    out = n.explain(state)
    assert out["explanation"] == "Added validation."
    assert out["status"] == "proposed"