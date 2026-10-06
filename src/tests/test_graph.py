import uuid
from pathlib import Path

import pytest

from coding_agent.runner import resume_agent, stream_agent
from coding_agent.schemas import ChangeSet, FileChange, FileSelection, Plan, PlanStep
from tests.fakes import FakeLLM

SAMPLE = Path(__file__).resolve().parent.parent / "sample_project"
OLD_TESTS = (SAMPLE / "tests" / "test_users.py").read_text(encoding="utf-8")

VALIDATED_MODELS = '''from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    name: str = Field(min_length=1)
    email: str
    age: int = Field(ge=0, le=130)


class User(UserCreate):
    id: int
'''
# Compiles fine but breaks existing tests: every valid name is shorter than 100 chars.
BROKEN_MODELS = VALIDATED_MODELS.replace("min_length=1", "min_length=100")
NEW_TEST = '''

def test_create_user_rejects_negative_age(client):
    r = client.post("/users", json={**VALID, "age": -5})
    assert r.status_code == 422
'''

GOOD = ChangeSet(changes=[
    FileChange(path="models.py", content=VALIDATED_MODELS),
    FileChange(path="tests/test_users.py", content=OLD_TESTS + NEW_TEST),
])
BAD = ChangeSet(changes=[FileChange(path="models.py", content=BROKEN_MODELS)])


def real_plan() -> Plan:
    return Plan(
        summary="Add field constraints and a test.",
        steps=[PlanStep(file="models.py", action="Add Field constraints"),
               PlanStep(file="tests/test_users.py", action="Test invalid age")],
        assumptions=["Age must be between 0 and 130"],
        needs_files=[],
    )


def make_llm(files=("models.py", "tests/test_users.py", "ghost.py"),
             plan_requests=(), change_sets=None):
    return FakeLLM(
        structured={
            FileSelection: [FileSelection(relevant_files=list(files), reasoning="models + tests")],
            Plan: [*plan_requests, real_plan()],
            ChangeSet: list(change_sets or [GOOD]),
        },
        text=["Added constraints to the user model and a regression test."],
    )


@pytest.fixture
def thread_id():
    return uuid.uuid4().hex


def types(events):
    return [e.type for e in events]


def start(repo, thread_id, llm, task="Add input validation"):
    return list(stream_agent(task, str(repo), thread_id=thread_id, llm=llm))


def test_full_flow_applies_only_after_approval(sample_repo, thread_id):
    first = start(sample_repo, thread_id, make_llm())
    assert types(first) == ["start", "node_done", "node_done", "node_done", "plan", "interrupt"]
    assert first[2].data["relevant_files"] == ["models.py", "tests/test_users.py"]  # ghost dropped
    assert first[-1].data["stage"] == "plan"

    second = list(resume_agent(thread_id, {"approved": True}))
    assert types(second) == ["node_done", "node_done", "test", "diff", "node_done", "interrupt"]
    assert second[2].data["passed"] is True and second[2].data["attempt"] == 1
    assert second[-1].data["stage"] == "apply" and second[-1].data["test_passed"] is True
    assert "Field" not in (sample_repo / "models.py").read_text()  # nothing written yet

    third = list(resume_agent(thread_id, {"approved": True}))
    assert types(third) == ["node_done", "node_done", "done"]
    done = third[-1].data
    assert done["status"] == "applied" and done["applied"] is True and done["test_passed"] is True
    assert "Field(ge=0, le=130)" in (sample_repo / "models.py").read_text()
    assert "Field" not in (SAMPLE / "models.py").read_text()  # real sample untouched


def test_retry_fixes_a_failing_attempt(sample_repo, thread_id):
    llm = make_llm(change_sets=[BAD, GOOD])
    start(sample_repo, thread_id, llm)
    second = list(resume_agent(thread_id, {"approved": True}))

    assert types(second) == [
        "node_done", "node_done", "test", "node_done", "test", "diff", "node_done", "interrupt",
    ]
    tests = [e for e in second if e.type == "test"]
    assert tests[0].data["passed"] is False and tests[0].data["will_retry"] is True
    assert tests[1].data["passed"] is True and tests[1].data["attempt"] == 2

    retry_prompt = [m for name, m in llm.calls if name == "ChangeSet"][1][-1].content
    assert "previous attempt" in retry_prompt.lower()
    assert "FAILED" in retry_prompt and "test_create_user" in retry_prompt

    done = list(resume_agent(thread_id, {"approved": True}))[-1].data
    assert done["status"] == "applied" and done["attempts"] == 2


def test_gives_up_after_max_retries_and_user_declines(sample_repo, thread_id):
    start(sample_repo, thread_id, make_llm(change_sets=[BAD, BAD, BAD]))
    second = list(resume_agent(thread_id, {"approved": True}))

    tests = [e for e in second if e.type == "test"]
    assert [t.data["attempt"] for t in tests] == [1, 2, 3]
    assert tests[-1].data["will_retry"] is False and tests[-1].data["passed"] is False
    assert second[-1].type == "interrupt" and second[-1].data["test_passed"] is False

    last = list(resume_agent(thread_id, {"approved": False}))
    assert types(last) == ["node_done", "done"]
    assert last[-1].data["status"] == "declined" and last[-1].data["applied"] is False
    assert "Field" not in (sample_repo / "models.py").read_text()


def test_planner_can_request_more_files(sample_repo, thread_id):
    draft = Plan(summary="need entry point", steps=[], assumptions=[], needs_files=["main.py"])
    llm = make_llm(files=("routes.py",), plan_requests=[draft])
    events = start(sample_repo, thread_id, llm, "Add a /health endpoint")

    assert types(events) == ["start"] + ["node_done"] * 6 + ["plan", "interrupt"]
    assert events[4].data == {"needs_files": ["main.py"]}
    assert events[5].node == "expand_files"
    assert events[6].data["files_read"] == ["routes.py", "main.py"]

    plan_calls = [msgs for name, msgs in llm.calls if name == "Plan"]
    assert len(plan_calls) == 2
    assert '<file path="main.py">' in plan_calls[1][-1].content


def test_rejected_plan_stops_cleanly(sample_repo, thread_id):
    start(sample_repo, thread_id, make_llm())
    events = list(resume_agent(thread_id, {"approved": False}))
    assert types(events) == ["node_done", "done"]
    assert events[-1].data["status"] == "rejected"
    assert events[-1].data["diff"] == ""


def test_unmatched_task_returns_error_event(sample_repo, thread_id):
    events = start(sample_repo, thread_id, make_llm(["ghost.py"]), "do something")
    assert types(events) == ["start", "node_done", "error"]
    assert "could not match" in events[-1].data["message"]


def test_resume_unknown_run():
    events = list(resume_agent("does-not-exist", {"approved": True}))
    assert events[0].type == "error"