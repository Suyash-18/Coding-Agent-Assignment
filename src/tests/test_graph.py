import uuid
from pathlib import Path

import pytest

from coding_agent.runner import resume_agent, stream_agent
from coding_agent.schemas import ChangeSet, FileChange, FileSelection, Plan, PlanStep
from tests.fakes import FakeLLM

SAMPLE = Path(__file__).resolve().parent.parent / "sample_project"

VALIDATED_MODELS = '''from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    name: str = Field(min_length=1)
    email: str
    age: int = Field(ge=0, le=130)


class User(UserCreate):
    id: int
'''
NEW_TEST = '''

def test_create_user_rejects_negative_age(client):
    r = client.post("/users", json={**VALID, "age": -5})
    assert r.status_code == 422
'''


def make_llm(files=("models.py", "tests/test_users.py", "ghost.py")):
    old_tests = (SAMPLE / "tests" / "test_users.py").read_text()
    return FakeLLM(
        structured={
            FileSelection: [FileSelection(relevant_files=list(files), reasoning="models + tests")],
            Plan: [Plan(
                summary="Add field constraints and a test.",
                steps=[PlanStep(file="models.py", action="Add Field constraints"),
                       PlanStep(file="tests/test_users.py", action="Test invalid age")],
                assumptions=["Age must be between 0 and 130"],
            )],
            ChangeSet: [ChangeSet(changes=[
                FileChange(path="models.py", content=VALIDATED_MODELS),
                FileChange(path="tests/test_users.py", content=old_tests + NEW_TEST),
            ])],
        },
        text=["Added constraints to the user model and a regression test."],
    )


@pytest.fixture
def thread_id():
    return uuid.uuid4().hex


def types(events):
    return [e.type for e in events]


def test_full_flow_with_approval(thread_id):
    first = list(stream_agent("Add input validation", str(SAMPLE), thread_id=thread_id, llm=make_llm()))
    assert types(first) == ["start", "node_done", "node_done", "node_done", "plan", "interrupt"]
    assert first[2].data["relevant_files"] == ["models.py", "tests/test_users.py"]  # ghost dropped
    assert first[-1].data["stage"] == "plan"

    second = list(resume_agent(thread_id, {"approved": True}))
    assert types(second) == ["node_done", "node_done", "diff", "node_done", "done"]
    done = second[-1].data
    assert done["status"] == "completed"
    assert "Field(ge=0, le=130)" in done["diff"]
    assert done["changed_files"] == ["models.py", "tests/test_users.py"]
    assert "Field" not in (SAMPLE / "models.py").read_text()  # repo untouched


def test_rejected_plan_stops_cleanly(thread_id):
    list(stream_agent("Add input validation", str(SAMPLE), thread_id=thread_id, llm=make_llm()))
    events = list(resume_agent(thread_id, {"approved": False}))
    assert types(events) == ["node_done", "done"]
    assert events[-1].data["status"] == "rejected"
    assert events[-1].data["diff"] == ""


def test_unmatched_task_returns_error_event(thread_id):
    events = list(stream_agent("do something", str(SAMPLE), thread_id=thread_id, llm=make_llm(["ghost.py"])))
    assert types(events) == ["start", "node_done", "error"]
    assert "could not match" in events[-1].data["message"]


def test_resume_unknown_run():
    events = list(resume_agent("does-not-exist", {"approved": True}))
    assert events[0].type == "error"