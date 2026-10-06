"""Per-run JSON logs: complete, ordered, redacted, and never able to break a run."""

import json

from coding_agent.config import ConfigError
from coding_agent.runner import resume_agent, run_log_path
from coding_agent.schemas import FileSelection
from tests.fakes import FakeLLM
from tests.test_graph import make_llm, start, types

FAKE_KEY = "gsk_" + "a1B2c3D4e5" * 3


def read_log(thread_id):
    path = run_log_path(thread_id)
    assert path is not None and path.exists()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_log_records_every_event_in_order(sample_repo, thread_id):
    start(sample_repo, thread_id, make_llm())
    list(resume_agent(thread_id, {"approved": True}))
    records = read_log(thread_id)

    kinds = [r["type"] for r in records]
    assert kinds[:2] == ["start", "run_start"]
    assert {"plan", "test", "diff", "interrupt"} <= set(kinds)
    assert kinds.count("interrupt") == 2  # plan approval and apply approval
    assert all(r["run_id"] == thread_id for r in records)
    elapsed = [r["elapsed_ms"] for r in records]
    assert elapsed == sorted(elapsed)


def test_run_start_record_has_task_repo_and_model(sample_repo, thread_id):
    start(sample_repo, thread_id, make_llm(), task="Add input validation")
    first = read_log(thread_id)[1]
    assert first["task"] == "Add input validation" and first["repo"] == str(sample_repo)
    assert first["model"]


def test_secrets_are_redacted_in_the_log(sample_repo, thread_id):
    start(sample_repo, thread_id, make_llm(), task=f"Add a constant with the value {FAKE_KEY}")
    text = run_log_path(thread_id).read_text(encoding="utf-8")
    assert FAKE_KEY not in text and "[REDACTED]" in text


def test_unexpected_exception_keeps_its_traceback_in_the_log_only(sample_repo, thread_id):
    fake = FakeLLM(structured={FileSelection: [RuntimeError("kaboom")]})
    events = start(sample_repo, thread_id, fake)
    assert "traceback" not in events[-1].data  # never shown to the UI
    traces = [r for r in read_log(thread_id) if r["type"] == "traceback"]
    assert traces and "kaboom" in traces[0]["traceback"]


def test_config_error_before_the_graph_is_logged(sample_repo, thread_id, monkeypatch):
    def boom(*a, **k):
        raise ConfigError("GROQ_API_KEY is missing.")

    monkeypatch.setattr("coding_agent.runner.get_llm", boom)
    events = start(sample_repo, thread_id, None)
    assert types(events) == ["start", "error"]
    kinds = [r["type"] for r in read_log(thread_id)]
    assert kinds == ["start", "run_start", "traceback", "error"]


def test_logging_can_be_switched_off(sample_repo, thread_id, monkeypatch):
    monkeypatch.setenv("AGENT_RUNS_DIR", "off")
    events = start(sample_repo, thread_id, make_llm())
    assert types(events)[-1] == "interrupt"
    assert run_log_path(thread_id) is None


def test_an_unwritable_log_location_never_breaks_the_run(sample_repo, thread_id, tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder")
    monkeypatch.setenv("AGENT_RUNS_DIR", str(blocker / "logs"))
    events = start(sample_repo, thread_id, make_llm())
    assert types(events) == ["start", "node_done", "node_done", "node_done", "plan", "interrupt"]
    assert run_log_path(thread_id) is None