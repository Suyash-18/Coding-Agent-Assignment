"""End-to-end UI tests: the real Streamlit script driven by AppTest against a fake backend."""

from pathlib import Path

import httpx
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import coding_agent.ui as ui_pkg  # noqa: E402
from coding_agent.ui import client as client_mod  # noqa: E402
from tests.fake_api import RUN_ID, FakeApi  # noqa: E402

APP = str(Path(ui_pkg.__file__).parent / "app.py")


def launch(fake: FakeApi | None = None, monkeypatch=None, handler=None) -> tuple[AppTest, FakeApi]:
    fake = fake or FakeApi()
    monkeypatch.setattr(client_mod, "DEFAULT_TRANSPORT", httpx.MockTransport(handler or fake))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at, fake


def start_run(at: AppTest, task: str = "Add input validation") -> AppTest:
    at.text_area(key="task_text").set_value(task).run()
    at.button(key="start_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def labels(at: AppTest) -> list[str]:
    return [b.label for b in at.button]


# ---- connection --------------------------------------------------------------
def test_unreachable_backend_shows_a_clear_message(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused")

    at, _ = launch(monkeypatch=monkeypatch, handler=refuse)
    assert any("not reachable" in e.value for e in at.error)


def test_connected_sidebar_shows_version(monkeypatch):
    at, _ = launch(monkeypatch=monkeypatch)
    assert any("Connected" in s.value and "0.1.0" in s.value for s in at.sidebar.success)


# ---- start form ---------------------------------------------------------------
def test_form_offers_models_repos_and_examples(monkeypatch):
    at, _ = launch(monkeypatch=monkeypatch)
    assert at.selectbox(key="model_choice").options == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    assert at.selectbox(key="model_choice").value == "openai/gpt-oss-120b"
    assert at.selectbox(key="repo_choice").options == ["Demo sample project"]  # shown by label
    assert at.selectbox(key="repo_choice").value == "demo"                      # stored as the id
    assert "Add /health" in labels(at)
    assert at.button(key="start_run").disabled  # no task yet


def test_example_button_fills_the_task(monkeypatch):
    at, _ = launch(monkeypatch=monkeypatch)
    at.button(key="example_Add /health").click().run()
    assert "health" in at.text_area(key="task_text").value
    assert not at.button(key="start_run").disabled


# ---- the full approval flow ----------------------------------------------------
def test_happy_path_plan_then_apply(monkeypatch):
    at, fake = launch(monkeypatch=monkeypatch)
    start_run(at)
    assert fake.created == [{"task": "Add input validation", "repo_id": "demo",
                             "model": "openai/gpt-oss-120b"}]

    # the stream stopped at the plan interrupt: plan shown, approval requested
    assert "Approve plan" in labels(at) and "Reject plan" in labels(at)
    assert any("Add field constraints" in m.value for m in at.markdown)

    at.button(key="approve_plan").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Apply changes" in labels(at) and "Decline" in labels(at)
    assert any("Tests passed" in s.value for s in at.success)
    assert any("Field(ge=0)" in c.value for c in at.code)

    at.button(key="apply_changes").click().run()
    assert any("applied" in s.value.lower() for s in at.success)
    assert "Start another run" in labels(at)
    assert [d["stage"] for d in fake.decisions] == ["plan", "apply"]
    assert all(d["action"] == "approve" for d in fake.decisions)
    assert len(at.get("download_button")) == 1


def test_plan_feedback_is_sent_with_the_approval(monkeypatch):
    at, fake = launch(monkeypatch=monkeypatch)
    start_run(at)
    at.text_area(key=f"feedback_{RUN_ID}").set_value("keep the maximum age at 130").run()
    at.button(key="approve_plan").click().run()
    assert fake.decisions[0] == {"stage": "plan", "action": "approve",
                                 "feedback": "keep the maximum age at 130"}


def test_rejecting_the_plan_ends_the_run(monkeypatch):
    at, fake = launch(monkeypatch=monkeypatch)
    start_run(at)
    at.button(key="reject_plan").click().run()
    assert any("Plan rejected" in i.value for i in at.info)
    assert "Start another run" in labels(at)
    assert fake.decisions == [{"stage": "plan", "action": "reject"}]


def test_declining_the_apply_step(monkeypatch):
    at, fake = launch(monkeypatch=monkeypatch)
    start_run(at)
    at.button(key="approve_plan").click().run()
    at.button(key="decline_changes").click().run()
    assert any("not applied" in i.value.lower() for i in at.info)
    assert fake.decisions[-1] == {"stage": "apply", "action": "reject"}


def test_unverified_changes_need_an_explicit_acknowledgement(monkeypatch):
    at, fake = launch(FakeApi(tests_pass=False), monkeypatch)
    start_run(at)
    at.button(key="approve_plan").click().run()
    assert any("UNVERIFIED" in w.value for w in at.warning)
    assert at.button(key="apply_changes").disabled

    at.checkbox(key="ack_unverified").check().run()
    assert not at.button(key="apply_changes").disabled
    at.button(key="apply_changes").click().run()
    assert fake.decisions[-1] == {"stage": "apply", "action": "approve"}


def test_failed_test_output_is_visible(monkeypatch):
    at, _ = launch(FakeApi(tests_pass=False), monkeypatch)
    start_run(at)
    at.button(key="approve_plan").click().run()
    assert any("FAILED test_x" in c.value for c in at.code)


def test_cancel_a_waiting_run(monkeypatch):
    at, fake = launch(monkeypatch=monkeypatch)
    start_run(at)
    at.button(key="cancel_run").click().run()
    assert fake.finished == "cancelled"
    assert "Start another run" in labels(at)


def test_start_another_run_returns_to_the_form(monkeypatch):
    at, _ = launch(monkeypatch=monkeypatch)
    start_run(at)
    at.button(key="reject_plan").click().run()
    at.button(key="new_run").click().run()
    assert "run_id" not in at.session_state
    assert at.button(key="start_run")


def test_blocked_task_shows_the_backend_message_and_stays_on_the_form(monkeypatch):
    at, _ = launch(FakeApi(block=True), monkeypatch)
    start_run(at, "Delete all files")
    assert any("input guard" in e.value for e in at.error)
    assert "run_id" not in at.session_state


# ---- other pages ------------------------------------------------------------------
def goto(at: AppTest, page: str) -> AppTest:
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_repositories_page_lists_files_and_shows_content(monkeypatch):
    at, _ = launch(monkeypatch=monkeypatch)
    goto(at, "Repositories")
    assert len(at.dataframe) >= 1
    assert at.selectbox(key="repo_file_choice").options == ["models.py", "utils.py"]
    assert any("x = 1" in c.value for c in at.code)


def test_repository_reset_requires_confirmation(monkeypatch):
    at, _ = launch(monkeypatch=monkeypatch)
    goto(at, "Repositories")
    assert at.button(key="reset_repo").disabled
    at.checkbox(key="reset_sure").check().run()
    at.button(key="reset_repo").click().run()
    assert any("reset" in s.value.lower() for s in at.success)


def test_history_page_shows_a_log(monkeypatch):
    at, _ = launch(monkeypatch=monkeypatch)
    goto(at, "History")
    assert at.selectbox(key="history_choice").options == ["20261001T000000Z_abcd1234.jsonl"]
    assert len(at.json) >= 1


def test_scripts_page_runs_a_script(monkeypatch):
    at, _ = launch(monkeypatch=monkeypatch)
    goto(at, "Scripts")
    at.button(key="run_script").click().run()
    assert not at.exception
    assert any("Latest job" in s.value for s in at.subheader)


def test_scripts_page_rejects_invalid_json(monkeypatch):
    at, _ = launch(monkeypatch=monkeypatch)
    goto(at, "Scripts")
    at.text_area(key="script_body").set_value("{not json").run()
    at.button(key="run_script").click().run()
    assert any("Invalid JSON" in e.value for e in at.error)
