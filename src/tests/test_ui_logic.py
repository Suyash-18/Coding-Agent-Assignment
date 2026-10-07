"""UI building blocks that do not need Streamlit: SSE parsing, the API client, timeline text."""

import httpx
import pytest

from coding_agent.ui import client as client_mod
from coding_agent.ui.client import ApiClient, ApiUnavailable, UiApiError, parse_sse
from coding_agent.ui.timeline import describe, timeline_markdown
from tests.fake_api import RUN_ID, FakeApi


def make(handler) -> ApiClient:
    return ApiClient("http://api.test", transport=httpx.MockTransport(handler))


# ---- SSE parsing ------------------------------------------------------------
class TestParseSse:
    def test_named_event_with_type_in_data(self):
        lines = ["id: 3", "event: node_done",
                 'data: {"type": "node_done", "node": "scan_repo", "data": {"file_count": 4}}', ""]
        assert list(parse_sse(lines)) == [
            {"type": "node_done", "node": "scan_repo", "data": {"file_count": 4}, "seq": 3}]

    def test_event_name_is_used_when_data_has_no_type(self):
        lines = ["event: plan", 'data: {"plan": {"summary": "s"}}', ""]
        (event,) = parse_sse(lines)
        assert event["type"] == "plan" and event["data"] == {"plan": {"summary": "s"}}

    def test_seq_can_come_from_the_payload(self):
        (event,) = parse_sse(['data: {"type": "start", "seq": 7}', ""])
        assert event["seq"] == 7

    def test_comments_are_ignored_and_multiline_data_is_joined(self):
        lines = [": keep-alive", "", "event: x", 'data: {"a":', 'data: 1}', ""]
        (event,) = parse_sse(lines)
        assert event["data"] == {"a": 1}

    def test_non_json_data_is_kept_as_a_message(self):
        (event,) = parse_sse(["event: note", "data: hello", ""])
        assert event["type"] == "note" and event["data"] == {"message": "hello"}

    def test_a_final_frame_without_blank_line_is_still_delivered(self):
        assert len(list(parse_sse(['data: {"type": "done"}']))) == 1

    def test_several_events(self):
        lines = ['data: {"type": "a"}', "", 'data: {"type": "b"}', ""]
        assert [e["type"] for e in parse_sse(lines)] == ["a", "b"]


# ---- client -----------------------------------------------------------------
class TestClient:
    def test_create_run_sends_task_repo_and_model(self):
        fake = FakeApi()
        make(fake).create_run("Add a test", "demo", "qwen/qwen3.8-27b")
        assert fake.created == [{"task": "Add a test", "repo_id": "demo", "model": "qwen/qwen3.8-27b"}]

    def test_model_is_omitted_when_not_chosen(self):
        fake = FakeApi()
        make(fake).create_run("t", "demo")
        assert "model" not in fake.created[0]

    def test_decision_payload(self):
        fake = FakeApi()
        api = make(fake)
        api.create_run("t", "demo")
        list(api.stream_events(RUN_ID))
        api.decide(RUN_ID, "plan", "approve", "keep it small")
        assert fake.decisions == [{"stage": "plan", "action": "approve", "feedback": "keep it small"}]

    def test_backend_error_detail_becomes_a_readable_error(self):
        api = make(lambda r: httpx.Response(400, json={"detail": {"code": "input_blocked", "message": "Nope."}}))
        with pytest.raises(UiApiError) as exc:
            api.health()
        assert (exc.value.status, exc.value.code, exc.value.message) == (400, "input_blocked", "Nope.")

    def test_validation_errors_are_summarised(self):
        detail = [{"loc": ["body", "task"], "msg": "Field required", "type": "missing"}]
        api = make(lambda r: httpx.Response(422, json={"detail": detail}))
        with pytest.raises(UiApiError) as exc:
            api.health()
        assert exc.value.code == "validation_error" and "task: Field required" in exc.value.message

    def test_retry_after_is_kept(self):
        api = make(lambda r: httpx.Response(429, json={"detail": {"code": "rate_limited", "message": "Slow down"}},
                                            headers={"retry-after": "12"}))
        with pytest.raises(UiApiError) as exc:
            api.health()
        assert exc.value.retry_after == 12.0 and "retry in 12s" in str(exc.value)

    def test_unreachable_backend(self):
        def boom(request):
            raise httpx.ConnectError("refused")

        with pytest.raises(ApiUnavailable, match="Could not reach"):
            make(boom).health()

    def test_stream_events_yields_parsed_events_after_a_sequence_number(self):
        fake = FakeApi()
        api = make(fake)
        api.create_run("t", "demo")
        events = list(api.stream_events(RUN_ID, after=3))
        assert [e["seq"] for e in events] == [4, 5, 6]
        assert events[-1]["type"] == "interrupt" and events[-1]["data"]["stage"] == "plan"

    def test_stream_error_status_raises(self):
        api = make(lambda r: httpx.Response(404, json={"detail": {"code": "not_found", "message": "No run."}}))
        with pytest.raises(UiApiError, match="No run"):
            list(api.stream_events("missing"))

    def test_read_timeout_ends_the_stream_quietly(self):
        def slow(request):
            raise httpx.ReadTimeout("idle")

        assert list(make(slow).stream_events(RUN_ID)) == []

    def test_patch_returns_bytes(self):
        assert make(FakeApi()).patch(RUN_ID).startswith(b"--- a/models.py")

    def test_default_transport_is_used_when_none_is_given(self, monkeypatch):
        monkeypatch.setattr(client_mod, "DEFAULT_TRANSPORT", httpx.MockTransport(FakeApi()))
        assert ApiClient("http://x").health()["status"] == "ok"


# ---- timeline ---------------------------------------------------------------
class TestTimeline:
    @pytest.mark.parametrize("event, expected", [
        ({"type": "node_done", "node": "scan_repo", "data": {"file_count": 7}}, "Scanned repository (7 files)"),
        ({"type": "node_done", "node": "select_files", "data": {"relevant_files": ["a.py", "b.py"]}},
         "Selected files: a.py, b.py"),
        ({"type": "plan", "data": {"plan": {"steps": [{}, {}]}}}, "Plan ready (2 steps)"),
        ({"type": "test", "data": {"passed": True, "attempt": 2}}, "Tests passed (attempt 2)"),
        ({"type": "test", "data": {"passed": False, "attempt": 1, "will_retry": True}},
         "Tests failed (attempt 1), retrying with the failure output"),
        ({"type": "test", "data": {"passed": False, "attempt": 3, "no_tests": True}},
         "No tests were collected (attempt 3)"),
        ({"type": "diff", "data": {"files": ["a.py"]}}, "Diff ready (1 file)"),
        ({"type": "interrupt", "data": {"stage": "apply"}}, "Waiting for your decision (apply)"),
        ({"type": "error", "data": {"message": "boom"}}, "Error: boom"),
        ({"type": "done", "data": {"status": "applied"}}, "Finished (applied)"),
        ({"type": "node_done", "node": "plan_approval", "data": {"approved": False}}, "Plan rejected"),
        ({"type": "something_new", "node": "x", "data": {}}, "something_new: x"),
    ])
    def test_describe(self, event, expected):
        assert describe(event)[1] == expected

    def test_uninteresting_node_events_are_skipped(self):
        assert describe({"type": "node_done", "node": "mystery_node", "data": {}}) is None

    def test_markdown_for_no_events(self):
        assert "Waiting" in timeline_markdown([])

    def test_markdown_joins_lines(self):
        text = timeline_markdown([{"type": "start"}, {"type": "done", "data": {"status": "applied"}}])
        assert "Run started" in text and "Finished (applied)" in text
