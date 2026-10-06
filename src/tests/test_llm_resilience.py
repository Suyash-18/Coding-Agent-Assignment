"""Backoff, error translation, size limits and the error taxonomy (no real waiting)."""

import logging

import groq
import httpx
import pytest

from coding_agent import config, llm
from coding_agent.errors import (
    AgentError,
    ConfigError,
    GuardrailViolation,
    LLMError,
    LLMTimeoutError,
    PathViolation,
    PromptTooLargeError,
    RateLimitError,
    ToolError,
)
from coding_agent.nodes import AgentNodes
from coding_agent.schemas import ChangeSet, FileSelection, Plan
from tests.fakes import FakeLLM
from tests.test_graph import GOOD, real_plan, start, types

SEL = FileSelection(relevant_files=["models.py", "tests/test_users.py"], reasoning="r")
REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def status_error(cls, status, message="boom", headers=None):
    resp = httpx.Response(status, request=REQ, headers=headers or {})
    return cls(message, response=resp, body=None)


def rate_limit(message="Rate limit reached", retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else None
    return status_error(groq.RateLimitError, 429, message, headers)


@pytest.fixture
def waits(monkeypatch):
    """Record sleeps instead of sleeping; make jitter deterministic."""
    recorded: list[float] = []
    monkeypatch.setattr(llm, "_sleep", recorded.append)
    monkeypatch.setattr(llm, "_jitter", lambda a, b: 0.0)
    return recorded


def call(queue):
    fake = FakeLLM(structured={FileSelection: queue})
    return fake, lambda: llm.invoke_structured(fake, FileSelection, ["m"])


# ---- rate limits ----------------------------------------------------------
class TestRateLimits:
    def test_waits_for_retry_after_header_then_succeeds(self, waits):
        fake, run = call([rate_limit(retry_after=7), SEL])
        assert run() == SEL
        assert waits == [7.0] and len(fake.calls) == 2

    def test_parses_try_again_in_from_the_message(self, waits):
        fake, run = call([rate_limit("Limit 8000. Please try again in 6.5s."), SEL])
        assert run() == SEL
        assert waits == [6.5]

    def test_long_wait_fails_fast_without_sleeping(self, waits):
        fake, run = call([rate_limit(retry_after=3600), SEL])
        with pytest.raises(RateLimitError, match="Try again in about 60 min"):
            run()
        assert waits == [] and len(fake.calls) == 1

    def test_gives_up_after_max_attempts_with_exponential_waits(self, waits):
        fake, run = call([rate_limit()] * llm.MAX_LLM_ATTEMPTS)
        with pytest.raises(RateLimitError, match="rate limit"):
            run()
        assert waits == [2.0, 4.0, 8.0]
        assert len(fake.calls) == llm.MAX_LLM_ATTEMPTS

    def test_backoff_is_capped(self, waits, monkeypatch):
        monkeypatch.setattr(llm, "MAX_DELAY", 5.0)
        fake, run = call([rate_limit()] * llm.MAX_LLM_ATTEMPTS)
        with pytest.raises(RateLimitError):
            run()
        assert waits == [2.0, 4.0, 5.0]

    def test_retry_is_announced_in_the_log(self, waits, caplog):
        _, run = call([rate_limit(retry_after=3), SEL])
        with caplog.at_level(logging.WARNING, logger="coding_agent.llm"):
            run()
        assert "rate limited" in caplog.text and "attempt 2 of 4" in caplog.text


# ---- timeouts and service errors ------------------------------------------
class TestTransientFailures:
    def test_timeout_then_success(self, waits):
        fake, run = call([groq.APITimeoutError(request=REQ), SEL])
        assert run() == SEL and len(fake.calls) == 2

    def test_repeated_timeouts_become_llm_timeout_error(self, waits):
        _, run = call([groq.APITimeoutError(request=REQ)] * llm.MAX_LLM_ATTEMPTS)
        with pytest.raises(LLMTimeoutError, match="did not answer within"):
            run()

    def test_server_error_is_retried(self, waits):
        fake, run = call([status_error(groq.InternalServerError, 503), SEL])
        assert run() == SEL and len(fake.calls) == 2

    def test_connection_failure_is_retried_then_reported(self, waits):
        _, run = call([groq.APIConnectionError(request=REQ)] * llm.MAX_LLM_ATTEMPTS)
        with pytest.raises(LLMError, match="could not be reached"):
            run()


# ---- non-retryable --------------------------------------------------------
class TestNonRetryable:
    @pytest.mark.parametrize("status, cls", [(401, groq.AuthenticationError),
                                             (403, groq.PermissionDeniedError)])
    def test_bad_key_is_a_config_error(self, waits, status, cls):
        fake, run = call([status_error(cls, status), SEL])
        with pytest.raises(ConfigError, match="GROQ_API_KEY"):
            run()
        assert len(fake.calls) == 1 and waits == []

    def test_413_is_prompt_too_large(self, waits):
        fake, run = call([status_error(groq.APIStatusError, 413), SEL])
        with pytest.raises(PromptTooLargeError):
            run()
        assert len(fake.calls) == 1

    def test_unrelated_bad_request_is_re_raised_untouched(self, waits):
        err = status_error(groq.BadRequestError, 400, "invalid model")
        fake, run = call([err, SEL])
        with pytest.raises(groq.BadRequestError):
            run()
        assert len(fake.calls) == 1

    def test_backoff_and_structured_retry_work_together(self, waits):
        tool_err = RuntimeError("Error code: 400 - tool_use_failed: tried a tool")
        fake, run = call([rate_limit(retry_after=1), tool_err, SEL])
        assert run() == SEL
        assert len(fake.calls) == 3  # rate limit retried, then one corrective retry


# ---- prompt size ----------------------------------------------------------
class TestPromptSize:
    def test_oversized_prompt_is_refused_before_any_call(self, monkeypatch):
        monkeypatch.setattr(llm, "MAX_PROMPT_CHARS", 50)
        fake = FakeLLM(structured={FileSelection: [SEL]})
        with pytest.raises(PromptTooLargeError, match="limit is 50"):
            llm.invoke_structured(fake, FileSelection, ["x" * 60])
        assert fake.calls == []

    def test_invoke_text_cleans_and_retries(self, waits):
        fake = FakeLLM(text=[rate_limit(retry_after=1), "<think>hm</think>Done."])
        assert llm.invoke_text(fake, ["m"]) == "Done."
        assert len(fake.calls) == 2

    def test_invoke_text_checks_size(self, monkeypatch):
        monkeypatch.setattr(llm, "MAX_PROMPT_CHARS", 5)
        with pytest.raises(PromptTooLargeError):
            llm.invoke_text(FakeLLM(text=["x"]), ["too long for the limit"])


# ---- explain degrades gracefully ------------------------------------------
class TestExplainFallback:
    STATE = {"task": "t", "plan": {"summary": "Add validation.", "steps": [], "assumptions": []},
             "diff": "d"}

    def test_rate_limited_explain_falls_back_to_the_plan_summary(self, waits):
        fake = FakeLLM(text=[rate_limit()] * llm.MAX_LLM_ATTEMPTS)
        out = AgentNodes(fake).explain(self.STATE)
        assert out["explanation"].startswith("Add validation.")
        assert "unavailable" in out["explanation"] and out["status"] == "proposed"

    def test_unexpected_errors_still_propagate(self):
        with pytest.raises(RuntimeError, match="bug"):
            AgentNodes(FakeLLM(text=[RuntimeError("bug")])).explain(self.STATE)


# ---- configuration --------------------------------------------------------
def test_get_llm_disables_sdk_retries_and_sets_a_timeout(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-value-not-a-real-credential")
    client = llm.get_llm("some-model")
    assert client.max_retries == 0
    assert client.request_timeout == config.LLM_TIMEOUT_SECONDS


@pytest.mark.parametrize("raw, expected", [("12", 12.0), ("oops", 60.0), (None, 60.0)])
def test_number_env_falls_back_on_garbage(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", raw)
    assert config._number_env("LLM_TIMEOUT_SECONDS", 60.0) == expected


# ---- taxonomy -------------------------------------------------------------
class TestTaxonomy:
    @pytest.mark.parametrize("cls", [ConfigError, ToolError, PathViolation, GuardrailViolation,
                                     LLMError, RateLimitError, LLMTimeoutError, PromptTooLargeError])
    def test_everything_is_an_agent_error(self, cls):
        assert issubclass(cls, AgentError)

    def test_hierarchy(self):
        assert issubclass(PathViolation, ToolError)
        assert all(issubclass(c, LLMError)
                   for c in (RateLimitError, LLMTimeoutError, PromptTooLargeError))

    def test_config_error_is_importable_from_config_too(self):
        assert config.ConfigError is ConfigError

    @pytest.mark.parametrize("cls", [ConfigError, LLMError, RateLimitError,
                                     LLMTimeoutError, PromptTooLargeError])
    def test_user_facing_errors_have_a_hint(self, cls):
        assert cls.hint


# ---- through the runner ---------------------------------------------------
class TestThroughTheRunner:
    def test_rate_limit_that_clears_does_not_change_the_event_stream(
        self, sample_repo, thread_id, waits
    ):
        fake = FakeLLM(
            structured={FileSelection: [rate_limit(retry_after=1), SEL],
                        Plan: [real_plan()], ChangeSet: [GOOD]},
            text=["ok"],
        )
        events = start(sample_repo, thread_id, fake)
        assert types(events) == ["start", "node_done", "node_done", "node_done", "plan", "interrupt"]
        assert waits == [1.0]

    def test_exhausted_rate_limit_surfaces_kind_and_hint(self, sample_repo, thread_id, waits):
        fake = FakeLLM(structured={FileSelection: [rate_limit()] * llm.MAX_LLM_ATTEMPTS})
        events = start(sample_repo, thread_id, fake)
        assert types(events) == ["start", "node_done", "error"]
        data = events[-1].data
        assert data["kind"] == "RateLimitError"
        assert "Groq" in data["hint"] and "rate limit" in data["message"].lower()

    def test_unexpected_error_has_no_hint(self, sample_repo, thread_id):
        fake = FakeLLM(structured={FileSelection: [RuntimeError("boom")]})
        data = start(sample_repo, thread_id, fake)[-1].data
        assert data["kind"] == "RuntimeError" and data["hint"] == ""