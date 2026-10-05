import pytest

from coding_agent.errors import AgentError
from coding_agent.llm import clean_text, invoke_structured
from coding_agent.schemas import FileSelection
from tests.fakes import FakeLLM

SEL = FileSelection(relevant_files=["a.py"], reasoning="r")
TOOL_ERR = RuntimeError("Error code: 400 - tool_use_failed: attempted to call tool 'get_file'")


def test_strips_think_block():
    assert clean_text("<think>reasoning\nmore</think>\nFinal answer.") == "Final answer."


def test_plain_text_unchanged():
    assert clean_text("hello") == "hello"


def test_list_content():
    assert clean_text(["a", {"text": "b"}]) == "ab"


class TestInvokeStructured:
    def test_success_first_try(self):
        llm = FakeLLM(structured={FileSelection: [SEL]})
        assert invoke_structured(llm, FileSelection, ["m"]) == SEL
        assert len(llm.calls) == 1

    def test_retries_once_after_tool_call_error(self):
        llm = FakeLLM(structured={FileSelection: [TOOL_ERR, SEL]})
        assert invoke_structured(llm, FileSelection, ["m"]) == SEL
        assert len(llm.calls) == 2
        assert "no tools" in llm.calls[1][1][-1].content.lower()

    def test_gives_up_after_second_failure(self):
        llm = FakeLLM(structured={FileSelection: [TOOL_ERR, TOOL_ERR]})
        with pytest.raises(AgentError, match="valid structured answer"):
            invoke_structured(llm, FileSelection, ["m"])

    def test_unrelated_errors_are_not_swallowed(self):
        llm = FakeLLM(structured={FileSelection: [RuntimeError("network down")]})
        with pytest.raises(RuntimeError, match="network down"):
            invoke_structured(llm, FileSelection, ["m"])
        assert len(llm.calls) == 1