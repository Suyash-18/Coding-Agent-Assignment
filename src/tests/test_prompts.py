"""Prompt-size control for retries."""

from coding_agent import prompts

PLAN = {"summary": "s", "steps": [], "assumptions": []}


def human(messages) -> str:
    return messages[-1].content


class TestClipTail:
    def test_short_text_is_unchanged(self):
        assert prompts.clip_tail("abc", 10) == "abc"

    def test_long_text_keeps_the_end_and_says_so(self):
        out = prompts.clip_tail("x" * 100 + "THE END", 20)
        assert out.endswith("THE END") and "earlier output omitted" in out
        assert len(out) < 100


class TestGenerateMessages:
    def retry(self, **kw):
        args = dict(
            task="t", plan=PLAN,
            contents={"a.py": "ORIGINAL_A", "b.py": "ORIGINAL_B"},
            previous={"a.py": "PREVIOUS_A"}, test_output="FAILED test_a",
        )
        return prompts.generate_messages(**{**args, **kw})

    def test_failure_output_is_clipped_to_its_tail(self):
        out = human(self.retry(test_output="x" * 10_000 + "LAST LINE"))
        assert "LAST LINE" in out and "earlier output omitted" in out
        assert len(out) < 8_000

    def test_within_budget_both_versions_of_a_changed_file_are_sent(self):
        out = human(self.retry())
        assert "ORIGINAL_A" in out and "PREVIOUS_A" in out and "ORIGINAL_B" in out

    def test_over_budget_drops_the_original_of_changed_files_only(self, monkeypatch):
        monkeypatch.setattr(prompts, "MAX_PROMPT_CHARS", 1)
        out = human(self.retry())
        assert "ORIGINAL_A" not in out and "superseded" in out
        assert "PREVIOUS_A" in out and "ORIGINAL_B" in out

    def test_first_attempt_is_never_slimmed(self, monkeypatch):
        monkeypatch.setattr(prompts, "MAX_PROMPT_CHARS", 1)
        out = human(self.retry(previous=None, test_output=""))
        assert "ORIGINAL_A" in out and "superseded" not in out