from coding_agent.llm import clean_text


def test_strips_think_block():
    assert clean_text("<think>reasoning\nmore</think>\nFinal answer.") == "Final answer."


def test_plain_text_unchanged():
    assert clean_text("hello") == "hello"


def test_list_content():
    assert clean_text(["a", {"text": "b"}]) == "ab"