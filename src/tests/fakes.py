from types import SimpleNamespace


class _FakeStructured:
    def __init__(self, queue, log, name):
        self._queue, self._log, self._name = queue, log, name

    def invoke(self, messages):
        self._log.append((self._name, messages))
        if not self._queue:
            raise AssertionError(f"FakeLLM has no response left for {self._name}")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeLLM:
    """Deterministic stand-in for a chat model.

    structured: {SchemaClass: [response_or_exception, ...]}
    text: [str, ...] for plain .invoke calls
    """

    def __init__(self, structured=None, text=None):
        self._structured = {k: list(v) for k, v in (structured or {}).items()}
        self._text = list(text or [])
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        queue = self._structured.setdefault(schema, [])
        return _FakeStructured(queue, self.calls, schema.__name__)

    def invoke(self, messages):
        self.calls.append(("text", messages))
        if not self._text:
            raise AssertionError("FakeLLM has no text response left")
        item = self._text.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(content=item)