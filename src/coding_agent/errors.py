class AgentError(Exception):
    """Base class for all agent errors."""


class ToolError(AgentError):
    """A tool could not do what was asked (missing file, too large, etc.)."""


class PathViolation(ToolError):
    """A path tried to escape the repo or touch a protected file."""


class GuardrailViolation(AgentError):
    """A guardrail blocked the request or the model output."""