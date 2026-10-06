"""Error taxonomy. Every failure the agent can explain to a user is an AgentError.

AgentError
├── ConfigError          missing/invalid configuration (e.g. no API key)
├── ToolError            a filesystem/test tool could not do its job
│   └── PathViolation    a path escaped the repo or hit a protected file
├── GuardrailViolation   a guardrail blocked the request or the model output
└── LLMError             the model service failed
    ├── RateLimitError       Groq rate/usage limit (per minute or per day)
    ├── LLMTimeoutError      no answer within the timeout
    └── PromptTooLargeError  the request does not fit the model's limits

`hint` is a short, actionable next step shown next to the message.
"""


class AgentError(Exception):
    """Base class for all agent errors."""

    hint: str = ""


class ConfigError(AgentError):
    """Required configuration is missing or invalid."""

    hint = "Copy .env.example to .env and set GROQ_API_KEY."


class ToolError(AgentError):
    """A tool could not do what was asked (missing file, too large, etc.)."""


class PathViolation(ToolError):
    """A path tried to escape the repo or touch a protected file."""


class GuardrailViolation(AgentError):
    """A guardrail blocked the request or the model output."""


class LLMError(AgentError):
    """The model service failed."""

    hint = "This is usually temporary. Try again in a moment."


class RateLimitError(LLMError):
    """Groq's rate or usage limit was hit."""

    hint = (
        "Groq's free tier limits tokens and requests per minute and per day. "
        "Wait and retry, or pick another model with --model."
    )


class LLMTimeoutError(LLMError):
    """The model did not answer in time."""

    hint = "Retry, or narrow the task so the model has less to produce."


class PromptTooLargeError(LLMError):
    """The request is larger than the model or our budget allows."""

    hint = "Narrow the task, or point --repo at a smaller folder."