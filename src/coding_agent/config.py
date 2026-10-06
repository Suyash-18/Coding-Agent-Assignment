import os
from pathlib import Path

from dotenv import load_dotenv

from coding_agent.errors import ConfigError  # noqa: F401  (re-exported for older imports)

load_dotenv()

DEFAULT_MODEL = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")


def _number_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


LLM_TIMEOUT_SECONDS = _number_env("LLM_TIMEOUT_SECONDS", 60.0)
MAX_PROMPT_CHARS = int(_number_env("MAX_PROMPT_CHARS", 30_000))  # ~7.5k tokens


def runs_dir() -> Path | None:
    """Where per-run JSON logs go (read at call time). AGENT_RUNS_DIR=off disables them."""
    raw = os.getenv("AGENT_RUNS_DIR", "runs").strip()
    return None if raw.lower() in {"", "off", "none", "0"} else Path(raw)


def get_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ConfigError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
        )
    return key