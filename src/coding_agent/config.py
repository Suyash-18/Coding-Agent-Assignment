import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")


class ConfigError(Exception):
    """Raised when required configuration is missing."""


def get_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ConfigError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
        )
    return key