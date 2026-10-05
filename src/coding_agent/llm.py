from langchain_groq import ChatGroq

from coding_agent.config import DEFAULT_MODEL, get_api_key


def get_llm(model_name: str | None = None, temperature: float = 0.0) -> ChatGroq:
    """The single place where LLM clients are created."""
    return ChatGroq(
        model=model_name or DEFAULT_MODEL,
        temperature=temperature,
        api_key=get_api_key(),
    )