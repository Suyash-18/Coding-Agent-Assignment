"""Run the API:  python -m coding_agent.api"""

import os


def main() -> None:
    import uvicorn

    uvicorn.run(
        "coding_agent.api.app:create_app",
        factory=True,
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
