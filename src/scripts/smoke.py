from coding_agent.llm import get_llm


def main() -> None:
    llm = get_llm()
    reply = llm.invoke("Reply with exactly one word: ready")
    print("Model replied:", reply.content)


if __name__ == "__main__":
    main()