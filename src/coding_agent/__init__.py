def main() -> None:
    """Console-script entry point (`coding-agent`). Imported lazily to keep `import coding_agent` light."""
    from coding_agent.cli import main as cli_main

    cli_main()