from pathlib import Path

from coding_agent.tools import build_diff, cleanup_temp, copy_to_temp, list_files, read_file, run_tests

SAMPLE = Path("src/sample_project")


def main() -> None:
    print("Files the agent can see:")
    for f in list_files(SAMPLE):
        print(f"  {f.path} ({f.size} bytes)")

    work = copy_to_temp(SAMPLE)
    try:
        old = read_file(work, "utils.py")
        new = old + "\n\ndef is_blank(text: str) -> bool:\n    return not text.strip()\n"
        print("\nDiff:")
        print(build_diff(old, new, "utils.py"))
        print("Tests passed:", run_tests(work).passed)
    finally:
        cleanup_temp(work)


if __name__ == "__main__":
    main()