from pathlib import Path

from coding_agent.tools import run_tests

DEMO = Path(__file__).resolve().parents[2] / "demo" / "sample_project"

result = run_tests(DEMO)
print(result.output)
print("PASSED" if result.passed else "FAILED")