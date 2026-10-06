from pathlib import Path

from coding_agent import tools
from coding_agent.tools import apply_changes, cleanup_temp, copy_to_temp, run_tests

SAMPLE = Path(__file__).resolve().parent.parent / "sample_project"

PASSING = {"test_ok.py": "def test_ok():\n    assert 1 + 1 == 2\n"}
FAILING = {"test_bad.py": "def test_bad():\n    assert 1 == 2\n"}


def test_passing_tests(tmp_path, make_repo):
    make_repo(tmp_path, PASSING)
    result = run_tests(tmp_path)
    assert result.passed and result.returncode == 0
    assert "1 passed" in result.output


def test_failing_tests(tmp_path, make_repo):
    make_repo(tmp_path, FAILING)
    result = run_tests(tmp_path)
    assert not result.passed
    assert "assert 1 == 2" in result.output


def test_timeout(tmp_path, make_repo):
    make_repo(tmp_path, {"test_slow.py": "import time\ndef test_slow():\n    time.sleep(10)\n"})
    result = run_tests(tmp_path, timeout=2)
    assert result.timed_out and not result.passed


def test_no_tests_is_not_a_pass(tmp_path, make_repo):
    make_repo(tmp_path, {"app.py": "x = 1\n"})
    assert not run_tests(tmp_path).passed


def test_secrets_not_passed_to_tests(tmp_path, make_repo, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "super-secret")
    make_repo(tmp_path, {
        "test_env.py": "import os\ndef test_env():\n    assert 'GROQ_API_KEY' not in os.environ\n"
    })
    assert run_tests(tmp_path).passed


def test_output_is_truncated(tmp_path, make_repo):
    make_repo(tmp_path, {"test_big.py": "def test_big():\n    print('x' * 20000)\n    assert False\n"})
    result = run_tests(tmp_path)
    assert len(result.output) <= tools.MAX_TEST_OUTPUT_CHARS + 20


def test_sample_project_passes_in_temp_copy():
    copy = copy_to_temp(SAMPLE)
    try:
        result = run_tests(copy)
        assert result.passed, result.output
        assert "7 passed" in result.output
    finally:
        cleanup_temp(copy)

def test_temp_copy_runs_its_own_code_not_the_original():
    """Fails if tests import the ORIGINAL sample project instead of the temp copy."""
    work = copy_to_temp(SAMPLE)
    try:
        assert run_tests(work).passed
        original = (work / "models.py").read_text(encoding="utf-8")
        apply_changes(work, {"models.py": original + "\nraise RuntimeError('canary')\n"})
        assert not run_tests(work).passed
    finally:
        cleanup_temp(work)


def test_package_style_imports_resolve_to_the_copy(tmp_path, make_repo, monkeypatch):
    """Regression: `from <folder>.x import ...` must hit the copy, even if the original is importable."""
    repo = make_repo(tmp_path / "demo_pkg", {
        "calc.py": "def add(a, b):\n    return a + b\n",
        "test_calc.py": "from demo_pkg.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    })
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))  # original importable, like an editable install
    work = copy_to_temp(repo)
    try:
        assert run_tests(work).passed
        apply_changes(work, {"calc.py": "def add(a, b):\n    return a - b\n"})
        assert not run_tests(work).passed  # a broken change must be detected
    finally:
        cleanup_temp(work)


def test_cleanup_removes_the_whole_container(tmp_path, make_repo):
    repo = make_repo(tmp_path / "demo_pkg", {"a.py": "x = 1\n"})
    work = copy_to_temp(repo)
    container = work.parent
    cleanup_temp(work)
    assert not container.exists()