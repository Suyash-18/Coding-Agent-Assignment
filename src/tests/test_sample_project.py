from pathlib import Path

SAMPLE = Path(__file__).resolve().parent.parent / "src/sample_project"


def test_sample_project_uses_flat_imports():
    """The agent tests a temp copy with a different folder name, so imports must be flat."""
    offenders = [p.name for p in SAMPLE.rglob("*.py") if "sample_project." in p.read_text(encoding="utf-8")]
    assert offenders == []