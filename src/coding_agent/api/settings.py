from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from coding_agent.config import DEFAULT_MODEL


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _csv(name: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in os.getenv(name, "").split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    root: Path
    default_model: str
    allowed_models: tuple[str, ...]
    runs_dir: Path | None
    max_runs_per_hour: int = 20
    max_runs_per_day: int = 100
    decision_ttl_s: int = 1800
    max_runs_kept: int = 50
    max_task_chars: int = 2000
    script_timeout_s: int = 180
    heartbeat_s: float = 15.0
    cors_origins: tuple[str, ...] = ()

    @property
    def sample_path(self) -> Path:
        return self.root / "src" / "sample_project"

    @property
    def demo_path(self) -> Path:
        # The folder must keep the name "sample_project" (see test isolation rules in PLAN.md).
        return self.root / "demo" / "sample_project"

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        root = Path(root or os.getenv("AGENT_ROOT") or Path(__file__).resolve().parents[3]).resolve()
        default = os.getenv("MODEL_NAME") or DEFAULT_MODEL
        allowed = _csv("ALLOWED_MODELS") or (default,)
        if default not in allowed:
            allowed = (default, *allowed)
        raw_runs = os.getenv("AGENT_RUNS_DIR", "runs")
        runs_dir = None if raw_runs.strip().lower() == "off" else (root / raw_runs)
        return cls(
            root=root,
            default_model=default,
            allowed_models=allowed,
            runs_dir=runs_dir,
            max_runs_per_hour=_int("API_MAX_RUNS_PER_HOUR", 20),
            max_runs_per_day=_int("API_MAX_RUNS_PER_DAY", 100),
            decision_ttl_s=_int("API_DECISION_TTL_S", 1800),
            cors_origins=_csv("API_CORS_ORIGINS"),
        )
