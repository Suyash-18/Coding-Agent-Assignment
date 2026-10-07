"""Thin, defensive bridges to the guardrails module (works even if a helper is missing)."""

from __future__ import annotations

import json
import logging
from typing import Any

from coding_agent.api.errors import ApiError
from coding_agent.errors import GuardrailViolation

log = logging.getLogger(__name__)


def precheck_text(text: str, field: str = "task") -> None:
    """Run the agent's own input guardrail early so bad requests never start a run."""
    try:
        from coding_agent.guardrails import validate_task
    except ImportError:
        return
    try:
        validate_task(text)
    except GuardrailViolation as exc:
        raise ApiError(
            422, "guardrail_blocked", str(exc), field=field, rule=getattr(exc, "rule", None)
        ) from exc
    except Exception:  # an unexpected guardrail bug must not take the API down
        log.exception("input guardrail raised unexpectedly")


def redact_text(text: str) -> str:
    try:
        from coding_agent.guardrails import redact
    except ImportError:
        return text
    try:
        out = redact(text)
        return out if isinstance(out, str) else text
    except Exception:
        return text


def redact_obj(obj: Any) -> Any:
    """Redact secrets inside a JSON-able object; withhold it if redaction breaks the JSON."""
    try:
        from coding_agent.guardrails import redact
    except ImportError:
        return obj
    try:
        text = json.dumps(obj, default=str)
        out = redact(text)
        if not isinstance(out, str) or out == text:
            return obj
        return json.loads(out)
    except json.JSONDecodeError:
        return {"withheld": True, "reason": "redaction failed"}
    except Exception:
        return obj
