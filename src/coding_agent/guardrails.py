"""Deterministic guardrails: pure Python, no LLM calls.

Input guard   -> validate_task()
File guard    -> validate_file_access()
Output guard  -> validate_change_set(), validate_no_secrets()
Budget guard  -> validate_budget()

Every violation raises GuardrailViolation (or PathViolation for bad paths) with a
message that is safe to show to the user. Messages never echo a detected secret.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path

from coding_agent.errors import GuardrailViolation, PathViolation
from coding_agent.tools import is_protected, safe_path

# ---- limits ---------------------------------------------------------------
MAX_TASK_CHARS = 2_000
MAX_CHANGED_FILES = 10
MAX_FILE_CHARS = 60_000
MAX_TOTAL_CHARS = 150_000

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # keeps \t \n \r


# ---- input rules ----------------------------------------------------------
# Matched against the task after NFKC normalisation, casefolding and whitespace
# collapsing, so patterns are lowercase and fullwidth/obfuscated text is caught.
_INPUT_RULES: list[tuple[str, list[str]]] = [
    ("prompt injection", [
        r"\b(?:ignore|disregard|forget)\s+(?:(?:all|any|the|your|my|these|those)\s+)*"
        r"(?:(?:previous|prior|above|earlier|system|developer|original)\s+)*"
        r"(?:instructions?|prompts?|directions|guidelines)\b",
        r"\b(?:ignore|disregard|forget)\s+(?:(?:all|any|the|your|my)\s+)*"
        r"(?:previous|prior|above|earlier|system|developer|original)\s+rules\b",
        r"\b(?:reveal|show|print|display|repeat|output|leak)\b[^.\n]{0,20}"
        r"\b(?:system|hidden|initial)\s+(?:prompt|instructions?|message)\b",
        r"\b(?:jailbreak|developer\s+mode|do\s+anything\s+now)\b",
        r"\bpretend\s+(?:to\s+be|you\s+are|you're)\b",
        r"\b(?:bypass|disable|circumvent|turn\s+off)\s+(?:(?:the|all|any|your)\s+)*"
        r"(?:guardrails?|safety|safeguards?|approval|sandbox|restrictions)\b",
        r"<\|[a-z_]+\|>|</?(?:system|assistant)>|\[/?inst\]",
    ]),
    ("secret extraction", [
        r"\b(?:print|cat|echo|dump|leak|exfiltrate|reveal|expose|send|(?:show|tell|give)\s+me)"
        r"(?:\s+(?:out|the|my|your|its|all|any|every|a|an|contents?\s+of))*\s+"
        r"(?:(?<!\w)\.env\b|api[\s_-]?keys?\b|secret\s+(?:keys?|tokens?)\b|secrets\b|"
        r"private\s+keys?\b|credentials?\b|(?:access|auth)\s+tokens?\b)",
        r"\$\{?\w*(?:api_key|secret|token|password)\w*",
        r"\bprintenv\b",
    ]),
    ("path escape", [
        r"(?<![\w.])\.\.[/\\]",
        r"/etc/(?:passwd|shadow)\b|~/\.ssh|\.ssh[/\\]|\bid_rsa\b|\bid_ed25519\b|c:\\windows\\system32",
    ]),
    ("destructive request", [
        r"\b(?:delete|remove|wipe|erase|destroy|purge|nuke)\s+(?:absolutely\s+)?"
        r"(?:everything|all\s+(?:(?:the|of\s+the|my)\s+)?"
        r"(?:files|code|tests?|data|source|folders|directories)|"
        r"(?:the\s+)?(?:whole|entire)\s+(?:repo(?:sitory)?|project|codebase|directory))\b",
        r"\brm\s+-[a-z]*r[a-z]*\b",
        r"\bformat\s+(?:the\s+)?(?:disk|drive|hard\s*drive)\b",
        r"\bgit\s+(?:reset\s+--hard|clean\s+-[a-z]*f|push\s+(?:--force|-f))\b",
        r"\bdel\s+/[sfq]\b",
    ]),
    ("test tampering", [
        r"\b(?:delete|remove|disable|skip|comment\s+out)\s+(?:(?:all|the|any|these|those|failing)\s+)*"
        r"tests?\s+(?:so|to|until)\b",
    ]),
    ("non-coding request", [
        r"\b(?:write|compose|tell)\s+me\s+(?:a|an)\s+(?:poem|story|joke|essay|song|haiku)\b",
        r"\bwhat(?:'s|\s+is)\s+the\s+weather\b",
        r"\bgive\s+me\s+(?:a\s+)?recipe\b",
    ]),
]
_COMPILED_INPUT_RULES = [
    (label, [re.compile(p) for p in patterns]) for label, patterns in _INPUT_RULES
]


def _canonical(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text)


def validate_task(task: str) -> str:
    """Return the cleaned task, or raise GuardrailViolation."""
    if not isinstance(task, str):
        raise GuardrailViolation("Blocked by input guard (invalid task): the task must be text.")
    cleaned = _CONTROL_CHARS.sub("", task).strip()
    if not cleaned:
        raise GuardrailViolation(
            "Blocked by input guard (empty task): describe the code change you want."
        )
    if len(cleaned) > MAX_TASK_CHARS:
        raise GuardrailViolation(
            f"Blocked by input guard (task too long): {len(cleaned):,} characters, "
            f"limit {MAX_TASK_CHARS:,}. Shorten the task."
        )
    text = _canonical(cleaned)
    for label, patterns in _COMPILED_INPUT_RULES:
        if any(p.search(text) for p in patterns):
            raise GuardrailViolation(
                f"Blocked by input guard ({label}). "
                "Rephrase the task as a code change inside the repository."
            )
    return cleaned


# ---- file access ----------------------------------------------------------
def validate_file_access(repo: str | Path, rel_path: str) -> Path:
    """Resolve rel_path inside repo, or raise PathViolation. Returns the absolute path."""
    root = Path(repo).resolve()
    target = safe_path(root, rel_path)  # traversal, absolute paths, symlink escapes
    if target == root:
        raise PathViolation(f"Not a file path: {rel_path}")
    if is_protected(target.relative_to(root)):
        raise PathViolation(f"The model tried to change a protected file: {rel_path}")
    return target


# ---- output: secrets ------------------------------------------------------
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Groq API key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("API key (sk- style)", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
]
_SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _env_secret_values(min_len: int = 16) -> list[str]:
    return [
        v for k, v in os.environ.items()
        if len(v) >= min_len and any(m in k.upper() for m in _SECRET_ENV_MARKERS)
    ]


def _matches(pattern: re.Pattern[str], text: str) -> set[str]:
    return {m.group(0) for m in pattern.finditer(text)}


def validate_no_secrets(
    path: str,
    new_content: str,
    old_content: str = "",
    env_secrets: Sequence[str] | None = None,
) -> None:
    """Reject secrets the change *introduces* (ones already in the file are ignored)."""
    for label, pattern in _SECRET_PATTERNS:
        if _matches(pattern, new_content) - _matches(pattern, old_content):
            raise GuardrailViolation(
                f"Blocked by output guard (possible {label} in {path}). "
                "The proposed change was discarded; nothing was written."
            )
    values = _env_secret_values() if env_secrets is None else env_secrets
    for value in values:
        if value in new_content and value not in old_content:
            raise GuardrailViolation(
                f"Blocked by output guard (a secret from the environment appears in {path}). "
                "The proposed change was discarded; nothing was written."
            )


def redact(text: str, env_secrets: Sequence[str] | None = None) -> str:
    """Replace anything that looks like a secret with [REDACTED] (used for run logs)."""
    for _, pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    values = _env_secret_values() if env_secrets is None else env_secrets
    for value in sorted(values, key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return text


# ---- output: change set ---------------------------------------------------
def validate_change_set(
    repo: str | Path,
    changes: Mapping[str, str],
    existing: Mapping[str, str] | None = None,
) -> None:
    """Validate the model's proposed {relative_path: new_content} before any test run."""
    existing = existing or {}
    if not changes:
        raise GuardrailViolation("Blocked by output guard (no changes were proposed).")
    if len(changes) > MAX_CHANGED_FILES:
        raise GuardrailViolation(
            f"Blocked by output guard (too many files): {len(changes)} changed, "
            f"limit {MAX_CHANGED_FILES}. Try a narrower task."
        )
    total = 0
    for rel, content in changes.items():
        validate_file_access(repo, rel)
        if not isinstance(content, str):
            raise GuardrailViolation(f"Blocked by output guard (content for {rel} is not text).")
        if "\0" in content:
            raise GuardrailViolation(f"Blocked by output guard (binary content in {rel}).")
        if len(content) > MAX_FILE_CHARS:
            raise GuardrailViolation(
                f"Blocked by output guard ({rel} is {len(content):,} characters, "
                f"limit {MAX_FILE_CHARS:,})."
            )
        total += len(content)
        validate_no_secrets(rel, content, existing.get(rel, ""))
    if total > MAX_TOTAL_CHARS:
        raise GuardrailViolation(
            f"Blocked by output guard (change is {total:,} characters in total, "
            f"limit {MAX_TOTAL_CHARS:,}). Try a narrower task."
        )


# ---- budget ---------------------------------------------------------------
def validate_budget(attempts: int, max_retries: int) -> None:
    """Defence in depth: never generate again once the retry budget is spent."""
    if attempts > max_retries:
        raise GuardrailViolation(
            f"Blocked by budget guard: {attempts} test runs already used "
            f"(limit {max_retries + 1}). Stopping to avoid wasting LLM calls."
        )