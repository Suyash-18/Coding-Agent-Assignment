"""Command-line interface for the coding agent.

A thin consumer of the event stream: it holds no agent logic. Everything comes from
stream_agent() / resume_agent(), so the CLI, backend and UI share one contract.

Exit codes
    0  finished (changes applied, dry run, or you declined/rejected on purpose)
    1  error: guardrail violation, bad input, configuration or runtime failure
    2  tests did not pass, so the proposed change is unverified
  130  interrupted with Ctrl-C (no files are written before the apply approval)
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from coding_agent.runner import AgentEvent, resume_agent, stream_agent

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_UNVERIFIED = 2
EXIT_INTERRUPTED = 130

MAX_DIFF_LINES = 400
MAX_TEST_OUTPUT_LINES = 20

app = typer.Typer(
    help="AI coding agent: plan, change, test and (with your approval) apply code changes.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """AI coding agent."""


# ---- rendering ------------------------------------------------------------
class Renderer:
    """Turns AgentEvents into terminal output. Every dynamic string is escaped."""

    def __init__(self, console: Console, dry_run: bool = False):
        self.console = console
        self.dry_run = dry_run
        self.last_test: dict[str, Any] | None = None
        utf = (console.encoding or "").lower().startswith("utf")
        self.ok, self.bad, self.arrow = ("✔", "✘", "→") if utf else ("OK", "FAIL", "->")

    # -- small helpers
    def info(self, text: str) -> None:
        self.console.print(f"[dim]{escape(text)}[/dim]")

    def good(self, text: str) -> None:
        self.console.print(f"[green]{self.ok}[/green] {escape(text)}")

    def warn(self, text: str) -> None:
        self.console.print(f"[yellow]![/yellow] {escape(text)}")

    def header(self, task: str, repo: Path, model: str | None, thread_id: str) -> None:
        body = (
            f"[bold]Task[/bold]   {escape(task)}\n"
            f"[bold]Repo[/bold]   {escape(str(repo))}\n"
            f"[bold]Model[/bold]  {escape(model or 'default')}\n"
            f"[dim]run {thread_id}[/dim]"
        )
        self.console.print(Panel(body, title="Coding Agent", border_style="cyan"))

    # -- event dispatch
    def show(self, ev: AgentEvent) -> None:
        handler = getattr(self, f"_on_{ev.type}", None)
        if handler:
            handler(ev)

    def _on_node_done(self, ev: AgentEvent) -> None:
        d, node = ev.data, ev.node
        if node == "scan_repo":
            self.good(f"Scanned repository ({d.get('file_count', '?')} files)")
        elif node == "select_files":
            files = ", ".join(d.get("relevant_files", []))
            self.good(f"Selected files: {files}")
            if d.get("reason"):
                self.info(f"  {d['reason']}")
        elif node == "read_files":
            self.good(f"Read {len(d.get('files_read', []))} file(s)")
        elif node == "make_plan" and d.get("needs_files"):
            self.warn(f"Planner asked for more files: {', '.join(d['needs_files'])}")
        elif node == "expand_files":
            self.info("Added the requested files to the context")
        elif node == "generate_changes":
            self.good(f"Generated changes for: {', '.join(d.get('files', []))}")
        elif node == "explain":
            self.console.print(
                Panel(escape(d.get("explanation", "")), title="Explanation", border_style="blue")
            )
        elif node == "apply_changes":
            self.good(f"Applied {len(d.get('files', []))} file(s)")

    def _on_plan(self, ev: AgentEvent) -> None:
        plan = ev.data["plan"]
        table = Table(show_header=True, header_style="bold", expand=True)
        table.add_column("File", overflow="fold")
        table.add_column("Action", overflow="fold")
        for step in plan.get("steps", []):
            table.add_row(escape(step["file"]), escape(step["action"]))
        parts: list[Any] = [escape(plan.get("summary", "")), "", table]
        for note in plan.get("assumptions", []):
            parts.append(f"[dim]assumption:[/dim] {escape(note)}")
        self.console.print(Panel(Group(*parts), title="Plan", border_style="magenta"))

    def _on_test(self, ev: AgentEvent) -> None:
        d = ev.data
        self.last_test = d
        attempt = d.get("attempt", "?")
        if d.get("passed"):
            self.good(f"Tests passed (attempt {attempt})")
            return
        if d.get("no_tests"):
            self.warn(f"No tests were collected (attempt {attempt}): the change cannot be verified")
            return
        self.console.print(f"[red]{self.bad}[/red] Tests failed (attempt {attempt})")
        lines = str(d.get("output", "")).strip().splitlines()[-MAX_TEST_OUTPUT_LINES:]
        if lines:
            self.console.print(
                Panel("\n".join(escape(line) for line in lines), title="Test output (tail)",
                      border_style="red", style="dim")
            )
        if d.get("will_retry"):
            self.warn("Retrying: the failure output is being sent back to the model")

    def _on_diff(self, ev: AgentEvent) -> None:
        diff = ev.data.get("diff", "")
        lines = diff.splitlines()
        shown = "\n".join(lines[:MAX_DIFF_LINES])
        self.console.print(
            Panel(Syntax(shown, "diff", theme="ansi_dark", word_wrap=True),
                  title="Proposed diff", border_style="green")
        )
        if len(lines) > MAX_DIFF_LINES:
            self.warn(f"Diff truncated: {len(lines) - MAX_DIFF_LINES} more lines are not shown")

    def _on_error(self, ev: AgentEvent) -> None:
        kind = ev.data.get("kind", "Error")
        self.console.print(
            Panel(escape(ev.data.get("message", "Unknown error")), title=f"Error ({kind})",
                  border_style="red")
        )

    # -- approvals and summary
    def apply_summary(self, data: dict[str, Any]) -> None:
        files = ", ".join(data.get("files", []))
        self.console.print(f"\n[bold]Files to change:[/bold] {escape(files)}")
        if data.get("test_passed"):
            self.good(f"Tests passed after {data.get('attempts', '?')} attempt(s)")
        elif data.get("no_tests"):
            self.warn("No tests were found, so this change is UNVERIFIED")
        else:
            self.warn(f"Tests did NOT pass after {data.get('attempts', '?')} attempt(s): UNVERIFIED")

    def finish(self, data: dict[str, Any]) -> int:
        status = data.get("status", "proposed")
        attempts = data.get("attempts", 0)
        passed = bool(data.get("test_passed"))
        unverified = attempts > 0 and not passed

        if status == "applied":
            files = ", ".join(data.get("changed_files", []))
            self.console.print(f"\n[bold green]Done.[/bold green] Applied: {escape(files)}")
        elif status == "rejected":
            self.console.print("\n[bold]Plan rejected.[/bold] No files were changed.")
        elif self.dry_run:
            self.console.print("\n[bold]Dry run.[/bold] No files were changed.")
        elif status == "declined":
            self.console.print("\n[bold]Changes not applied.[/bold] No files were changed.")
        else:
            self.console.print("\nNo files were changed.")

        if unverified:
            self.warn("Tests did not pass: treat this result as unverified (exit code 2)")
            return EXIT_UNVERIFIED
        return EXIT_OK


# ---- status text shown while the agent works ------------------------------
def _next_label(ev: AgentEvent, current: str) -> str:
    if ev.type == "test":
        if ev.data.get("will_retry"):
            return f"Retrying with test feedback (attempt {ev.data.get('attempt', 0) + 1})"
        return "Building the diff"
    if ev.type == "diff":
        return "Writing the explanation"
    if ev.type != "node_done":
        return current
    return {
        "scan_repo": "Selecting relevant files",
        "select_files": "Reading files",
        "read_files": "Planning",
        "make_plan": "Reading the extra files",
        "expand_files": "Re-reading files",
        "plan_approval": "Generating changes",
        "generate_changes": "Validating and testing the changes",
        "apply_approval": "Applying changes",
    }.get(ev.node, current)


# ---- approvals ------------------------------------------------------------
def _decide(data: dict[str, Any], *, yes: bool, dry_run: bool, repo: Path, r: Renderer) -> dict:
    stage = data.get("stage")

    if stage == "plan":
        if yes:
            r.info("Plan auto-approved (--yes)")
            return {"approved": True}
        while True:
            raw = typer.prompt(
                "\nApprove this plan? [y]es / [n]o / [f]eedback", default="y", show_default=False
            ).strip().lower()
            if raw in ("y", "yes", ""):
                return {"approved": True}
            if raw in ("n", "no"):
                return {"approved": False}
            if raw in ("f", "feedback"):
                note = typer.prompt("Your note for the model").strip()
                return {"approved": True, "feedback": note}
            r.warn("Please answer y, n or f")

    if stage == "apply":
        r.apply_summary(data)
        if dry_run:
            r.info("Dry run: not applying")
            return {"approved": False}
        if yes:
            if data.get("test_passed"):
                r.info("Auto-applying (--yes, tests passed)")
                return {"approved": True}
            r.warn("--yes never applies a change whose tests did not pass")
            return {"approved": False}
        ok = typer.confirm(f"\nApply these changes to {repo}?", default=False)
        return {"approved": ok}

    r.warn(f"Unknown approval stage {stage!r}: declining")
    return {"approved": False}


# ---- orchestration --------------------------------------------------------
def execute(
    task: str,
    repo: Path,
    model: str | None,
    yes: bool,
    dry_run: bool,
    console: Console,
    llm=None,
) -> int:
    """Run one task end to end and return the process exit code."""
    r = Renderer(console, dry_run=dry_run)
    thread_id = uuid.uuid4().hex
    r.header(task, repo, model, thread_id)

    stream = stream_agent(task, str(repo), model=model, thread_id=thread_id, llm=llm)
    label = "Checking the task and scanning the repository"
    done: AgentEvent | None = None
    failed = False

    while True:
        pending: AgentEvent | None = None
        with console.status(label):
            for ev in stream:
                if ev.type == "interrupt":
                    pending = ev
                    break
                r.show(ev)
                failed = failed or ev.type == "error"
                if ev.type == "done":
                    done = ev
                label = _next_label(ev, label)
        if pending is None:
            break
        decision = _decide(pending.data, yes=yes, dry_run=dry_run, repo=repo, r=r)
        stream = resume_agent(thread_id, decision)

    if failed:
        return EXIT_ERROR
    if done is None:
        console.print("[red]The run ended without a result.[/red]")
        return EXIT_ERROR
    return r.finish(done.data)


@app.command()
def run(
    task: Annotated[str, typer.Option("--task", "-t", help="What should change in the repo.")],
    repo: Annotated[Path, typer.Option("--repo", "-r", help="Path to the repository.")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Groq model name.")] = None,
    yes: Annotated[bool, typer.Option(
        "--yes", "-y",
        help="Skip prompts. Applies only if tests passed; never applies unverified changes.",
    )] = False,
    dry_run: Annotated[bool, typer.Option(
        "--dry-run", help="Run everything but never write to the repository.",
    )] = False,
) -> None:
    """Plan, change, test and (with your approval) apply a code change."""
    console = Console()
    path = repo.expanduser().resolve()
    if not path.is_dir():
        console.print(f"[red]Repository path is not a directory:[/red] {escape(str(repo))}")
        raise typer.Exit(EXIT_ERROR)
    raise typer.Exit(execute(task, path, model, yes, dry_run, console))


def main() -> None:
    """Console-script entry point. Keeps our exit-code contract for every failure."""
    try:
        code = app(standalone_mode=False)
    except typer.Exit as exc:
        code = exc.exit_code
    except typer.Abort:
        print("Aborted. No files were changed.", file=sys.stderr)
        code = EXIT_ERROR
    except KeyboardInterrupt:
        print("\nInterrupted. No files were changed.", file=sys.stderr)
        code = EXIT_INTERRUPTED
    except Exception as exc:
        show = getattr(exc, "show", None)
        if callable(show) and hasattr(exc, "exit_code"):  # usage error (missing option, etc.)
            show()
            code = EXIT_ERROR
        else:
            raise
    sys.exit(code or 0)