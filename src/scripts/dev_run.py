import argparse
import uuid

from coding_agent.runner import resume_agent, stream_agent
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = str(ROOT / "src" / "sample_project")

def show(ev) -> None:
    d = ev.data
    if ev.type == "start":
        print(f"[start] run {d['thread_id']}")
    elif ev.type == "plan":
        plan = d["plan"]
        print(f"[plan] {plan['summary']}")
        for step in plan["steps"]:
            print(f"   - {step['file']}: {step['action']}")
        for note in plan["assumptions"]:
            print(f"   assumption: {note}")
    elif ev.type == "test":
        if d["passed"]:
            print(f"[test] attempt {d['attempt']}: PASSED")
        else:
            if d["will_retry"]:
                note = "retrying..."
            else:
                note = "no tests found" if d["no_tests"] else "giving up"
            tail = "\n".join(d["output"].splitlines()[-8:])
            print(f"[test] attempt {d['attempt']}: FAILED ({note})\n{tail}")
    elif ev.type == "diff":
        print("[diff]\n" + d["diff"])
    elif ev.type == "interrupt":
        if d["stage"] == "plan":
            print(f"[plan approval] files: {d['files']}")
        else:
            state = "tests passed" if d["test_passed"] else "TESTS NOT PASSING"
            print(f"[apply approval] {len(d['files'])} file(s), {state}")
    elif ev.type == "done":
        print(f"[done] status={d['status']}, test_passed={d['test_passed']}, attempts={d['attempts']}")
        if d["applied"]:
            print("Changes were written to the repo:", ", ".join(d["changed_files"]))
        elif d["status"] != "rejected":
            print("No files were changed in your repo.")
        if d["explanation"]:
            print("Explanation:", d["explanation"])
    elif ev.type == "error":
        print(f"[error] {d['message']}")
    else:
        print(f"[ok] {ev.node}: {d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev runner (auto-answers the approval prompts).")
    parser.add_argument("--task", required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reject", action="store_true", help="reject the plan")
    parser.add_argument("--apply", action="store_true", help="write approved changes to --repo")
    args = parser.parse_args()

    thread_id = uuid.uuid4().hex
    events = stream_agent(args.task, args.repo, args.model, thread_id)
    while True:
        pending = None
        for ev in events:
            show(ev)
            if ev.type == "interrupt":
                pending = ev.data
        if pending is None:
            break
        approved = (not args.reject) if pending["stage"] == "plan" else args.apply
        events = resume_agent(thread_id, {"approved": approved})


if __name__ == "__main__":
    main()