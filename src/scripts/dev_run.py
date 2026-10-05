import argparse
import uuid

from coding_agent.runner import resume_agent, stream_agent


def show(ev) -> None:
    if ev.type == "start":
        print(f"[start] run {ev.data['thread_id']}")
    elif ev.type == "plan":
        plan = ev.data["plan"]
        print(f"[plan] {plan['summary']}")
        for step in plan["steps"]:
            print(f"   - {step['file']}: {step['action']}")
        for note in plan["assumptions"]:
            print(f"   assumption: {note}")
    elif ev.type == "diff":
        print("[diff]\n" + ev.data["diff"])
    elif ev.type == "interrupt":
        print(f"[approval needed] files: {ev.data['files']}")
    elif ev.type == "done":
        print(f"[done] status={ev.data['status']}")
        if ev.data["explanation"]:
            print("Explanation:", ev.data["explanation"])
    elif ev.type == "error":
        print(f"[error] {ev.data['message']}")
    else:
        print(f"[ok] {ev.node}: {ev.data}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dev runner: auto-approves the plan.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--repo", default="src/sample_project")
    parser.add_argument("--model", default=None)
    parser.add_argument("--reject", action="store_true", help="reject the plan instead")
    args = parser.parse_args()

    thread_id = uuid.uuid4().hex
    events = stream_agent(args.task, args.repo, args.model, thread_id)
    while True:
        paused = False
        for ev in events:
            show(ev)
            paused = paused or ev.type == "interrupt"
        if not paused:
            break
        events = resume_agent(thread_id, {"approved": not args.reject})


if __name__ == "__main__":
    main()