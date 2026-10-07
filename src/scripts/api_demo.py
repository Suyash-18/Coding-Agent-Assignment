"""Drive one run through the HTTP API and print the live event stream.

    uv run python -m scripts.api_demo --task "Add a /health endpoint with a test" --reset
"""

import argparse
import json
import sys
import time

import httpx


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch an agent run through the HTTP API.")
    ap.add_argument("--task", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--reset", action="store_true", help="reset the demo repo first")
    ap.add_argument("--no-apply", action="store_true", help="decline the apply step")
    args = ap.parse_args()

    api = httpx.Client(base_url=args.base, timeout=60)
    t0 = time.time()

    def say(text: str) -> None:
        print(f"{time.time() - t0:6.1f}s  {text}", flush=True)

    if args.reset:
        say(f"reset demo: {api.post('/repos/demo/reset').json()}")
    created = api.post("/runs", json={"task": args.task, **({"model": args.model} if args.model else {})})
    if created.status_code != 201:
        say(f"could not start: {created.status_code} {created.json()['detail']}")
        return 1
    run_id = created.json()["run_id"]
    say(f"run {run_id[:8]} started")

    with httpx.stream("GET", f"{args.base}/runs/{run_id}/events", timeout=None) as stream:
        for line in stream.iter_lines():
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            kind, data = ev["type"], ev["data"]
            if kind == "plan":
                say(f"plan: {data['plan']['summary']}")
            elif kind == "test":
                say(f"test attempt {data['attempt']}: {'PASSED' if data['passed'] else 'FAILED'}")
            elif kind == "diff":
                say(f"diff for {', '.join(data['files'])}")
            elif kind == "error":
                say(f"ERROR: {data.get('message')}")
            elif kind == "done":
                say(f"done: status={data.get('status')} test_passed={data.get('test_passed')}")
            elif kind == "interrupt":
                stage = data["stage"]
                approve = not (stage == "apply" and args.no_apply)
                say(f"{stage} approval requested -> {'approve' if approve else 'reject'}")
                api.post(f"/runs/{run_id}/decision", json={"stage": stage, "action": "approve" if approve else "reject"})
            else:
                say(f"{kind} {ev['node']}".strip())

    result = api.get(f"/runs/{run_id}/result").json()
    print("\n--- result ---")
    print(f"status={result['status']} applied={result['applied']} tests_passed={result['test_passed']}")
    if result["explanation"]:
        print("explanation:", result["explanation"])
    print(result["diff"] or "(no diff)")
    return 0 if result["state"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
