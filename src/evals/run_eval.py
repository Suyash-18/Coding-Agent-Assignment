#!/usr/bin/env python3
"""
Evaluation Runner for Coding Agent Benchmarking.
Executes scenario x model x mode matrix using the live coding_agent CLI tool.
"""

import os
import sys
import time
import json
import yaml
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List

from evals.baseline import BaselineRunner

MODELS = {
    "GPT-OSS": "openai/gpt-oss-120b",
    "Qwen": "qwen/qwen3-32b"
}

MODES = ["llm-only", "agent"]
REPEATS = 2
PACE_DELAY_SEC = 5

def run_check_command(command: str, cwd: Path) -> bool:
    """Executes verification test command (e.g., pytest) in target repository directory."""
    try:
        res = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=45
        )
        return res.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False

def run_agent_cli(scenario: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """
    Executes live agent CLI command:
    uv run python -m coding_agent run --repo <repo_path> --task "<task>" --model <model_id>
    """
    start_time = time.time()
    repo_path = scenario["repo_path"]
    task = scenario["task"]

    cmd = [
        "uv", "run", "python", "-m", "coding_agent", "run",
        "--repo", repo_path,
        "--task", task,
        "--model", model_id
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180
        )
        latency = time.time() - start_time
        
        # Parse CLI stdout for tokens/retries if output as JSON or structured log
        tokens = 0
        retries = 1
        
        return {
            "success": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "retries": retries,
            "tokens": tokens,
            "latency_sec": round(latency, 2)
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Agent run timed out after 180 seconds",
            "retries": 1,
            "tokens": 0,
            "latency_sec": 180.0
        }
    except Exception as err:
        return {
            "success": False,
            "error": str(err),
            "retries": 0,
            "tokens": 0,
            "latency_sec": round(time.time() - start_time, 2)
        }

def evaluate_scenario_run(
    scenario: Dict[str, Any],
    model_name: str,
    model_id: str,
    mode: str,
    run_idx: int
) -> Dict[str, Any]:
    """Runs evaluation trial, backs up state, applies/verifies patches, and calculates metrics."""
    repo_path = Path(scenario["repo_path"]).resolve()
    backup_path = repo_path.parent / f"{repo_path.name}_backup_{run_idx}"

    # 1. Backup target project state before execution
    if repo_path.exists():
        shutil.copytree(repo_path, backup_path, dirs_exist_ok=True)

    try:
        files_produced = {}
        parsed_first_try = True
        
        if mode == "llm-only":
            runner = BaselineRunner()
            run_res = runner.run_scenario(scenario, model_id)
            files_produced = run_res.get("files_produced", {})
            parsed_first_try = run_res.get("parsed_first_try", False)
            
            # Apply patches for LLM-only mode
            for rel_path, content in files_produced.items():
                target_file = repo_path / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(content, encoding="utf-8")
        else:
            # Live Agent mode executes changes directly on disk
            run_res = run_agent_cli(scenario, model_id)

        # Metric 1: Check expected files modified on disk
        expected_files = scenario.get("expected_files", [])
        files_correct = all((repo_path / ef).exists() for ef in expected_files) if expected_files else False

        # Metric 2: Run verification test suite
        tests_pass = False
        if scenario.get("check_command"):
            tests_pass = run_check_command(scenario["check_command"], cwd=repo_path.parent.parent)

        patch_applies = run_res.get("success", True) if mode == "agent" else len(files_produced) > 0
        manual_score = 5 if (tests_pass and files_correct) else (3 if patch_applies else 1)

        metric_entry = {
            "scenario_id": scenario["id"],
            "model": model_name,
            "mode": mode,
            "run_index": run_idx,
            "files_correct": files_correct,
            "parsed_first_try": parsed_first_try,
            "patch_applies": patch_applies,
            "tests_pass": tests_pass,
            "retries": run_res.get("retries", 0),
            "tokens": run_res.get("tokens", 0),
            "latency_sec": run_res.get("latency_sec", 0.0),
            "manual_score": manual_score
        }

    finally:
        # Restore project directory state to pristine baseline
        if backup_path.exists():
            if repo_path.exists():
                shutil.rmtree(repo_path)
            shutil.move(backup_path, repo_path)

    return metric_entry

def compute_aggregates(raw_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    combos = {}
    for entry in raw_results:
        key = f"{entry['model']} / {entry['mode']}"
        if key not in combos:
            combos[key] = {
                "runs": 0, "files_correct": 0, "parsed_first_try": 0,
                "patch_applies": 0, "tests_pass": 0, "retries": 0,
                "tokens": 0, "latency": 0.0, "manual_score": 0
            }
        
        c = combos[key]
        c["runs"] += 1
        if entry["files_correct"]: c["files_correct"] += 1
        if entry["parsed_first_try"]: c["parsed_first_try"] += 1
        if entry["patch_applies"]: c["patch_applies"] += 1
        if entry["tests_pass"]: c["tests_pass"] += 1
        c["retries"] += entry["retries"]
        c["tokens"] += entry["tokens"]
        c["latency"] += entry["latency_sec"]
        c["manual_score"] += entry["manual_score"]

    aggregates = {}
    for key, c in combos.items():
        n = max(c["runs"], 1)
        aggregates[key] = {
            "runs": n,
            "files_correct_pct": round((c["files_correct"] / n) * 100, 1),
            "parsed_first_try_pct": round((c["parsed_first_try"] / n) * 100, 1),
            "patch_applies_pct": round((c["patch_applies"] / n) * 100, 1),
            "tests_pass_pct": round((c["tests_pass"] / n) * 100, 1),
            "avg_retries": round(c["retries"] / n, 2),
            "avg_tokens": int(c["tokens"] / n),
            "avg_latency_sec": round(c["latency"] / n, 2),
            "avg_manual_score": round(c["manual_score"] / n, 2)
        }
    return aggregates

def main():
    scenarios_file = Path("src/evals/scenarios.yaml")
    if not scenarios_file.exists():
        print(f"Error: {scenarios_file} not found.")
        sys.exit(1)

    with open(scenarios_file, "r", encoding="utf-8") as f:
        scenarios = yaml.safe_load(f).get("scenarios", [])

    all_results = []
    for model_name, model_id in MODELS.items():
        for mode in MODES:
            print(f"\n--- Model: {model_name} | Mode: {mode} ---")
            for sc in scenarios:
                for r in range(1, REPEATS + 1):
                    print(f"Running {sc['id']} (Run {r}/{REPEATS})...", end="", flush=True)
                    res = evaluate_scenario_run(sc, model_name, model_id, mode, r)
                    all_results.append(res)
                    print(f" Done (Pass: {res['tests_pass']}, Time: {res['latency_sec']}s)")
                    time.sleep(PACE_DELAY_SEC)

    aggregates = compute_aggregates(all_results)
    
    Path("src/evals/results.json").write_text(
        json.dumps({"aggregates": aggregates, "raw_results": all_results}, indent=2),
        encoding="utf-8"
    )
    print("\nSaved evaluation results to evals/results.json and updated benchmark status.")

if __name__ == "__main__":
    main()