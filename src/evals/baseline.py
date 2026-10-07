"""
Baseline LLM-only evaluator module.
Gathers repository files into a single prompt context block and submits a single request
without tools, interactive loops, or automated testing steps.
"""

import os
import re
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from groq import Groq

BASELINE_SYSTEM_PROMPT = """You are an expert AI software engineer.
You are provided with a complete repository context containing code and test files, along with a task.
Analyze the codebase and produce the required code fixes or implementations.

You MUST format your output as a single valid JSON object with the following structure:
{
  "files": [
    {
      "path": "relative/path/to/file.py",
      "content": "Full updated code content here"
    }
  ]
}

Do not include markdown code block backticks outside the JSON object. Output ONLY the raw JSON object.
"""

class BaselineRunner:
    def __init__(self, api_key: str | None = None):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def load_repo_context(self, repo_path: Path) -> str:
        """Reads all readable files in repo_path into a formatted text block."""
        context_parts = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.startswith(".") or file.endswith((".pyc", ".git")):
                    continue
                full_path = Path(root) / file
                rel_path = full_path.relative_to(repo_path)
                try:
                    content = full_path.read_text(encoding="utf-8")
                    context_parts.append(f"=== File: {rel_path} ===\n{content}\n")
                except Exception:
                    continue
        return "\n".join(context_parts)

    def run_scenario(
        self,
        scenario: Dict[str, Any],
        model_id: str,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """Runs single-prompt baseline execution on a scenario."""
        repo_path = Path(scenario["repo_path"])
        repo_context = self.load_repo_context(repo_path)
        
        user_prompt = (
            f"TASK:\n{scenario['task']}\n\n"
            f"REPOSITORY CONTEXT:\n{repo_context}\n\n"
            f"Provide the complete modified files required to solve the task."
        )

        start_time = time.time()
        retries = 0
        parsed_first_try = False
        files_dict = {}
        total_tokens = 0

        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=4096
            )
            latency = time.time() - start_time
            
            if response.usage:
                total_tokens = (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)

            raw_text = response.choices[0].message.content or ""
            
            # Parsing logic
            parsed_data, parsed_first_try = self._parse_json_response(raw_text)
            if parsed_data and "files" in parsed_data:
                for item in parsed_data["files"]:
                    if "path" in item and "content" in item:
                        files_dict[item["path"]] = item["content"]

        except Exception as err:
            latency = time.time() - start_time
            return {
                "success": False,
                "error": str(err),
                "parsed_first_try": False,
                "files_produced": {},
                "retries": retries,
                "tokens": total_tokens,
                "latency_sec": round(latency, 2)
            }

        return {
            "success": len(files_dict) > 0,
            "parsed_first_try": parsed_first_try,
            "files_produced": files_dict,
            "retries": retries,
            "tokens": total_tokens,
            "latency_sec": round(latency, 2)
        }

    def _parse_json_response(self, text: str) -> Tuple[Dict[str, Any] | None, bool]:
        """Attempts to parse JSON response directly or via fallback regex extraction."""
        text_str = text.strip()
        try:
            data = json.loads(text_str)
            return data, True
        except json.JSONDecodeError:
            # Fallback extraction from markdown block
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_str, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    return data, False
                except json.JSONDecodeError:
                    pass
            return None, False