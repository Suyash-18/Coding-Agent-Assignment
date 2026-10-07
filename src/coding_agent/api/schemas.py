from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RunCreate(BaseModel):
    task: str = Field(min_length=1, max_length=2000, description="The coding request in plain language.")
    repo_id: str = Field(default="demo", description="Which repo to work on. Only 'demo' is writable.")
    model: str | None = Field(default=None, description="One of GET /models. Defaults to the server default.")


class DecisionRequest(BaseModel):
    stage: Literal["plan", "apply"] = Field(description="Which approval gate you are answering.")
    action: Literal["approve", "reject"]
    feedback: str | None = Field(
        default=None, max_length=1000,
        description="Plan stage only: extra notes that steer the code generation (an 'edited plan').",
    )


class ScriptRequest(BaseModel):
    task: str | None = Field(default=None, max_length=2000, description="dev_run only.")
    model: str | None = Field(default=None, description="dev_run only.")
    apply: bool = Field(default=False, description="dev_run only: write approved changes to the demo repo.")
    reject: bool = Field(default=False, description="dev_run only: reject the plan.")
