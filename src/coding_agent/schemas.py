from pydantic import BaseModel, Field


class FileSelection(BaseModel):
    relevant_files: list[str] = Field(
        description="Paths from the provided file list that must be read or changed."
    )
    reasoning: str = Field(description="One or two sentences on why these files.")


class PlanStep(BaseModel):
    file: str = Field(description="The file this step touches.")
    action: str = Field(description="The specific change to make in that file.")


class Plan(BaseModel):
    summary: str = Field(description="One sentence describing the overall change.")
    steps: list[PlanStep]
    assumptions: list[str] = Field(
        description="Assumptions made about ambiguous requirements. Empty list if none."
    )


class FileChange(BaseModel):
    path: str = Field(description="Path relative to the repo root.")
    content: str = Field(description="The COMPLETE new content of the file.")


class ChangeSet(BaseModel):
    changes: list[FileChange]