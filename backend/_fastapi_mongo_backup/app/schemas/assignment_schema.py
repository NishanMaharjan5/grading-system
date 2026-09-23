from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

AssignmentType = Literal["text", "code"]


def _new_id() -> str:
    return uuid4().hex[:8]


class Criterion(BaseModel):
    """One rubric line. Scored 0..max_points by the text grader."""
    id: str = Field(default_factory=_new_id)
    name: str = Field(min_length=1, max_length=100)
    description: str = ""
    max_points: float = Field(gt=0, le=1000)


class TestCase(BaseModel):
    """Run the submission with `input` on stdin and compare stdout to `expected_output`."""
    __test__ = False  # stop pytest mistaking this for a test class

    input: str = ""
    expected_output: str
    points: float = Field(default=1, gt=0)
    hidden: bool = False  # hidden cases are never shown to students


class AssignmentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = ""
    type: AssignmentType
    due_date: Optional[datetime] = None
    rubric: list[Criterion] = []       # required for text assignments
    test_cases: list[TestCase] = []    # required for code assignments

    @model_validator(mode="after")
    def check_type_fields(self):
        if self.type == "text":
            if not self.rubric:
                raise ValueError("Text assignments need at least one rubric criterion")
            if self.test_cases:
                raise ValueError("Text assignments cannot have test cases")
            if len({c.id for c in self.rubric}) != len(self.rubric):
                raise ValueError("Rubric criterion ids must be unique")
        else:
            if not self.test_cases:
                raise ValueError("Code assignments need at least one test case")
            if self.rubric:
                raise ValueError("Code assignments are graded by test cases, not a rubric")
        return self


class AssignmentResponse(BaseModel):
    id: str
    title: str
    description: str
    type: AssignmentType
    due_date: Optional[datetime] = None
    rubric: list[Criterion]
    test_cases: list[TestCase]
    total_points: float
    created_by: str
    created_at: datetime


def total_points(a: AssignmentCreate) -> float:
    if a.type == "text":
        return sum(c.max_points for c in a.rubric)
    return sum(t.points for t in a.test_cases)


def redact_hidden_tests(a: AssignmentResponse) -> AssignmentResponse:
    """Student-facing view: hidden test cases are removed, visible ones keep their expected output."""
    return a.model_copy(update={"test_cases": [t for t in a.test_cases if not t.hidden]})
