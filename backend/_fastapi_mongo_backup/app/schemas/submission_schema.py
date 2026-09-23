from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

# submitted -> ai_graded -> approved   (grading_failed if the engine errors, teacher can grade manually)
SubmissionStatus = Literal["submitted", "ai_graded", "grading_failed", "approved"]


class SubmissionCreate(BaseModel):
    assignment_id: str
    content: str = Field(min_length=1, max_length=50_000)


class CriterionScore(BaseModel):
    criterion_id: str
    score: float = Field(ge=0)
    comment: str = ""


class TestResult(BaseModel):
    __test__ = False

    passed: bool
    points: float
    error: Optional[str] = None  # e.g. "timeout", or a traceback line; never the expected output for hidden cases


class AIResult(BaseModel):
    """The engine's suggestion. Stored separately from the teacher's final grade and never shown to students."""
    criterion_scores: list[CriterionScore] = []   # text submissions
    test_results: list[TestResult] = []           # code submissions
    total: float
    feedback: str
    graded_at: datetime


class FinalGrade(BaseModel):
    """What the student sees, set when the teacher approves."""
    criterion_scores: list[CriterionScore] = []
    total: float
    feedback: str
    approved_by: str
    approved_at: datetime


class ReviewRequest(BaseModel):
    """Teacher approves the AI suggestion as-is or overrides scores/feedback."""
    criterion_scores: list[CriterionScore] = []
    total: Optional[float] = Field(default=None, ge=0)  # code submissions: override total directly
    feedback: str = Field(max_length=5000)

    @model_validator(mode="after")
    def need_some_score(self):
        if not self.criterion_scores and self.total is None:
            raise ValueError("Provide criterion_scores (text) or total (code)")
        return self


class TeacherSubmissionView(BaseModel):
    id: str
    assignment_id: str
    student_id: str
    content: str
    status: SubmissionStatus
    ai_result: Optional[AIResult] = None
    final: Optional[FinalGrade] = None
    created_at: datetime


class StudentSubmissionView(BaseModel):
    """No ai_result: the student only sees a grade once the teacher has approved it."""
    id: str
    assignment_id: str
    content: str
    status: Literal["submitted", "approved"]  # ai_graded/grading_failed both read as "submitted"
    final: Optional[FinalGrade] = None
    created_at: datetime
