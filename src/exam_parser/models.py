from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .math_text import normalize_geometry_notation, normalize_latex_delimiters


DEFAULT_MAX_SOLUTION_CHARS = 8000


def repair_latex_control_characters(value: str) -> str:
    """Восстанавливает LaTeX-команды, ошибочно декодированные как JSON escapes."""
    return (
        value.replace("\x08", "\\b")
        .replace("\x0c", "\\f")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def normalize_math_text(value: str) -> str:
    repaired = repair_latex_control_characters(value)
    delimited = normalize_latex_delimiters(repaired)
    return normalize_geometry_notation(delimited)


class ExtractedTask(BaseModel):
    task_num: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    image_id: str | None = None

    @field_validator("condition")
    @classmethod
    def normalize_condition(cls, value: str) -> str:
        return normalize_math_text(value)


class PageExtraction(BaseModel):
    tasks: list[ExtractedTask]


class ExtractedAnswer(BaseModel):
    task_num: str = Field(min_length=1)
    answer: str = Field(min_length=1)

    @field_validator("answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        return normalize_math_text(value)


class DocumentAnswerExtraction(BaseModel):
    answers: list[ExtractedAnswer]


class TaskSolution(BaseModel):
    solution: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_SOLUTION_CHARS,
    )
    answer: str = Field(min_length=1)

    @field_validator("solution", "answer")
    @classmethod
    def normalize_solution(cls, value: str) -> str:
        return normalize_math_text(value)


class SolutionVerification(BaseModel):
    """Результат независимой проверки решения одной задачи."""

    is_correct: bool
    issues: list[str] = Field(default_factory=list)
    solution: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_SOLUTION_CHARS,
    )
    answer: str = Field(min_length=1)

    @field_validator("solution", "answer")
    @classmethod
    def normalize_verified_result(cls, value: str) -> str:
        return normalize_math_text(value)


class SolutionConfirmation(BaseModel):
    """Независимое подтверждение уже исправленного решения."""

    is_valid: bool
    issues: list[str] = Field(default_factory=list)


class ProofAudit(BaseModel):
    """Независимый аудит полноты математического доказательства."""

    is_complete: bool
    issues: list[str] = Field(default_factory=list)
    solution: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_SOLUTION_CHARS,
    )
    answer: str = Field(min_length=1)

    @field_validator("solution", "answer")
    @classmethod
    def normalize_audited_result(cls, value: str) -> str:
        return normalize_math_text(value)


class TaskDetailedSolution(BaseModel):
    solution: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_SOLUTION_CHARS,
    )

    @field_validator("solution")
    @classmethod
    def normalize_solution(cls, value: str) -> str:
        return normalize_math_text(value)


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1)

    @field_validator("answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        return normalize_math_text(value)


class TaskRecord(BaseModel):
    task_num: str
    condition: str
    image_name: str | None = None
    solution: str = ""
    answer: str = ""

    @field_validator("condition", "solution", "answer")
    @classmethod
    def normalize_record_math(cls, value: str) -> str:
        return normalize_math_text(value)
