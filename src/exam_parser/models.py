from __future__ import annotations

import re
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .math_text import normalize_geometry_notation, normalize_latex_delimiters
from .result_quality import (
    complete_proof_subpart_answer,
    normalize_answer_text,
    normalize_math_typography,
)


DEFAULT_MAX_SOLUTION_CHARS = 8000

EMPTY_HTML_CONTAINER_PATTERN = re.compile(
    r"<(?P<tag>div|p|figure|center)\b[^<>]*>\s*</(?P=tag)\s*>",
    re.IGNORECASE,
)


def repair_latex_control_characters(value: str) -> str:
    """Восстанавливает LaTeX-команды, ошибочно декодированные как JSON escapes."""
    return (
        value.replace("\x08", "\\b")
        .replace("\x0c", "\\f")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def remove_empty_html_containers(value: str) -> str:
    """Удаляет пустые служебные HTML-обёртки, не затрагивая их содержимое."""
    cleaned = value
    while True:
        updated = EMPTY_HTML_CONTAINER_PATTERN.sub("", cleaned)
        if updated == cleaned:
            break
        cleaned = updated

    if cleaned == value:
        return value
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_math_text(value: str) -> str:
    repaired = repair_latex_control_characters(value)
    without_empty_html = remove_empty_html_containers(repaired)
    delimited = normalize_latex_delimiters(without_empty_html)
    typographic = normalize_math_typography(delimited)
    return normalize_geometry_notation(typographic)


def normalize_result_answer(value: str) -> str:
    return normalize_answer_text(normalize_math_text(value))


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


class AngleNotationCheck(BaseModel):
    """Результат отдельной проверки подозрительного обозначения угла."""

    corrected_notation: str | None = None

    @field_validator("corrected_notation")
    @classmethod
    def normalize_corrected_notation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = re.sub(r"[\s$]", "", value).upper()
        if re.fullmatch(r"[A-ZА-ЯЁ]{3}", normalized) is None:
            raise ValueError(
                "исправленное обозначение угла должно состоять из трёх букв"
            )
        return normalized


class ExtractedAnswer(BaseModel):
    task_num: str = Field(min_length=1)
    answer: str = Field(min_length=1)

    @field_validator("answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        return normalize_result_answer(value)


class DocumentAnswerExtraction(BaseModel):
    answers: list[ExtractedAnswer]


class TaskSolution(BaseModel):
    solution: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_SOLUTION_CHARS,
    )
    answer: str = Field(min_length=1)

    @field_validator("solution")
    @classmethod
    def normalize_solution(cls, value: str) -> str:
        return normalize_math_text(value)

    @field_validator("answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        return normalize_result_answer(value)


class SolutionVerification(BaseModel):
    """Результат независимой проверки решения одной задачи."""

    is_correct: bool
    issues: list[str] = Field(default_factory=list)
    solution: str = Field(
        min_length=1,
        max_length=DEFAULT_MAX_SOLUTION_CHARS,
    )
    answer: str = Field(min_length=1)

    @field_validator("solution")
    @classmethod
    def normalize_verified_solution(cls, value: str) -> str:
        return normalize_math_text(value)

    @field_validator("answer")
    @classmethod
    def normalize_verified_answer(cls, value: str) -> str:
        return normalize_result_answer(value)


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

    @field_validator("solution")
    @classmethod
    def normalize_audited_solution(cls, value: str) -> str:
        return normalize_math_text(value)

    @field_validator("answer")
    @classmethod
    def normalize_audited_answer(cls, value: str) -> str:
        return normalize_result_answer(value)


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
        return normalize_result_answer(value)


class VariantMetadata(BaseModel):
    """Метаданные одного экзаменационного варианта для листа about."""

    model_config = ConfigDict(populate_by_name=True)

    school_class: int | str | None = Field(default=None, alias="class")
    year: int | None = None
    date: date | datetime | str | None = None
    topic: int | str | None = None
    exam_id: int | str | None = None
    title: str | None = None
    code: str | None = None
    source_name: str | None = None
    is_public: bool | str | None = None
    source_url: str | None = None
    description: str | None = None


class TaskRecord(BaseModel):
    task_num: str
    condition: str
    image_name: str | None = None
    solution: str = ""
    answer: str = ""
    exams_id: int | None = None
    topics_id: int | None = None

    @field_validator("condition", "solution")
    @classmethod
    def normalize_record_math(cls, value: str) -> str:
        return normalize_math_text(value)

    @field_validator("answer")
    @classmethod
    def normalize_record_answer(cls, value: str) -> str:
        return normalize_result_answer(value)

    @model_validator(mode="after")
    def complete_safe_proof_answer(self) -> "TaskRecord":
        self.answer = complete_proof_subpart_answer(self.condition, self.answer)
        return self
