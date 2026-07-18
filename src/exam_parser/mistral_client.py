from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from mistralai.client import Mistral
from pydantic import BaseModel

from .models import (
    DocumentAnswerExtraction,
    ExtractedAnswer,
    ExtractedTask,
    GeneratedAnswer,
    PageExtraction,
    TaskDetailedSolution,
    TaskSolution,
)
from .task_prompts import (
    ANSWER_ONLY_PROMPT,
    SOLUTION_ONLY_PROMPT,
    SOLUTION_PROMPT,
    build_document_answers_prompt,
    build_task_extraction_prompt,
)


T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


class MistralTaskClient:
    provider_name = "Mistral"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        project_dir = Path(__file__).resolve().parents[2]
        load_dotenv(project_dir / ".env")
        resolved_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not resolved_key:
            raise ValueError("В корневом .env не задан MISTRAL_API_KEY")

        self.client = Mistral(api_key=resolved_key)
        self.model = model or os.getenv("MISTRAL_MODEL", "mistral-large-2512")

    def extract_markdown(
        self,
        markdown: str,
        image_ids: list[str],
    ) -> list[ExtractedTask]:
        prompt = build_task_extraction_prompt(markdown, image_ids)
        response = _with_rate_limit_retry(
            lambda: self.client.chat.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=PageExtraction,
                temperature=0,
            )
        )
        return _parse_response(response, PageExtraction).tasks

    def extract_document_answers(self, markdown: str) -> list[ExtractedAnswer]:
        prompt = build_document_answers_prompt(markdown)
        response = _with_rate_limit_retry(
            lambda: self.client.chat.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=DocumentAnswerExtraction,
                temperature=0,
            )
        )
        return _parse_response(response, DocumentAnswerExtraction).answers

    def solve_task(self, task: ExtractedTask) -> TaskSolution:
        return self._request_task_result(task, SOLUTION_PROMPT, TaskSolution)

    def generate_solution(self, task: ExtractedTask) -> TaskDetailedSolution:
        return self._request_task_result(
            task,
            SOLUTION_ONLY_PROMPT,
            TaskDetailedSolution,
        )

    def generate_answer(self, task: ExtractedTask) -> GeneratedAnswer:
        return self._request_task_result(task, ANSWER_ONLY_PROMPT, GeneratedAnswer)

    def _request_task_result(
        self,
        task: ExtractedTask,
        prompt_template: str,
        response_model: type[T],
    ) -> T:
        response = _with_rate_limit_retry(
            lambda: self.client.chat.parse(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_template.format(
                            task_num=task.task_num,
                            condition=task.condition,
                        ),
                    }
                ],
                response_format=response_model,
                temperature=0,
            )
        )
        return _parse_response(response, response_model)


def _parse_response(response: object, model: type[T]) -> T:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("Mistral не вернул вариант ответа")

    message = choices[0].message
    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, model):
        return parsed
    if parsed is not None:
        return model.model_validate(parsed)

    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Mistral вернул пустой структурированный ответ")
    return model.model_validate_json(content)


def _with_rate_limit_retry(call: Callable[[], R], attempts: int = 7) -> R:
    for attempt in range(attempts):
        try:
            return call()
        except Exception as error:
            message = str(error).lower()
            if not ("status 429" in message or "rate limit" in message):
                raise
            if attempt == attempts - 1:
                raise
            delay = min(30 * (2**attempt), 180)
            print(f"Лимит Mistral; повтор через {delay} сек.", flush=True)
            time.sleep(delay)
    raise RuntimeError("Недостижимое состояние повторного запроса")
