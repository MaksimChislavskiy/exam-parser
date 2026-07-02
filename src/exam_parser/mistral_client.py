from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from mistralai.client import Mistral
from pydantic import BaseModel

from .models import ExtractedTask, PageExtraction, TaskSolution


SOLUTION_PROMPT = """
Реши математическую задачу. Верни подробное пошаговое решение и короткий
финальный ответ. Сохрани язык условия. Все математические переменные и формулы
оборачивай в одиночные знаки $...$. Корректно экранируй обратные слэши LaTeX.

Задача {task_num}:
{condition}
""".strip()

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


class MistralTaskClient:
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
        candidates = "\n".join(f"- {image_id}" for image_id in image_ids) or "- нет"
        prompt = f"""
Извлеки математические задачи из OCR Markdown.

Правила:
1. Markdown — единственный источник истины. Не изменяй числа, переменные,
   формулы, номера задач и названия углов.
2. Игнорируй заголовки, общие инструкции, строки ответа, колонтитулы и
   справочные материалы. Верни задачи в порядке чтения.
3. Объедини все подпункты одной нумерованной задачи в поле condition.
4. Сохрани существующий LaTeX. Остальные математические фрагменты оберни в $...$.
5. Для картинки верни точное имя файла в image_id. Если картинки нет — null.
6. Не решай задачи на этом этапе и не придумывай отсутствующий текст.

Файлы изображений:
{candidates}

Markdown:
{markdown}
""".strip()
        response = _with_rate_limit_retry(
            lambda: self.client.chat.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=PageExtraction,
                temperature=0,
            )
        )
        return _parse_response(response, PageExtraction).tasks

    def solve_task(self, task: ExtractedTask) -> TaskSolution:
        response = _with_rate_limit_retry(
            lambda: self.client.chat.parse(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": SOLUTION_PROMPT.format(
                            task_num=task.task_num,
                            condition=task.condition,
                        ),
                    }
                ],
                response_format=TaskSolution,
                temperature=0,
            )
        )
        return _parse_response(response, TaskSolution)


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
