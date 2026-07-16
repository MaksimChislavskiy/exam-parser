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


SOLUTION_PROMPT = """
Реши математическую задачу. Верни подробное пошаговое решение и короткий
финальный ответ. Сохрани язык условия. Все математические переменные, формулы,
названия отрезков, прямых, углов и вершин оборачивай в одиночные знаки $...$.
Индексы записывай в LaTeX: A1 как $A_1$, A2BB2 как $A_2BB_2$.
Корректно экранируй обратные слэши LaTeX.

Задача {task_num}:
{condition}
""".strip()

SOLUTION_ONLY_PROMPT = """
Реши математическую задачу. Верни только подробное пошаговое решение.
Не добавляй отдельный короткий ответ. Сохрани язык условия. Все математические
переменные, формулы, названия отрезков, прямых, углов и вершин оборачивай в
одиночные знаки $...$. Индексы записывай в LaTeX: A1 как $A_1$,
A2BB2 как $A_2BB_2$. Корректно экранируй обратные слэши LaTeX.

Задача {task_num}:
{condition}
""".strip()

ANSWER_ONLY_PROMPT = """
Реши математическую задачу и верни только короткий финальный ответ без объяснений
и без подробного решения. Сохрани точные числа, знаки, дроби и необходимый LaTeX.

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
2. Игнорируй заголовки, общие инструкции, строки ответа, колонтитулы, справочные
   материалы и страницы/таблицы с готовыми ответами. Верни задачи в порядке чтения.
3. Объедини все подпункты одной нумерованной задачи в поле condition.
4. Сохрани существующий LaTeX. Все остальные математические фрагменты,
   геометрические обозначения и имена вершин оберни в $...$.
   Примеры: A1 -> $A_1$, CC1 -> $CC_1$,
   A2BB2 -> $A_2BB_2$, ABCDA1B1C1D1 -> $ABCDA_1B_1C_1D_1$.
5. Для картинки верни точное имя файла в image_id. Выбирай изображение,
   расположенное внутри блока этой задачи. Если картинки нет — null.
6. Не решай задачи и не придумывай отсутствующий текст.
7. Если на странице нет условий задач, верни пустой список tasks.

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

    def extract_document_answers(self, markdown: str) -> list[ExtractedAnswer]:
        prompt = f"""
Извлеки только готовые ответы из раздела, страницы или таблицы ответов в OCR Markdown.

Правила:
1. Ответы должны быть явно написаны в документе. Никогда не решай задачи сам.
2. Для каждого номера задания верни одну запись task_num + answer.
3. Для заданий с подпунктами сохрани все подпункты одного задания в одном поле
   answer, включая обозначения «а)», «б)», «в)» и математические интервалы.
4. Сохрани точные числа, знаки, дроби и существующий LaTeX. Не сокращай ответ.
5. Игнорируй пустые строки «Ответ: ____» внутри самих заданий.
6. Если явного раздела с ответами нет, верни пустой список answers.

Markdown всех страниц документа:
{markdown}
""".strip()
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
