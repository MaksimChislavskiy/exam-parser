from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from mistralai.client import Mistral
from pydantic import BaseModel, Field

from .models import ExtractedTask


class VisualContext(BaseModel):
    """Математически значимое описание рисунка задачи."""

    description: str = Field(min_length=1, max_length=12000)


VISION_CONTEXT_PROMPT = """
Проанализируй рисунок к математической задаче и извлеки только данные,
необходимые для её решения. Не решай задачу и не придумывай невидимые данные.

Требования к описанию:
1. Перечисли все подписи, числа, координаты, деления шкал и обозначения.
2. Для векторов укажи координаты начала, конца и самих векторов.
3. Для графиков укажи координаты отмеченных точек, интервалы возрастания и
   убывания, экстремумы, пересечения и другие явно видимые особенности.
4. Для геометрического рисунка укажи вершины, равенства, углы, параллельность,
   перпендикулярность и положение отмеченных точек.
5. Если часть данных нельзя прочитать уверенно, явно укажи это вместо догадки.
6. Верни компактное, но исчерпывающее текстовое описание на языке условия.

Текст задачи для контекста:
{condition}
""".strip()


T = TypeVar("T")


class MistralVisionContextProvider:
    """Получает текстовый контекст рисунка через vision-модель Mistral."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        project_dir = Path(__file__).resolve().parents[2]
        load_dotenv(project_dir / ".env")
        resolved_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Для анализа изображений в корневом .env не задан MISTRAL_API_KEY"
            )

        self.client = Mistral(api_key=resolved_key)
        self.model = model or os.getenv(
            "MISTRAL_VISION_MODEL",
            "mistral-large-2512",
        )

    def describe(self, task: ExtractedTask, image_path: Path) -> str:
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Не найдено изображение задачи {task.task_num}: {image_path}"
            )

        data_url = _image_data_url(image_path)
        prompt = VISION_CONTEXT_PROMPT.format(condition=task.condition)
        response = _with_rate_limit_retry(
            lambda: self.client.chat.parse(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": data_url},
                        ],
                    }
                ],
                response_format=VisualContext,
                temperature=0,
            )
        )
        return _parse_visual_context(response).description.strip()


def read_cached_visual_context(cache_path: Path) -> str | None:
    if not cache_path.is_file():
        return None
    value = cache_path.read_text(encoding="utf-8").strip()
    return value or None


def write_visual_context(cache_path: Path, description: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(description.strip() + "\n", encoding="utf-8")


def enrich_task_with_visual_context(
    task: ExtractedTask,
    description: str,
) -> ExtractedTask:
    condition = (
        f"{task.condition}\n\n"
        "Данные рисунка, извлечённые vision-моделью и являющиеся частью условия:\n"
        f"{description.strip()}"
    )
    return ExtractedTask(
        task_num=task.task_num,
        condition=condition,
        image_id=task.image_id,
    )


def _image_data_url(image_path: Path) -> str:
    mime_by_suffix = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime = mime_by_suffix.get(image_path.suffix.lower())
    if mime is None:
        raise ValueError(
            f"Неподдерживаемый формат изображения задачи: {image_path.suffix}"
        )
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _parse_visual_context(response: object) -> VisualContext:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("Mistral Vision не вернул вариант ответа")

    message = choices[0].message
    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, VisualContext):
        return parsed
    if parsed is not None:
        return VisualContext.model_validate(parsed)

    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Mistral Vision вернул пустой структурированный ответ")
    return VisualContext.model_validate_json(content)


def _with_rate_limit_retry(call: Callable[[], T], attempts: int = 7) -> T:
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
            print(f"Лимит Mistral Vision; повтор через {delay} сек.", flush=True)
            time.sleep(delay)
    raise RuntimeError("Недостижимое состояние повторного vision-запроса")
