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


VISION_CACHE_HEADER = "exam-parser-vision-context-v2"


class VisualContext(BaseModel):
    """Первичное математически значимое описание рисунка задачи."""

    description: str = Field(min_length=1, max_length=12000)
    checks: list[str] = Field(default_factory=list)


class VisualContextAudit(BaseModel):
    """Независимая проверка первичного описания по исходному рисунку."""

    is_accurate: bool
    issues: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1, max_length=12000)


VISION_CONTEXT_PROMPT = """
Проанализируй рисунок к математической задаче и извлеки только данные,
необходимые для её решения. Не решай саму задачу и не придумывай невидимые данные.

Требования к описанию:
1. Перечисли все подписи, числа, координаты, деления шкал и обозначения.
2. Для координатной сетки сначала найди масштаб по подписанному единичному отрезку.
3. Для каждого вектора обязательно различи начало и конец по наконечнику стрелки.
   Посчитай отдельно число клеток по горизонтали и вертикали от начала к концу.
   Координаты самого вектора запиши как это смещение. Абсолютные координаты начала
   и конца указывай только тогда, когда начало координат и нужные деления видны
   однозначно.
4. После определения вектора выполни обратную проверку: движение от конца к началу
   должно давать компоненты с противоположными знаками. В поле checks кратко запиши,
   как были пересчитаны клетки и проверено направление стрелки.
5. Для графиков укажи координаты отмеченных точек, интервалы возрастания и
   убывания, экстремумы, пересечения и другие явно видимые особенности.
6. Для геометрического рисунка укажи вершины, равенства, углы, параллельность,
   перпендикулярность и положение отмеченных точек.
7. Если часть данных нельзя прочитать уверенно, явно укажи это вместо догадки.
8. Верни компактное, но исчерпывающее описание на языке условия.

Текст задачи для контекста:
{condition}
""".strip()


VISION_AUDIT_PROMPT = """
Независимо перепроверь описание математического рисунка по самому изображению.
Сначала проанализируй изображение заново, не доверяя черновику, затем сравни.

Особенно тщательно проверяй количественные данные:
- масштаб координатной сетки и число клеток;
- положение наконечников стрелок и направление каждого вектора;
- горизонтальное и вертикальное смещение от начала вектора к концу;
- координаты точек, подписи длин, углов и значений на осях;
- любые равенства, параллельность и перпендикулярность.

Для векторов не полагайся только на предполагаемые абсолютные координаты точек.
Независимо пересчитай клетки непосредственно от хвоста к наконечнику стрелки.
Если черновик ошибочен, перечисли ошибки и верни полностью исправленное описание.
Если он верен, верни его смысл без потери данных. Саму задачу не решай.

Текст задачи:
{condition}

Черновое описание:
{description}

Черновые проверки:
{checks}
""".strip()


T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


class MistralVisionContextProvider:
    """Получает и независимо проверяет текстовый контекст рисунка через Mistral."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        audit_model: str | None = None,
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
        self.audit_model = audit_model or os.getenv(
            "MISTRAL_VISION_AUDIT_MODEL",
            self.model,
        )

    def describe(self, task: ExtractedTask, image_path: Path) -> str:
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Не найдено изображение задачи {task.task_num}: {image_path}"
            )

        data_url = _image_data_url(image_path)
        draft = self._request_visual_context(
            model=self.model,
            prompt=VISION_CONTEXT_PROMPT.format(condition=task.condition),
            data_url=data_url,
            response_model=VisualContext,
        )
        audit = self._request_visual_context(
            model=self.audit_model,
            prompt=VISION_AUDIT_PROMPT.format(
                condition=task.condition,
                description=draft.description,
                checks="\n".join(draft.checks) or "не указаны",
            ),
            data_url=data_url,
            response_model=VisualContextAudit,
        )
        if not audit.is_accurate:
            issues = "; ".join(audit.issues) or "найдены неточности"
            print(
                f"Mistral Vision: первичное описание рисунка задачи "
                f"{task.task_num} исправлено: {issues}",
                flush=True,
            )
        return audit.description.strip()

    def _request_visual_context(
        self,
        *,
        model: str,
        prompt: str,
        data_url: str,
        response_model: type[M],
    ) -> M:
        response = _with_rate_limit_retry(
            lambda: self.client.chat.parse(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": data_url},
                        ],
                    }
                ],
                response_format=response_model,
                temperature=0,
            )
        )
        return _parse_structured_response(response, response_model)


def read_cached_visual_context(cache_path: Path) -> str | None:
    if not cache_path.is_file():
        return None
    value = cache_path.read_text(encoding="utf-8").strip()
    if not value:
        return None
    lines = value.splitlines()
    if not lines or lines[0].strip() != VISION_CACHE_HEADER:
        return None
    description = "\n".join(lines[1:]).strip()
    return description or None


def write_visual_context(cache_path: Path, description: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        VISION_CACHE_HEADER + "\n" + description.strip() + "\n",
        encoding="utf-8",
    )


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


def _parse_structured_response(response: object, model: type[M]) -> M:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("Mistral Vision не вернул вариант ответа")

    message = choices[0].message
    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, model):
        return parsed
    if parsed is not None:
        return model.model_validate(parsed)

    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Mistral Vision вернул пустой структурированный ответ")
    return model.model_validate_json(content)


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
