from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from mistralai.client import Mistral
from pydantic import BaseModel, Field


AUDIT_CACHE_VERSION = "exam-parser-condition-audit-v1"


class ConditionCorrection(BaseModel):
    task_num: str = Field(min_length=1)
    source_fragment: str = Field(min_length=1)
    replacement: str
    reason: str = Field(min_length=1)


class PageConditionCorrections(BaseModel):
    corrections: list[ConditionCorrection] = Field(default_factory=list)


CONDITION_AUDIT_PROMPT = r"""
Сверь OCR Markdown с изображением исходной экзаменационной страницы.
Проверь все условия задач на странице посимвольно, особенно:
- числа, знаки, интервалы и разделители координат;
- штрихи производных, степени, индексы и функции;
- греческие и латинские переменные в формулах;
- обозначения фигур, вершин, углов и плоскостей;
- русские слова, окончания, сокращения и метки подпунктов;
- попадание текста следующей задачи в предыдущую.

Верни только объективные ошибки, которые однозначно видны на изображении.
Не исправляй стиль, пробелы и равносильное оформление LaTeX. Не решай задачи и
не восстанавливай текст по смыслу, если символ нельзя уверенно прочитать.

Для каждой ошибки:
1. task_num — номер задачи;
2. source_fragment — дословный фрагмент из OCR Markdown. Он должен встречаться
   на странице ровно один раз; если фрагмент повторяется, включи больше контекста;
3. replacement — точная замена с сохранением подходящего LaTeX;
4. reason — кратко, что именно видно на изображении.

OCR Markdown страницы:
{markdown}
""".strip()


CONDITION_AUDIT_CONFIRMATION_PROMPT = r"""
Независимо перепроверь предложенные исправления OCR по изображению страницы.
Сначала прочитай соответствующие места на изображении самостоятельно, затем
сравни их с OCR и черновыми исправлениями.

Верни только подтверждённые исправления. Удали спорные и стилистические правки.
Если черновая замена неточна, исправь её. source_fragment обязан быть дословно
скопирован из OCR Markdown и встречаться в нём ровно один раз. replacement должен
содержать только текст, действительно видимый на странице. Не решай задачи и не
исправляй их по знаниям предмета без визуального подтверждения.

OCR Markdown страницы:
{markdown}

Черновые исправления:
{draft}
""".strip()


T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


class MistralConditionAuditor:
    """Дважды сверяет OCR-условия с изображением полной страницы."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        confirmation_model: str | None = None,
    ) -> None:
        project_dir = Path(__file__).resolve().parents[2]
        load_dotenv(project_dir / ".env")
        resolved_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Для визуальной сверки условий в корневом .env не задан "
                "MISTRAL_API_KEY"
            )

        self.client = Mistral(api_key=resolved_key)
        self.model = model or os.getenv(
            "MISTRAL_CONDITION_AUDIT_MODEL",
            os.getenv("MISTRAL_VISION_MODEL", "mistral-large-2512"),
        )
        self.confirmation_model = confirmation_model or os.getenv(
            "MISTRAL_CONDITION_CONFIRMATION_MODEL",
            os.getenv("MISTRAL_VISION_AUDIT_MODEL", self.model),
        )

    def audit(
        self,
        markdown: str,
        page_image: Path,
    ) -> list[ConditionCorrection]:
        if not page_image.is_file():
            raise FileNotFoundError(
                f"Не найдено изображение страницы для сверки: {page_image}"
            )

        data_url = _image_data_url(page_image)
        draft = self._request(
            model=self.model,
            prompt=CONDITION_AUDIT_PROMPT.format(markdown=markdown),
            data_url=data_url,
        )
        confirmed = self._request(
            model=self.confirmation_model,
            prompt=CONDITION_AUDIT_CONFIRMATION_PROMPT.format(
                markdown=markdown,
                draft=json.dumps(
                    draft.model_dump(),
                    ensure_ascii=False,
                    indent=2,
                ),
            ),
            data_url=data_url,
        )
        return confirmed.corrections

    def _request(
        self,
        *,
        model: str,
        prompt: str,
        data_url: str,
    ) -> PageConditionCorrections:
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
                response_format=PageConditionCorrections,
                temperature=0,
            )
        )
        return _parse_structured_response(response, PageConditionCorrections)


def verify_markdown_conditions(
    markdown_dir: str | Path,
    pages_dir: str | Path,
    verified_dir: str | Path,
    cache_dir: str | Path,
    *,
    auditor: MistralConditionAuditor | None = None,
) -> Path:
    """Применяет только дважды подтверждённые точечные исправления OCR."""

    markdown_dir = Path(markdown_dir)
    pages_dir = Path(pages_dir)
    verified_dir = Path(verified_dir)
    cache_dir = Path(cache_dir)
    markdown_pages = sorted(
        markdown_dir.glob("page_*/page_*.md"),
        key=_page_number,
    )
    if not markdown_pages:
        return markdown_dir

    auditor = auditor or MistralConditionAuditor()
    replacements: dict[Path, str] = {}
    applied_count = 0

    for markdown_path in markdown_pages:
        page_num = _page_number(markdown_path)
        page_image = pages_dir / f"page_{page_num}.png"
        if not page_image.is_file():
            raise FileNotFoundError(
                f"Для визуальной сверки не найдена страница: {page_image}"
            )

        markdown = markdown_path.read_text(encoding="utf-8")
        corrections = _cached_or_audit(
            auditor,
            markdown,
            page_image,
            cache_dir / f"page_{page_num}.json",
        )
        corrected, applied = _apply_verified_corrections(
            markdown,
            corrections,
            page_num=page_num,
        )
        if corrected != markdown:
            replacements[markdown_path] = corrected
            applied_count += applied

    if not replacements:
        print("Визуальная сверка условий: исправления не требуются", flush=True)
        return markdown_dir

    if verified_dir.exists():
        shutil.rmtree(verified_dir)
    shutil.copytree(markdown_dir, verified_dir)
    for source_path, corrected in replacements.items():
        relative = source_path.relative_to(markdown_dir)
        (verified_dir / relative).write_text(corrected, encoding="utf-8")

    print(
        f"Визуальная сверка условий: применено исправлений {applied_count}",
        flush=True,
    )
    return verified_dir


def _cached_or_audit(
    auditor: MistralConditionAuditor,
    markdown: str,
    page_image: Path,
    cache_path: Path,
) -> list[ConditionCorrection]:
    source_hash = _sha256(markdown.encode("utf-8"))
    image_hash = _sha256(page_image.read_bytes())
    cached = _read_cache(cache_path, source_hash, image_hash)
    if cached is not None:
        print(
            f"Визуальная сверка: используется кэш страницы {_page_number(page_image)}",
            flush=True,
        )
        return cached

    page_num = _page_number(page_image)
    print(f"Mistral Vision: сверка условий страницы {page_num}", flush=True)
    corrections = auditor.audit(markdown, page_image)
    _write_cache(cache_path, source_hash, image_hash, corrections)
    return corrections


def _apply_verified_corrections(
    markdown: str,
    corrections: list[ConditionCorrection],
    *,
    page_num: int,
) -> tuple[str, int]:
    corrected = markdown
    applied = 0
    for correction in corrections:
        source = correction.source_fragment
        replacement = correction.replacement
        occurrences = corrected.count(source)
        if occurrences != 1:
            print(
                f"Визуальная сверка: страница {page_num}, задача "
                f"{correction.task_num}: замена пропущена, исходный фрагмент "
                f"встречается {occurrences} раз",
                flush=True,
            )
            continue
        if source == replacement:
            continue
        corrected = corrected.replace(source, replacement, 1)
        applied += 1
        print(
            f"Визуальная сверка: страница {page_num}, задача "
            f"{correction.task_num}: {correction.reason}",
            flush=True,
        )
    return corrected, applied


def _read_cache(
    cache_path: Path,
    source_hash: str,
    image_hash: str,
) -> list[ConditionCorrection] | None:
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("version") != AUDIT_CACHE_VERSION:
            return None
        if payload.get("source_sha256") != source_hash:
            return None
        if payload.get("image_sha256") != image_hash:
            return None
        parsed = PageConditionCorrections.model_validate(
            {"corrections": payload.get("corrections", [])}
        )
    except (OSError, ValueError, TypeError):
        return None
    return parsed.corrections


def _write_cache(
    cache_path: Path,
    source_hash: str,
    image_hash: str,
    corrections: list[ConditionCorrection],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": AUDIT_CACHE_VERSION,
        "source_sha256": source_hash,
        "image_sha256": image_hash,
        "corrections": [item.model_dump() for item in corrections],
    }
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
            f"Неподдерживаемый формат страницы: {image_path.suffix}"
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


def _page_number(path: Path) -> int:
    match = re.search(r"page_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Не удалось определить номер страницы: {path}")
    return int(match.group(1))
