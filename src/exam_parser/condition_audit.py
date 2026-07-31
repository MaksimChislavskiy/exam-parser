from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from mistralai.client import Mistral
from PIL import Image
from pydantic import BaseModel, Field


AUDIT_CACHE_VERSION = "exam-parser-condition-audit-v2"

TASK_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*(?P<num>(?:1[0-9]|[1-9])(?:\.\d+)*)"
    r"(?:\.[ \t]+|[ \t]+(?=[A-Za-zА-Яа-яЁё0-9])|[ \t]*$)"
)
ANSWER_FIELD_PATTERN = re.compile(
    r"(?i)(?:\b(?:ответ|otvet)\s*:|_{2,})"
)
ANSWER_INSTRUCTION_PATTERN = re.compile(r"(?i)\bответ\s+выразите\b")
LATEX_SIZE_COMMAND_PATTERN = re.compile(
    r"\\(?:left|right|big|Big|bigg|Bigg|bigl|bigr|Bigl|Bigr)\b"
)


class ConditionCorrection(BaseModel):
    task_num: str = Field(min_length=1)
    source_fragment: str = Field(min_length=1)
    replacement: str
    reason: str = Field(min_length=1)


class PageConditionCorrections(BaseModel):
    corrections: list[ConditionCorrection] = Field(default_factory=list)


CONDITION_AUDIT_PROMPT = r"""
Сверь OCR Markdown с изображениями исходной экзаменационной страницы. Если
страница широкая, изображения показывают её увеличенные левую и правую половины
в порядке слева направо.

Проверь все условия задач на странице посимвольно, особенно:
- числа, знаки, интервалы и разделители координат;
- штрихи производных, степени, индексы и функции;
- греческие и латинские переменные в формулах;
- обозначения фигур, вершин, углов и плоскостей;
- русские слова, окончания, сокращения и метки подпунктов;
- попадание текста следующей задачи в предыдущую.

Верни только объективные ошибки, которые однозначно видны на изображении.
Если хоть один символ нельзя уверенно прочитать, не возвращай исправление.
Не исправляй стиль, пробелы, размер скобок, равносильное оформление LaTeX или
написание разрядов числа с пробелом. Не решай задачи и не восстанавливай текст
по смыслу.

Не изменяй поля для записи ответа вида «Ответ: ____» и не включай их в условия.
Но фраза «Ответ выразите в ...» является частью условия: не удаляй и не добавляй
её без буквального визуального подтверждения.

Для каждой ошибки:
1. task_num — номер задачи;
2. source_fragment — минимальный дословный фрагмент из OCR Markdown. Он должен
   встречаться ровно один раз внутри указанной задачи; не объединяй несколько
   ошибок в одну длинную замену;
3. replacement — минимальная точная замена с сохранением подходящего LaTeX;
4. reason — кратко, что именно видно на изображении.

Не возвращай одну ошибку дважды и не создавай пересекающиеся замены.

OCR Markdown страницы:
{markdown}
""".strip()


CONDITION_AUDIT_CONFIRMATION_PROMPT = r"""
Выполни независимую повторную сверку OCR Markdown с изображениями исходной
экзаменационной страницы. Не опирайся на результаты какого-либо другого прохода:
прочитай страницу заново самостоятельно. Если страница широкая, изображения
показывают её увеличенные левую и правую половины в порядке слева направо.

Верни только объективные ошибки, которые однозначно видны на изображении. Если
хоть один символ нельзя уверенно прочитать, не возвращай исправление. Не исправляй
стиль, пробелы, размер скобок, равносильное оформление LaTeX или разряды числа.
Не изменяй поля «Ответ: ____». Фразу «Ответ выразите в ...» считай частью условия.

Для каждой ошибки верни номер задачи, минимальный уникальный внутри этой задачи
дословный source_fragment, минимальный replacement и краткую причину. Не объединяй
несколько ошибок в одну замену, не создавай пересекающиеся замены и не решай задачи.

OCR Markdown страницы:
{markdown}
""".strip()


T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


class MistralConditionAuditor:
    """Дважды сверяет OCR-условия с увеличенными половинами страницы."""

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

        data_urls = _page_image_data_urls(page_image)
        draft = self._request(
            model=self.model,
            prompt=CONDITION_AUDIT_PROMPT.format(markdown=markdown),
            data_urls=data_urls,
        )
        confirmed = self._request(
            model=self.confirmation_model,
            prompt=CONDITION_AUDIT_CONFIRMATION_PROMPT.format(
                markdown=markdown,
            ),
            data_urls=data_urls,
        )
        return _exactly_confirmed_corrections(
            draft.corrections,
            confirmed.corrections,
        )

    def _request(
        self,
        *,
        model: str,
        prompt: str,
        data_urls: list[str],
    ) -> PageConditionCorrections:
        content = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": data_url}
            for data_url in data_urls
        )
        response = _with_rate_limit_retry(
            lambda: self.client.chat.parse(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": content,
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
    candidates: list[tuple[int, int, ConditionCorrection]] = []
    seen: set[tuple[str, str, str]] = set()
    task_spans = _task_spans(markdown)

    for correction in sorted(
        corrections,
        key=lambda item: (len(item.source_fragment), item.task_num),
    ):
        source = correction.source_fragment
        replacement = correction.replacement
        identity = (correction.task_num, source, replacement)
        if identity in seen:
            continue
        seen.add(identity)

        declared_spans = task_spans.get(correction.task_num, [])
        if not declared_spans:
            print(
                f"Визуальная сверка: страница {page_num}, задача "
                f"{correction.task_num}: замена пропущена, задача не найдена",
                flush=True,
            )
            continue

        matches: list[tuple[int, str, int]] = []
        occurrence_count = 0
        for task_start, task_end in declared_spans:
            task_block = markdown[task_start:task_end]
            occurrences = task_block.count(source)
            occurrence_count += occurrences
            if occurrences == 1:
                matches.append((task_start, task_block, task_block.index(source)))
        if occurrence_count != 1 or len(matches) != 1:
            print(
                f"Визуальная сверка: страница {page_num}, задача "
                f"{correction.task_num}: замена пропущена, исходный фрагмент "
                f"встречается в задаче {occurrence_count} раз",
                flush=True,
            )
            continue
        if source == replacement:
            continue
        if _is_unsafe_or_stylistic_correction(source, replacement):
            print(
                f"Визуальная сверка: страница {page_num}, задача "
                f"{correction.task_num}: стилистическая или небезопасная "
                "замена пропущена",
                flush=True,
            )
            continue

        task_start, _, source_offset = matches[0]
        start = task_start + source_offset
        end = start + len(source)
        overlaps = any(
            start < accepted_end and end > accepted_start
            for accepted_start, accepted_end, _ in candidates
        )
        if overlaps:
            print(
                f"Визуальная сверка: страница {page_num}, задача "
                f"{correction.task_num}: пересекающаяся замена пропущена",
                flush=True,
            )
            continue
        candidates.append((start, end, correction))

    corrected = markdown
    for start, end, correction in sorted(candidates, reverse=True):
        corrected = corrected[:start] + correction.replacement + corrected[end:]
        print(
            f"Визуальная сверка: страница {page_num}, задача "
            f"{correction.task_num}: {correction.reason}",
            flush=True,
        )
    return corrected, len(candidates)


def _task_spans(markdown: str) -> dict[str, list[tuple[int, int]]]:
    headings = list(TASK_HEADING_PATTERN.finditer(markdown))
    result: dict[str, list[tuple[int, int]]] = {}
    for index, heading in enumerate(headings):
        task_num = heading.group("num")
        end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(markdown)
        )
        result.setdefault(task_num, []).append((heading.end(), end))
    return result


def _is_unsafe_or_stylistic_correction(source: str, replacement: str) -> bool:
    if ANSWER_FIELD_PATTERN.search(source) or ANSWER_FIELD_PATTERN.search(
        replacement
    ):
        return True
    source_has_instruction = bool(ANSWER_INSTRUCTION_PATTERN.search(source))
    replacement_has_instruction = bool(
        ANSWER_INSTRUCTION_PATTERN.search(replacement)
    )
    if source_has_instruction != replacement_has_instruction:
        return True
    if len(source) > 500 or len(replacement) > 500:
        return True
    if len(replacement) > len(source) * 4 + 120:
        return True
    return _surface_without_styling(source) == _surface_without_styling(replacement)


def _surface_without_styling(value: str) -> str:
    value = LATEX_SIZE_COMMAND_PATTERN.sub("", value)
    return re.sub(r"\s+", "", value)


def _exactly_confirmed_corrections(
    draft: list[ConditionCorrection],
    confirmed: list[ConditionCorrection],
) -> list[ConditionCorrection]:
    draft_keys = {
        (item.task_num, item.source_fragment, item.replacement)
        for item in draft
    }
    result: list[ConditionCorrection] = []
    seen: set[tuple[str, str, str]] = set()
    for item in confirmed:
        key = (item.task_num, item.source_fragment, item.replacement)
        if key not in draft_keys or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


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


def _page_image_data_urls(image_path: Path) -> list[str]:
    """Возвращает страницу целиком либо две увеличенные половины.

    Широкий PDF-лист в этом проекте содержит два книжных разворота. Передача
    половин отдельными изображениями не даёт vision-модели уменьшить мелкий текст
    обеих страниц до одного низкого разрешения.
    """

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except OSError as error:
        raise ValueError(
            f"Не удалось прочитать изображение страницы: {image_path}"
        ) from error

    width, height = image.size
    if width < height * 1.2:
        return [_pil_image_data_url(image)]

    middle = width // 2
    return [
        _pil_image_data_url(image.crop((0, 0, middle, height))),
        _pil_image_data_url(image.crop((middle, 0, width, height))),
    ]


def _pil_image_data_url(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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
