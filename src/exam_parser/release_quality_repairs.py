"""Финальные универсальные исправления качества перед экспортом."""

from __future__ import annotations

import re
from pathlib import Path

from .models import ExtractedTask


_LATEX_SPAN_PATTERN = re.compile(r"\$(?P<body>.*?)\$", re.DOTALL)
_INLINE_PARAGRAPH_TAG_PATTERN = re.compile(r"</?p\b[^>]*>", re.IGNORECASE)
_SUBPART_PARAGRAPH_PATTERN = re.compile(
    r"<p\b[^>]*>\s*[а-еa-e]\)\s+",
    re.IGNORECASE,
)
_LINE_WRAP_HYPHEN_PATTERN = re.compile(
    r"(?P<left>[А-Яа-яЁё]{2,})-"
    r"(?:(?:[ \t]{2,})|(?:[ \t]*\r?\n[ \t]*\r?\n?[ \t]*))"
    r"(?P<right>[а-яё]{2,})"
)
_DETACHED_NO_SOLUTIONS_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:<[^>\n]+>[ \t]*)*"
    r"не[ \t]+имеет[ \t]+решени(?:й|я)"
    r"[.!]?(?:[ \t]*</[^>\n]+>)*[ \t]*$"
)
_INCOMPLETE_PARAMETER_INEQUALITY_PATTERN = re.compile(
    r"(?is)^\s*Найдите\s+все\s+значения\s+"
    r"(?:\$[^$]+\$|[A-Za-zА-Яа-яЁё])\s*,?\s*"
    r"при\s+каждом\s+из\s+которых\s+неравенство\b"
)
_PRISM_CONTEXT_PATTERN = re.compile(
    r"(?P<prefix>\bпризм[ыаеи]\s+)"
    r"\$(?P<body>[A-Z](?:[A-Z]|_\{?1\}?){5,40})\$"
    r"(?P<between>\s*[-–—]\s*параллелограмм\s+)"
    r"(?P<base>[A-Z]{3,6})\b",
    re.IGNORECASE,
)
_SPLIT_ANGLE_DEGREES_PATTERN = re.compile(
    r"\$\s*\\angle\s*\$\s*"
    r"\$\s*(?P<label>[A-Z]{3})\s*\$\s*=\s*"
    r"(?P<degrees>\d{1,3})\s*"
    r"\$\s*\^\s*\{\s*\\circ\s*\}\s*\$"
)
_ARC_PYRAMID_PATTERN = re.compile(
    r"(?is)(?P<prefix>\bТочка\s+)\$?(?P<point>[A-Z])\$?"
    r"(?P<middle>\s+выбрана\s+на\s+дуге\s+)"
    r"\$?(?P<left>[A-Z])(?P<right>[A-Z])\$?"
    r"(?P<context>.{0,260}?\bобъ[её]м\s+пирамиды\s+)"
    r"\$(?P<name>[A-Z]{4,8})\$"
    r"(?P<suffix>\s+наибольш\w*)"
)
_GEOMETRY_WORD_CONTEXTS = (
    "точк",
    "дуг",
    "угл",
    "плоскост",
    "параллелограмм",
    "призм",
    "пирамид",
)


def install_release_quality_repairs() -> None:
    """Подключает только высокодоверительные финальные исправления."""

    from . import markdown_pipeline as pipeline

    original_clean = pipeline._clean_extracted_task
    if not getattr(original_clean, "_release_quality_repairs", False):
        def clean_extracted_task(task: ExtractedTask) -> ExtractedTask:
            cleaned = original_clean(task)
            condition = repair_release_condition(cleaned.condition)
            if condition == cleaned.condition:
                return cleaned
            return ExtractedTask(
                task_num=cleaned.task_num,
                condition=condition,
                image_id=cleaned.image_id,
            )

        clean_extracted_task._release_quality_repairs = True  # type: ignore[attr-defined]
        pipeline._clean_extracted_task = clean_extracted_task

    original_blocks = pipeline._task_condition_blocks
    if not getattr(original_blocks, "_release_quality_repairs", False):
        def task_condition_blocks(markdown: str) -> dict[str, str]:
            blocks = original_blocks(markdown)
            return _restore_detached_no_solutions(markdown, blocks)

        task_condition_blocks._release_quality_repairs = True  # type: ignore[attr-defined]
        pipeline._task_condition_blocks = task_condition_blocks

    original_embedded = pipeline._remove_embedded_task_conditions
    if not getattr(original_embedded, "_release_quality_repairs", False):
        def remove_embedded_task_conditions(
            extracted: list[tuple[ExtractedTask, Path]],
        ) -> list[tuple[ExtractedTask, Path]]:
            cleaned = original_embedded(extracted)
            return _remove_trailing_math_repeated_in_next_task(cleaned)

        remove_embedded_task_conditions._release_quality_repairs = True  # type: ignore[attr-defined]
        pipeline._remove_embedded_task_conditions = remove_embedded_task_conditions


def repair_release_condition(value: str) -> str:
    """Исправляет безопасно распознаваемые дефекты финального condition."""

    cleaned = _repair_common_ocr_words(value)
    cleaned = _repair_inline_paragraph_noise(cleaned)
    cleaned = _repair_derivative_choice_layout(cleaned)
    cleaned = _repair_prism_notation(cleaned)
    cleaned = _repair_split_angle_degrees(cleaned)
    cleaned = _repair_arc_pyramid_name(cleaned)
    cleaned = _wrap_geometry_labels(cleaned)
    return cleaned.strip()


def _repair_common_ocr_words(value: str) -> str:
    cleaned = re.sub(
        r"(?i)\bBuruncurre\b(?=\s*:)",
        "Вычислите",
        value,
    )
    cleaned = re.sub(
        r"\bперинодом\b",
        "периодом",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bНайлите\b",
        "Найдите",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?i)^\s*Наличие\s+производную\s+функции\b",
        "Найдите производную функции",
        cleaned,
        count=1,
    )
    return cleaned


def _repair_inline_paragraph_noise(value: str) -> str:
    cleaned = _LINE_WRAP_HYPHEN_PATTERN.sub(
        lambda match: match.group("left") + match.group("right"),
        value,
    )
    prose = _LATEX_SPAN_PATTERN.sub(" ", cleaned)
    if "<p" not in cleaned.lower() or _SUBPART_PARAGRAPH_PATTERN.search(prose):
        return cleaned

    cleaned = _INLINE_PARAGRAPH_TAG_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(
        r";\s*\n\s*([A-Za-z])\s*\)",
        lambda match: f";{match.group(1)})",
        cleaned,
    )
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _repair_derivative_choice_layout(value: str) -> str:
    if re.search(r"(?i)^\s*Найдите\s+производную\s+функции\b", value) is None:
        return value

    spans = list(_LATEX_SPAN_PATTERN.finditer(value))
    if not spans:
        return value
    function = spans[-1]
    body = function.group("body")
    if re.match(r"\s*y\s*=", body, re.IGNORECASE) is None:
        return value
    if value[function.end():].strip():
        return value

    prefix = value[:function.start()].rstrip()
    option_markers = [
        int(number)
        for number in re.findall(r"(?m)^\s*([1-4])\)\s+", prefix)
    ]
    if option_markers[-4:] != [1, 2, 3, 4]:
        return value

    option_area_start = re.search(r"(?m)^\s*1\)\s+", prefix)
    if option_area_start is None:
        return value
    intro = prefix[:option_area_start.start()].strip()
    options = prefix[option_area_start.start():].strip()
    if not intro:
        return value

    function_text = function.group(0)
    if len(re.findall(r"e\s*\^\s*\{\s*x\s*\}", options, re.IGNORECASE)) >= 2:
        function_text = re.sub(
            r"e\s*\^\s*\{\s*\d+\s*\}",
            r"e^{x}",
            function_text,
            count=1,
        )

    return f"{intro} {function_text}.\n\n{options}"


def _indexed_labels(value: str) -> list[str]:
    return [
        match.group(1)
        for match in re.finditer(r"([A-Z])_\{?1\}?", value)
    ]


def _repair_prism_notation(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        body = match.group("body")
        base = match.group("base").upper()
        compact = re.sub(r"\s+", "", body)
        if not compact.startswith(base):
            return match.group(0)

        indexed = _indexed_labels(compact[len(base):])
        expected = list(base)
        if indexed != expected[1:]:
            return match.group(0)

        repaired_body = base + "".join(f"{letter}_{{1}}" for letter in expected)
        return (
            match.group("prefix")
            + f"${repaired_body}$"
            + match.group("between")
            + f"${base}$"
        )

    return _PRISM_CONTEXT_PATTERN.sub(replace, value)


def _repair_split_angle_degrees(value: str) -> str:
    return _SPLIT_ANGLE_DEGREES_PATTERN.sub(
        lambda match: (
            f"$\\angle {match.group('label')}="
            f"{match.group('degrees')}^{{\\circ}}$"
        ),
        value,
    )


def _repair_arc_pyramid_name(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        point = match.group("point")
        left = match.group("left")
        right = match.group("right")
        name = match.group("name")
        if point in name:
            return match.group(0)

        pair = left + right
        pair_index = name.find(pair)
        if pair_index < 0:
            return match.group(0)
        repaired_name = (
            name[: pair_index + 1] + point + name[pair_index + 1 :]
        )

        tail = value[match.end(): match.end() + 180]
        if re.search(
            rf"\bточк[аиуеой]*\s+\$?{re.escape(point)}\$?\b",
            tail,
            re.IGNORECASE,
        ) is None:
            return match.group(0)

        return (
            match.group("prefix")
            + f"${point}$"
            + match.group("middle")
            + f"${left}{right}$"
            + match.group("context")
            + f"${repaired_name}$"
            + match.group("suffix")
        )

    return _ARC_PYRAMID_PATTERN.sub(replace, value)


def _wrap_geometry_labels(value: str) -> str:
    """Оборачивает смешанный список геометрических точек в LaTeX.

    Правило намеренно узкое: только после слова ``точка/точки`` и только
    список из двух и более односимвольных латинских обозначений через запятую.
    Оно не касается уже корректных индексированных обозначений вроде
    ``$A_{1}BC$``.
    """

    pattern = re.compile(
        r"(?i)(\bточк(?:а|и|у|е|ой|ами|ах)\s+)"
        r"(?P<labels>\$?[A-Z]\$?"
        r"(?:\s*,\s*\$?[A-Z]\$?){1,4})"
    )

    def replace(match: re.Match[str]) -> str:
        labels = match.group("labels")
        parts = re.split(r"(\s*,\s*)", labels)
        wrapped: list[str] = []
        for part in parts:
            stripped = part.strip()
            if not stripped or "," in part:
                wrapped.append(part)
                continue
            if stripped.startswith("$") and stripped.endswith("$"):
                wrapped.append(part)
                continue
            start_space = part[: len(part) - len(part.lstrip())]
            end_space = part[len(part.rstrip()) :]
            wrapped.append(start_space + f"${stripped}$" + end_space)
        return match.group(1) + "".join(wrapped)

    return pattern.sub(replace, value)


def _restore_detached_no_solutions(
    markdown: str,
    blocks: dict[str, str],
) -> dict[str, str]:
    if _DETACHED_NO_SOLUTIONS_PATTERN.search(markdown) is None:
        return blocks
    if any(
        re.search(r"(?i)\bне\s+имеет\s+решени(?:й|я)\b", condition)
        for condition in blocks.values()
    ):
        return blocks

    candidates = [
        task_num
        for task_num, condition in blocks.items()
        if _INCOMPLETE_PARAMETER_INEQUALITY_PATTERN.search(condition)
        and re.search(
            r"(?s)(?:\\leq|\\geq|<=|>=|<|>)\s*0\s*\$[.!]?\s*$",
            condition,
        )
    ]
    if len(candidates) != 1:
        return blocks

    task_num = candidates[0]
    repaired = dict(blocks)
    repaired[task_num] = blocks[task_num].rstrip() + "\n\nне имеет решений."
    return repaired


def _canonical_math(value: str) -> str:
    return re.sub(r"\s+", "", value).replace(r"\left", "").replace(r"\right", "")


def _remove_trailing_math_repeated_in_next_task(
    extracted: list[tuple[ExtractedTask, Path]],
) -> list[tuple[ExtractedTask, Path]]:
    if len(extracted) < 2:
        return extracted

    result: list[tuple[ExtractedTask, Path]] = []
    for index, (task, page_path) in enumerate(extracted):
        if index + 1 >= len(extracted):
            result.append((task, page_path))
            continue

        next_task, next_page = extracted[index + 1]
        if page_path != next_page:
            result.append((task, page_path))
            continue

        parts = re.split(r"(?:\r?\n){2,}", task.condition.rstrip())
        if len(parts) < 2:
            result.append((task, page_path))
            continue
        tail = parts[-1].strip()
        tail_match = re.fullmatch(r"\$\s*(?P<body>.*?)\s*\$[.!]?", tail, re.DOTALL)
        if tail_match is None:
            result.append((task, page_path))
            continue

        math = _canonical_math(tail_match.group("body"))
        if len(math) < 10:
            result.append((task, page_path))
            continue

        next_prefix = next_task.condition[:320]
        next_math = [
            _canonical_math(match.group("body"))
            for match in _LATEX_SPAN_PATTERN.finditer(next_prefix)
        ]
        if math not in next_math:
            result.append((task, page_path))
            continue

        prefix = "\n\n".join(parts[:-1]).rstrip()
        if len(prefix) < 40:
            result.append((task, page_path))
            continue
        if not (
            re.search(r"(?m)^\s*4\)\s*", prefix)
            or prefix.endswith((".", "?", "!", ")", ":"))
        ):
            result.append((task, page_path))
            continue

        result.append(
            (
                ExtractedTask(
                    task_num=task.task_num,
                    condition=prefix,
                    image_id=task.image_id,
                ),
                page_path,
            )
        )
    return result
