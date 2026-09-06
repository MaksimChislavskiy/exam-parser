"""Final deterministic cleanup for proven release-format artifacts.

This layer intentionally fixes only unambiguous presentation artifacts. It must
not invent or semantically rewrite OCR content.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .models import ExtractedTask


_INSTALLED = False

# A section heading can leak into an unnumbered task when the task is recovered
# from the prefix of a page. Only Markdown headings are stripped here; plain
# prose such as "Часть 2 ..." may be meaningful inside a condition.
_LEADING_SECTION_HEADING = re.compile(
    r"^\s*#{1,6}\s*Часть\s+\d+\s*[.:;—–-]?\s*",
    re.IGNORECASE,
)

# Broken delimiter nesting produced by OCR/normalization, e.g. ``$$B$_{1}$``.
# The repair is deliberately limited to one symbol plus a braced subscript.
_FRAGMENTED_SUBSCRIPT = re.compile(
    r"\$\$\s*(?P<symbol>[A-Za-zА-ЯЁ])\s*\$_\{\s*(?P<index>[A-Za-z0-9]+)\s*\}\$"
)

# Complete single-dollar inline math spans. Display math ``$$...$$`` and
# incomplete delimiters are intentionally ignored. Spacing is added only
# outside these spans, so already-correct math such as ``$A$`` remains byte-for-
# byte unchanged.
_INLINE_MATH_SPAN = re.compile(
    r"(?<!\$)\$(?!\$)[^$\n]*?(?<!\$)\$(?!\$)"
)
_PROSE_BEFORE_MATH = re.compile(r"[A-Za-zА-Яа-яЁё0-9]")
_PROSE_AFTER_MATH = re.compile(r"[A-Za-zА-Яа-яЁё]")


def _space_inline_math_boundaries(value: str) -> str:
    matches = list(_INLINE_MATH_SPAN.finditer(value))
    if not matches:
        return value

    parts: list[str] = []
    cursor = 0
    for match in matches:
        parts.append(value[cursor : match.start()])
        if (
            match.start() > 0
            and _PROSE_BEFORE_MATH.fullmatch(value[match.start() - 1])
        ):
            parts.append(" ")

        # Preserve the complete math span exactly as it appeared in the input.
        parts.append(match.group(0))

        if (
            match.end() < len(value)
            and _PROSE_AFTER_MATH.fullmatch(value[match.end()])
        ):
            parts.append(" ")
        cursor = match.end()

    parts.append(value[cursor:])
    return "".join(parts)


def clean_release_condition(value: str) -> str:
    """Fixes only deterministic service/Markdown formatting artifacts."""

    cleaned = _LEADING_SECTION_HEADING.sub("", value, count=1)
    cleaned = _FRAGMENTED_SUBSCRIPT.sub(
        lambda match: f"${match.group('symbol')}_{{{match.group('index')}}}$",
        cleaned,
    )
    cleaned = _space_inline_math_boundaries(cleaned)
    return cleaned.strip()


def install_release_condition_cleanup() -> None:
    """Installs final cleanup over the shared extracted-task normalizer."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import markdown_pipeline as pipeline

    original: Callable[[ExtractedTask], ExtractedTask] = pipeline._clean_extracted_task

    def clean_task(task: ExtractedTask) -> ExtractedTask:
        normalized = original(task)
        condition = clean_release_condition(normalized.condition)
        if condition == normalized.condition:
            return normalized
        return ExtractedTask(
            task_num=normalized.task_num,
            condition=condition,
            image_id=normalized.image_id,
        )

    pipeline._clean_extracted_task = clean_task
    _INSTALLED = True
