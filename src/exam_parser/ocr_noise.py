from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


OCR_UNREADABLE_REPEAT_MARKER = "⟦OCR_UNREADABLE_REPEAT⟧"
OCR_VERIFIED_CONDITION_START = "<!-- OCR_VERIFIED_CONDITION_START -->"
OCR_VERIFIED_CONDITION_END = "<!-- OCR_VERIFIED_CONDITION_END -->"
PATHOLOGICAL_REPEAT_MIN = 128

NON_WHITESPACE_TOKEN_PATTERN = re.compile(r"\S+")
SPACED_TOKEN_REPEAT_PATTERN = re.compile(
    rf"(?<!\S)(?P<unit>\S+)(?:[ \t]+(?P=unit))"
    rf"{{{PATHOLOGICAL_REPEAT_MIN - 1},}}(?!\S)"
)


@dataclass(frozen=True)
class OCRNoiseReplacement:
    unit: str
    repetitions: int
    style: Literal["contiguous", "spaced"]


def sanitize_pathological_ocr_repetitions(
    value: str,
) -> tuple[str, tuple[OCRNoiseReplacement, ...]]:
    """Заменяет только заведомо патологические многотысячные OCR-повторы.

    Порог намеренно намного выше длины обычной формулы. Исходный Markdown при
    этом не меняется: функция предназначена для копии, отправляемой модели и
    используемой дальше как нормализованный OCR-блок.
    """

    replacements: list[OCRNoiseReplacement] = []

    def replace_contiguous_token(match: re.Match[str]) -> str:
        token = match.group(0)
        character, repetitions, run_start, run_end = (
            _longest_identical_run(token)
        )
        if repetitions < PATHOLOGICAL_REPEAT_MIN:
            return token
        replacements.append(
            OCRNoiseReplacement(
                unit=character,
                repetitions=repetitions,
                style="contiguous",
            )
        )
        replacement_start = run_start
        while replacement_start > 0 and token[replacement_start - 1].isalnum():
            replacement_start -= 1
        replacement_end = run_end
        while replacement_end < len(token) and token[replacement_end].isalnum():
            replacement_end += 1
        return (
            token[:replacement_start]
            + OCR_UNREADABLE_REPEAT_MARKER
            + token[replacement_end:]
        )

    cleaned = NON_WHITESPACE_TOKEN_PATTERN.sub(
        replace_contiguous_token,
        value,
    )

    def replace_spaced_run(match: re.Match[str]) -> str:
        unit = match.group("unit")
        repetitions = len(NON_WHITESPACE_TOKEN_PATTERN.findall(match.group(0)))
        replacements.append(
            OCRNoiseReplacement(
                unit=unit,
                repetitions=repetitions,
                style="spaced",
            )
        )
        return OCR_UNREADABLE_REPEAT_MARKER

    cleaned = SPACED_TOKEN_REPEAT_PATTERN.sub(
        replace_spaced_run,
        cleaned,
    )
    return cleaned, tuple(replacements)


def _longest_identical_run(value: str) -> tuple[str, int, int, int]:
    if not value:
        return "", 0, 0, 0

    best_character = value[0]
    best_length = 1
    best_end = 1
    current_character = value[0]
    current_length = 1
    for index, character in enumerate(value[1:], start=1):
        if character == current_character:
            current_length += 1
        else:
            current_character = character
            current_length = 1
        if current_length > best_length:
            best_character = current_character
            best_length = current_length
            best_end = index + 1
    return best_character, best_length, best_end - best_length, best_end
