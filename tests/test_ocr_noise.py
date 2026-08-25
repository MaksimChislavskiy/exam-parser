from __future__ import annotations

import pytest

from exam_parser.ocr_noise import (
    OCR_UNREADABLE_REPEAT_MARKER,
    PATHOLOGICAL_REPEAT_MIN,
    sanitize_pathological_ocr_repetitions,
)


@pytest.mark.parametrize(
    ("pdf_stem", "markdown", "character", "style"),
    [
        (
            "00801",
            "C5 " + "▣ " * 512,
            "▣",
            "spaced",
        ),
        (
            "08426",
            "Когда число " + "9 " * 512,
            "9",
            "spaced",
        ),
        (
            "33937",
            "Дано трёхзначное число A из цифр S: 4" + "0" * 512,
            "0",
            "contiguous",
        ),
    ],
)
def test_sanitizes_uploaded_length_failure_patterns(
    pdf_stem: str,
    markdown: str,
    character: str,
    style: str,
) -> None:
    cleaned, replacements = sanitize_pathological_ocr_repetitions(markdown)

    assert OCR_UNREADABLE_REPEAT_MARKER in cleaned, pdf_stem
    assert len(cleaned) < 100, pdf_stem
    assert len(replacements) == 1
    assert replacements[0].character == character
    assert replacements[0].repetitions == 512
    assert replacements[0].style == style


def test_removes_entire_number_token_instead_of_leaving_false_prefix() -> None:
    markdown = "Условие: 4" + "0" * 512 + "."

    cleaned, _ = sanitize_pathological_ocr_repetitions(markdown)

    assert "Условие: 4" not in cleaned
    assert cleaned == f"Условие: {OCR_UNREADABLE_REPEAT_MARKER}."


def test_preserves_long_but_non_pathological_math() -> None:
    markdown = (
        "$10^{100}+11+22+33$ "
        + "0" * (PATHOLOGICAL_REPEAT_MIN - 1)
    )

    cleaned, replacements = sanitize_pathological_ocr_repetitions(markdown)

    assert cleaned == markdown
    assert replacements == ()


def test_sanitizer_is_idempotent() -> None:
    markdown = "C5 " + "▣ " * 512
    cleaned, _ = sanitize_pathological_ocr_repetitions(markdown)

    second_pass, replacements = sanitize_pathological_ocr_repetitions(cleaned)

    assert second_pass == cleaned
    assert replacements == ()
