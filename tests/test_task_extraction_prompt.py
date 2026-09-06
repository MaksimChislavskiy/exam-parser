from exam_parser.task_prompts import build_task_extraction_prompt


def test_prompt_allows_only_unambiguous_unnumbered_task_recovery() -> None:
    prompt = build_task_extraction_prompt(
        "8 Условие.\nБезномерное условие.\n10 Следующее условие.",
        [],
    )

    assert "однозначной локальной последовательности" in prompt
    assert "между задачами" in prompt
    assert "n и n+2" in prompt
    assert "Никогда не заполняй пропущенный номер копией условия соседней задачи" in prompt
    assert "condition должен быть взят только из её собственного OCR-блока" in prompt


def test_prompt_recognizes_ocr_number_marker_variants() -> None:
    prompt = build_task_extraction_prompt("N15\nУсловие", [])

    for marker in ("N15", "NO.15", "NO.1.3", "N=9.1"):
        assert marker in prompt
