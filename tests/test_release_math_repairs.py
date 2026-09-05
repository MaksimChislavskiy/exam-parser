from __future__ import annotations

from exam_parser.release_math_repairs import repair_release_math


def test_wraps_bare_degree_notation() -> None:
    source = "Углы $ABC$, $BCA$ равны 90° каждый."
    assert repair_release_math(source) == (
        "Углы $ABC$, $BCA$ равны $90^{\\circ}$ каждый."
    )


def test_keeps_degree_notation_already_inside_math() -> None:
    source = r"Угол равен $120^{\circ}$."
    assert repair_release_math(source) == source


def test_does_not_wrap_ordinary_numbers() -> None:
    source = "Радиус основания равен 9. Высота равна 12."
    assert repair_release_math(source) == source


def test_degree_repair_is_idempotent() -> None:
    source = "Угол равен 45°."
    once = repair_release_math(source)
    assert repair_release_math(once) == once
