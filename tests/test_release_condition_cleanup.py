from exam_parser.release_condition_cleanup import clean_release_condition


def test_removes_leading_markdown_section_heading() -> None:
    value = "## Часть 2. Найдите значение выражения $36\\sqrt{6}$."

    assert clean_release_condition(value) == "Найдите значение выражения $36\\sqrt{6}$."


def test_repairs_fragmented_subscript_math() -> None:
    value = "вершинами являются точки $A$, $B$, $C$, $$B$_{1}$ параллелепипеда"

    assert clean_release_condition(value) == (
        "вершинами являются точки $A$, $B$, $C$, $B_{1}$ параллелепипеда"
    )


def test_inserts_missing_space_before_inline_math() -> None:
    value = "параллелепипеда, у которого$AB = 3$, $AD = 3$"

    assert clean_release_condition(value) == (
        "параллелепипеда, у которого $AB = 3$, $AD = 3$"
    )


def test_does_not_strip_plain_part_phrase() -> None:
    value = "Часть 2 отрезка имеет длину $5$."

    assert clean_release_condition(value) == value
