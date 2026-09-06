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


def test_removes_embedded_trailing_markdown_heading() -> None:
    value = (
        "Найдите наименьшее значение функции $y=x^2$ на $[5;17]$. "
        "## Не забудьте перенести все ответы в бланк ответов № 1"
    )

    assert clean_release_condition(value) == (
        "Найдите наименьшее значение функции $y=x^2$ на $[5;17]$."
    )


def test_removes_nested_dollars_inside_cases_environment() -> None:
    value = (
        r"Решите систему $ \begin{cases}y+\sin x=0,\ $ "
        r"3\sqrt{\sin x}-1)(7y-5)=0.\end{cases} $"
    )

    assert clean_release_condition(value) == (
        r"Решите систему $ \begin{cases}y+\sin x=0,\  "
        r"3\sqrt{\sin x}-1)(7y-5)=0.\end{cases} $"
    )


def test_collapses_display_math_nested_inside_inline_math() -> None:
    value = (
        r"Известны длины ребер: $AB=5$, $AD=12$, $ $$CC_1$$=2$."
    )

    assert clean_release_condition(value) == (
        r"Известны длины ребер: $AB=5$, $AD=12$, $CC_1=2$."
    )


def test_removes_extra_inline_closing_dollar() -> None:
    value = r"Найдите угол между плоскостями $BDD_1$ и $AD_1B_1$$."

    assert clean_release_condition(value) == (
        r"Найдите угол между плоскостями $BDD_1$ и $AD_1B_1$."
    )


def test_does_not_change_valid_display_math() -> None:
    value = r"Вычислите $$x^2+1$$."

    assert clean_release_condition(value) == value
