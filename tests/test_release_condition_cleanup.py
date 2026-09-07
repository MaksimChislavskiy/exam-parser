from exam_parser.release_condition_cleanup import clean_release_condition


def test_removes_leading_markdown_section_heading() -> None:
    value = "## Часть 2. Найдите значение выражения $36\\sqrt{6}$."

    assert clean_release_condition(value) == "Найдите значение выражения $36\\sqrt{6}$."


def test_removes_only_generic_leading_markdown_marker() -> None:
    value = "## (Центр) Решите уравнение $x^2=4$."

    assert clean_release_condition(value) == "(Центр) Решите уравнение $x^2=4$."


def test_preserves_plain_leading_region_label() -> None:
    value = "(Центр) Решите уравнение $x^2=4$."

    assert clean_release_condition(value) == value


def test_repairs_fragmented_subscript_math() -> None:
    value = "вершинами являются точки $A$, $B$, $C$, $$B$_{1}$ параллелепипеда"

    assert clean_release_condition(value) == (
        "вершинами являются точки $A$, $B$, $C$, $B_{1}$ параллелепипеда"
    )


def test_repairs_fragmented_ratio_math() -> None:
    value = "Найдите отношение $CK$:KF$."

    assert clean_release_condition(value) == "Найдите отношение $CK:KF$."


def test_preserves_valid_ratio_math() -> None:
    value = "Найдите отношение $CK:KF$."

    assert clean_release_condition(value) == value


def test_repairs_fragmented_label_equality() -> None:
    value = r"У параллелепипеда $AB=3$, $AD=4$, $ $$AA_1 $=5$."

    assert clean_release_condition(value) == (
        r"У параллелепипеда $AB=3$, $AD=4$, $AA_1=5$."
    )


def test_preserves_valid_label_equality() -> None:
    value = r"У параллелепипеда $AB=3$, $AD=4$, $AA_1=5$."

    assert clean_release_condition(value) == value


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


def test_removes_trailing_task_service_marker() -> None:
    value = r"Решите уравнение $\log_5(x-6)=2$. Задачи №7. Условия"

    assert clean_release_condition(value) == r"Решите уравнение $\log_5(x-6)=2$."


def test_removes_only_terminal_next_task_label() -> None:
    value = "Найдите отношение $CK:KF$. C5 |"

    assert clean_release_condition(value) == "Найдите отношение $CK:KF$."


def test_does_not_remove_nonterminal_task_label() -> None:
    value = r"Текст задачи. C5 $x^2=4$ продолжение."

    assert clean_release_condition(value) == value


def test_removes_nested_dollars_inside_cases_environment() -> None:
    value = (
        r"Решите систему $ \begin{cases}y+\sin x=0,\ $ "
        r"3\sqrt{\sin x}-1)(7y-5)=0.\end{cases} $"
    )

    assert clean_release_condition(value) == (
        r"Решите систему $ \begin{cases}y+\sin x=0,\  "
        r"(3\sqrt{\sin x}-1)(7y-5)=0.\end{cases} $"
    )


def test_repairs_missing_opening_parenthesis_in_case_product() -> None:
    value = (
        r"Решите систему $\begin{cases}y-\cos x=0, "
        r"5\sqrt{\cos x}-1)(2y-4)=0.\end{cases}$"
    )

    assert clean_release_condition(value) == (
        r"Решите систему $\begin{cases}y-\cos x=0, "
        r"(5\sqrt{\cos x}-1)(2y-4)=0.\end{cases}$"
    )


def test_does_not_change_valid_case_product_parentheses() -> None:
    value = (
        r"Решите систему $\begin{cases}y-\cos x=0, "
        r"(5\sqrt{\cos x}-1)(2y-4)=0.\end{cases}$"
    )

    assert clean_release_condition(value) == value


def test_does_not_repair_similar_product_outside_cases() -> None:
    value = r"Выражение 5\sqrt{\cos x}-1)(2y-4)=0."

    assert clean_release_condition(value) == value


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


def test_repairs_ocr_kmh_and_acceleration_units() -> None:
    value = (
        r"Автомобиль имеет ускорение $a\,km/4^{2}$ и скорость 80 km/4. "
        r"Ответ выразите в км/ч²."
    )

    assert clean_release_condition(value) == (
        r"Автомобиль имеет ускорение $a\,км/ч^{2}$ и скорость 80 км/ч. "
        r"Ответ выразите в км/ч²."
    )


def test_preserves_km_over_four_outside_physics_context() -> None:
    value = r"Вычислите $km/4$ как отношение переменных."

    assert clean_release_condition(value) == value


def test_repairs_third_label_in_three_provider_table() -> None:
    value = (
        "Выберите самый дешёвый заказ. "
        "<table><tr><td>Фирма</td><td>Цена</td></tr>"
        "<tr><td>A</td><td>400</td></tr>"
        "<tr><td>B</td><td>420</td></tr>"
        "<tr><td>B</td><td>450</td></tr></table>"
    )

    expected = (
        "Выберите самый дешёвый заказ. "
        "<table><tr><td>Фирма</td><td>Цена</td></tr>"
        "<tr><td>A</td><td>400</td></tr>"
        "<tr><td>B</td><td>420</td></tr>"
        "<tr><td>C</td><td>450</td></tr></table>"
    )
    assert clean_release_condition(value) == expected


def test_repairs_cyrillic_confusable_labels_in_three_provider_table() -> None:
    value = (
        "<table><tr><td>Поставщик</td><td>Цена</td></tr>"
        "<tr><td>А</td><td>1</td></tr>"
        "<tr><td>В</td><td>2</td></tr>"
        "<tr><td>В</td><td>3</td></tr></table>"
    )

    expected = (
        "<table><tr><td>Поставщик</td><td>Цена</td></tr>"
        "<tr><td>А</td><td>1</td></tr>"
        "<tr><td>В</td><td>2</td></tr>"
        "<tr><td>C</td><td>3</td></tr></table>"
    )
    assert clean_release_condition(value) == expected


def test_does_not_repair_duplicate_labels_in_unrelated_table() -> None:
    value = (
        "<table><tr><td>Символ</td><td>Значение</td></tr>"
        "<tr><td>A</td><td>1</td></tr>"
        "<tr><td>B</td><td>2</td></tr>"
        "<tr><td>B</td><td>3</td></tr></table>"
    )

    assert clean_release_condition(value) == value


def test_does_not_change_already_correct_provider_table() -> None:
    value = (
        "<table><tr><td>Фирма</td><td>Цена</td></tr>"
        "<tr><td>A</td><td>1</td></tr>"
        "<tr><td>B</td><td>2</td></tr>"
        "<tr><td>C</td><td>3</td></tr></table>"
    )

    assert clean_release_condition(value) == value
