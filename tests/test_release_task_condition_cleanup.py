from exam_parser.release_condition_cleanup import clean_release_task_condition


def test_trims_sequential_b_section_boundary() -> None:
    value = (
        "Найдите площадь трапеции на клетчатой бумаге. "
        "Ответ дайте в квадратных сантиметрах. "
        r"B7 $\sin \alpha = \frac{\sqrt{21}}{5}$"
    )

    assert clean_release_task_condition("B6", value) == (
        "Найдите площадь трапеции на клетчатой бумаге. "
        "Ответ дайте в квадратных сантиметрах."
    )


def test_trims_sequential_a_section_boundary_after_choices() -> None:
    value = (
        r"Укажите функцию. 1) $y=2^x$ 2) $y=\log_2 x$ "
        r"3) $y=3^x$ 4) $y=\log_3 x$ A5 $\begin{array}{l}x\end{array}$"
    )

    assert clean_release_task_condition("A4", value) == (
        r"Укажите функцию. 1) $y=2^x$ 2) $y=\log_2 x$ "
        r"3) $y=3^x$ 4) $y=\log_3 x$"
    )


def test_does_not_trim_nonsequential_section_label() -> None:
    value = r"В условии упоминается B9 $x=2$ как обозначение из источника."

    assert clean_release_task_condition("B6", value) == value


def test_does_not_trim_numeric_task_number() -> None:
    value = r"Текст задачи 4. A5 $x=2$."

    assert clean_release_task_condition("4", value) == value
