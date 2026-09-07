from exam_parser.models import TaskRecord
from exam_parser.release_content_qa import find_release_content_issues


def _codes(*records: TaskRecord) -> set[str]:
    return {issue.code for issue in find_release_content_issues(records)}


def test_accepts_clean_release_record() -> None:
    codes = _codes(
        TaskRecord(
            task_num="7",
            condition=r"Решите уравнение $\log_{10}x=2$.",
        )
    )

    assert codes == set()


def test_flags_required_figure_without_image() -> None:
    codes = _codes(
        TaskRecord(
            task_num="7",
            condition=(
                "На рисунке изображен график производной функции. "
                "Найдите количество точек минимума."
            ),
        )
    )

    assert "MISSING_REQUIRED_IMAGE" in codes


def test_flags_grid_points_without_image() -> None:
    codes = _codes(
        TaskRecord(
            task_num="B5",
            condition=(
                "На клетчатой бумаге с размером клетки 1×1 отмечены "
                "точки A и B. Найдите длину отрезка AB."
            ),
        )
    )

    assert "MISSING_REQUIRED_IMAGE" in codes


def test_visual_prompt_with_image_is_allowed() -> None:
    codes = _codes(
        TaskRecord(
            task_num="7",
            condition="На рисунке изображен график функции.",
            image_name="task_7.png",
        )
    )

    assert "MISSING_REQUIRED_IMAGE" not in codes


def test_flags_service_source_text() -> None:
    codes = _codes(
        TaskRecord(
            task_num="18",
            condition=(
                "Найдите все значения параметра. Источник: ege.sdamgia.ru "
                "РЕПЕТИТОР ПО МАТЕМАТИКЕ ЯГУБОВ.РФ"
            ),
        )
    )

    assert "SERVICE_TEXT_LEAK" in codes


def test_flags_html_part_marker_leaked_after_condition() -> None:
    codes = _codes(
        TaskRecord(
            task_num="B12",
            condition=(
                "Найдите скорость велосипедиста. "
                '<div style="text-align: center;">Часть 2</div>'
            ),
        )
    )

    assert "SERVICE_TEXT_LEAK" in codes


def test_flags_answer_table_and_copying_notice() -> None:
    codes = _codes(
        TaskRecord(
            task_num="18",
            condition=(
                "Решите задачу. Проверьте, чтобы каждый ответ был записан "
                "рядом с номером соответствующего задания. "
                "<table><tr><td>Номер задания</td><td>Ответ</td></tr></table> "
                "Разрешается свободное копирование в образовательных целях"
            ),
        )
    )

    assert "SERVICE_TEXT_LEAK" in codes
    assert "ANSWER_LEAK" in codes


def test_flags_greek_ocr_answer_word() -> None:
    codes = _codes(
        TaskRecord(
            task_num="15",
            condition=r"Решите неравенство $x>0$. Οτθετ: $(0;+\infty)$",
        )
    )

    assert "GREEK_OCR_TEXT" in codes
    assert "ANSWER_LEAK" in codes


def test_flags_embedded_next_task_label() -> None:
    codes = _codes(
        TaskRecord(
            task_num="A4",
            condition=(
                "На рисунке изображен график одной из функций. Укажите функцию. "
                r"1) $y=x$ 2) $y=x^2$ 3) $y=2^x$ 4) $y=\log_2x$ "
                r"A5 $\begin{array}{l}"
            ),
            image_name="task_A4.png",
        )
    )

    assert "EMBEDDED_NEXT_TASK_LABEL" in codes


def test_flags_impossible_logarithm_base_one() -> None:
    codes = _codes(
        TaskRecord(
            task_num="C3",
            condition=r"Решите неравенство $17^{\frac{\log_{1}1}{17}}<7$.",
        )
    )

    assert "INVALID_LOG_BASE_ONE" in codes


def test_does_not_flag_logarithm_base_ten() -> None:
    codes = _codes(
        TaskRecord(
            task_num="C3",
            condition=r"Вычислите $\log_{10}100$.",
        )
    )

    assert "INVALID_LOG_BASE_ONE" not in codes


def test_flags_suspicious_acceleration_unit() -> None:
    codes = _codes(
        TaskRecord(
            task_num="10",
            condition=(
                r"Автомобиль движется с постоянным ускорением $a\,km/4^{2}$. "
                r"Скорость вычисляется по формуле $v=\sqrt{2la}$."
            ),
        )
    )

    assert "SUSPICIOUS_ACCELERATION_UNIT" in codes


def test_flags_duplicate_single_letter_table_label() -> None:
    condition = (
        "Выберите самый дешёвый вариант. "
        "<table><tr><td>Фирма</td><td>Цена</td></tr>"
        "<tr><td>A</td><td>510</td></tr>"
        "<tr><td>B</td><td>530</td></tr>"
        "<tr><td>B</td><td>570</td></tr></table>"
    )

    codes = _codes(TaskRecord(task_num="B5", condition=condition))

    assert "DUPLICATE_TABLE_LABEL" in codes


def test_flags_duplicate_markdown_table_label() -> None:
    condition = (
        "Цены приведены в таблице.\n\n"
        "| Поставщик | Цена |\n"
        "|---|---|\n"
        "| A | 100 |\n"
        "| B | 200 |\n"
        "| B | 300 |\n"
    )

    codes = _codes(TaskRecord(task_num="B5", condition=condition))

    assert "DUPLICATE_TABLE_LABEL" in codes


def test_flags_duplicate_task_numbers_and_conditions() -> None:
    records = (
        TaskRecord(task_num="17.1", condition="Первое условие."),
        TaskRecord(task_num="17.1.", condition="Другое условие."),
        TaskRecord(task_num="18", condition="Первое условие."),
    )

    codes = _codes(*records)

    assert "DUPLICATE_TASK_NUM" in codes
    assert "DUPLICATE_CONDITION" in codes


def test_flags_internal_gap_in_lettered_task_sequence() -> None:
    records = (
        TaskRecord(task_num="B1", condition="Первая задача."),
        TaskRecord(task_num="B2", condition="Вторая задача."),
        TaskRecord(task_num="B4", condition="Четвёртая задача."),
        TaskRecord(task_num="C1", condition="Следующая часть."),
    )

    issues = find_release_content_issues(records)

    assert any(
        issue.code == "INTERNAL_TASK_NUM_GAP" and "B3" in issue.detail
        for issue in issues
    )


def test_flags_internal_gap_in_numeric_task_sequence() -> None:
    records = (
        TaskRecord(task_num="12", condition="Двенадцатая."),
        TaskRecord(task_num="14", condition="Четырнадцатая."),
        TaskRecord(task_num="15", condition="Пятнадцатая."),
    )

    issues = find_release_content_issues(records)

    assert any(
        issue.code == "INTERNAL_TASK_NUM_GAP" and "13" in issue.detail
        for issue in issues
    )


def test_accepts_contiguous_partial_numeric_sequence() -> None:
    codes = _codes(
        TaskRecord(task_num="12", condition="Двенадцатая."),
        TaskRecord(task_num="13", condition="Тринадцатая."),
        TaskRecord(task_num="14", condition="Четырнадцатая."),
    )

    assert "INTERNAL_TASK_NUM_GAP" not in codes


def test_flags_unexpected_image_for_plain_algebra_task() -> None:
    codes = _codes(
        TaskRecord(
            task_num="B3",
            condition=r"Найдите корень уравнения $\log_6(5-x)=2$.",
            image_name="task_B3.png",
        )
    )

    assert "UNEXPECTED_IMAGE" in codes


def test_flags_malformed_right_latex_delimiter() -> None:
    codes = _codes(
        TaskRecord(
            task_num="C5",
            condition=r"$x^2+y^2=4$, $\left\{y=ax+1,\right $. $xy>0$",
        )
    )

    assert "MALFORMED_LATEX_DELIMITER" in codes


def test_flags_long_prose_hidden_inside_overline_math() -> None:
    codes = _codes(
        TaskRecord(
            task_num="B7",
            condition=(
                r"$\overline{\text{Hайдите соса, если } "
                r"\sin \alpha=-\frac{\sqrt{21}}{5}}$"
            ),
        )
    )

    assert "SUSPICIOUS_PROSE_IN_MATH" in codes
