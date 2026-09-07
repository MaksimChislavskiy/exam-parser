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


def test_flags_duplicate_task_numbers_and_conditions() -> None:
    records = (
        TaskRecord(task_num="17.1", condition="Первое условие."),
        TaskRecord(task_num="17.1.", condition="Другое условие."),
        TaskRecord(task_num="18", condition="Первое условие."),
    )

    codes = _codes(*records)

    assert "DUPLICATE_TASK_NUM" in codes
    assert "DUPLICATE_CONDITION" in codes
