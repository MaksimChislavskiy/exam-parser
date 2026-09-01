from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from exam_parser.models import ExtractedTask
from exam_parser.pipeline_runtime_v2 import (
    copy_task_image_high_resolution,
    remove_embedded_task_conditions,
    repair_final_condition,
)
from exam_parser.pipeline_runtime_v3 import repair_trapezoid_condition


def test_removes_task_14_and_15_from_task_13_actual_shape() -> None:
    own = (
        "<p>а) Решите уравнение $\\sin 2x=\\sqrt{3}\\sin x$.</p>\n"
        "<p>б) Найдите все корни на заданном отрезке. "
    )
    task_14 = (
        "В правильной четырёхугольной призме $ABCDA_1B_1C_1D_1$ "
        "сторона основания равна 40, а боковое ребро равно $20\\sqrt{2}$. "
        "На рёбрах отмечены точки $K$ и $L$. Плоскость $\\gamma$ "
        "параллельна прямой $BD$ и содержит точки $K$ и $L$.\n"
        "<p>а) Докажите перпендикулярность прямой и плоскости.</p>\n"
        "<p>б) Найдите расстояние от точки до плоскости.</p>"
    )
    task_15 = (
        "Решите неравенство $\\frac{(x+2)^2-4}{x+4}"
        "+\\frac{25}{x+2}\\leq8$."
    )
    page = Path("page_3.md")
    extracted = [
        (
            ExtractedTask(
                task_num="13",
                condition=own + task_14 + " 15 ☐ " + task_15 + "</p>",
                image_id=None,
            ),
            page,
        ),
        (
            ExtractedTask(task_num="14", condition=task_14, image_id=None),
            page,
        ),
        (
            ExtractedTask(task_num="15", condition=task_15, image_id=None),
            page,
        ),
    ]

    repaired = remove_embedded_task_conditions(extracted)

    condition = repaired[0][0].condition
    assert "четырёхугольной призме" not in condition
    assert "Решите неравенство" not in condition
    assert "Найдите все корни" in condition


def test_removes_visual_task_with_corrupted_first_word_after_paragraph() -> None:
    task_b5 = (
        "Функция $y=f(x)$ определена на промежутке $(a;b)$. "
        "На рисунке изображен график ее производной. Найдите число "
        "точек максимума функции $y=f(x)$ на промежутке $(a;b)$."
    )
    task_b4 = (
        "Решите уравнение $12^x-9\\cdot4^x=8\\cdot3^x-72$.\n\n"
        "(Если уравнение имеет более одного корня, то в бланке ответов "
        "запишите сумму корней).\n\n"
        "15 ∂$y=f(x)$ определена на промежутке $(a;b)$. На рисунке "
        "изображен график ее производной. Найдите число точек максимума "
        "функции $y=f(x)$ на промежутке $(a;b)$."
    )
    page = Path("page_2/page_2.md")
    extracted = [
        (ExtractedTask(task_num="B4", condition=task_b4), page),
        (ExtractedTask(task_num="B5", condition=task_b5), page),
    ]

    repaired = remove_embedded_task_conditions(extracted)

    assert repaired[0][0].condition == (
        "Решите уравнение $12^x-9\\cdot4^x=8\\cdot3^x-72$.\n\n"
        "(Если уравнение имеет более одного корня, то в бланке ответов "
        "запишите сумму корней)."
    )
    assert repaired[1][0].condition == task_b5


def test_removes_dangling_geometry_labels_from_credit_task() -> None:
    condition = (
        "В июле 2026 года планируется взять кредит в банке. "
        "Сколько рублей планируется взять в банке?\n\n$ K $\n\n$"
    )

    repaired = repair_final_condition(condition, task_num="16")

    assert repaired.endswith("банке?")
    assert "$ K $" not in repaired


def test_restores_task_17_intro_despite_word_trapezoid_in_subpart() -> None:
    condition = (
        "$K$\n\n$KC$\n"
        "<p>а) Докажите, что $CO = KO$</p>\n"
        "<p>б) Найдите длину основания $BC$, если площадь треугольника "
        "$BCK$ составляет часть площади трапеции $ABCD$.</p>"
    )

    repaired = repair_trapezoid_condition(condition, task_num="17")

    assert repaired.startswith("В трапеции $ABCD$ точка $E$")
    assert "$K$\n\n$KC$" not in repaired
    assert "$CO = KO$.</p>" in repaired


def test_saves_task_image_at_double_dimensions(tmp_path: Path) -> None:
    page_dir = tmp_path / "page_1"
    images_dir = page_dir / "imgs"
    output_dir = tmp_path / "result"
    images_dir.mkdir(parents=True)
    output_dir.mkdir()
    markdown_path = page_dir / "page_1.md"
    markdown_path.write_text("test", encoding="utf-8")

    source = images_dir / "img_in_image_box_0_0_120_80.jpg"
    image = Image.new("RGB", (120, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.line((5, 70, 60, 5, 115, 70), fill="black", width=2)
    image.save(source, "JPEG", quality=100)

    filename = copy_task_image_high_resolution(
        markdown_path,
        source.name,
        output_dir,
        "1",
    )

    assert filename == "task_1.png"
    with Image.open(output_dir / filename) as result:
        assert result.size == (240, 160)
