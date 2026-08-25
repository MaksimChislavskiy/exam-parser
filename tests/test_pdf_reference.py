from __future__ import annotations

import unittest

from exam_parser.pdf_reference import (
    PDF_TASK_HEADING_PATTERN,
    _pdf_geometry_symbols,
    _reconcile_block_symbols,
    _reconcile_block_words,
    _reconcile_reference_symbols,
    _repair_page,
    _restore_missing_task_headings,
    _task_blocks,
)


class _FakePdfPage:
    def get_text(self, kind: str):
        if kind == "words":
            return [
                (0, 0, 30, 10, "ABCDA", 0, 0, 0),
                (32, 0, 38, 10, "B", 0, 0, 1),
                (40, 0, 46, 10, "C", 0, 0, 2),
                (48, 0, 54, 10, "D", 0, 0, 3),
            ]
        if kind == "rawdict":
            return {
                "blocks": [
                    {
                        "lines": [
                            {
                                "spans": [
                                    {
                                        "chars": [
                                            {"c": "1", "bbox": (28.5, 4, 31.5, 10)},
                                            {"c": "1", "bbox": (36.5, 4, 39.5, 10)},
                                            {"c": "1", "bbox": (44.5, 4, 47.5, 10)},
                                            {"c": "1", "bbox": (52.5, 4, 55.5, 10)},
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        raise AssertionError(f"Неожиданный формат: {kind}")


class _MathItalicPdfPage:
    def get_text(self, kind: str):
        if kind == "words":
            return [(0, 0, 100, 10, "𝐴𝐵𝐶𝐷𝐴1𝐵1𝐶1𝐷1", 0, 0, 0)]
        if kind == "rawdict":
            return {"blocks": []}
        raise AssertionError(f"Неожиданный формат: {kind}")


class PdfReferenceTests(unittest.TestCase):
    def test_restores_missing_lettered_heading_from_pdf_text(self) -> None:
        markdown = (
            "B8 Решите предыдущую задачу.\n\n"
            "В равнобедренную трапецию вписана окружность. Боковая сторона "
            "трапеции делится точкой касания.\n"
        )
        pdf_text = (
            "B8*\nРешите предыдущую задачу.\n"
            "B9*\nВ равнобедренную трапецию вписана окружность. Боковая "
            "сторона трапеции делится точкой касания.\n"
        )

        repaired, changes = _restore_missing_task_headings(markdown, pdf_text)

        self.assertIn("B9 В равнобедренную трапецию", repaired)
        self.assertEqual(changes, [("B9", "пропущенный номер", "B9")])

    def test_restores_missing_composite_heading_from_pdf_text(self) -> None:
        markdown = (
            "13.7 Решите уравнение.\n\n"
            "Все рёбра правильной четырёхугольной пирамиды равны десяти. "
            "Докажите утверждение.\n\n"
            "14.2 Дана правильная пирамида.\n"
        )
        pdf_text = (
            "13.7 Решите уравнение.\n"
            "14.1 Все рёбра правильной четырёхугольной пирамиды равны "
            "десяти. Докажите утверждение.\n"
            "14.2 Дана правильная пирамида.\n"
        )

        repaired, changes = _restore_missing_task_headings(markdown, pdf_text)

        self.assertIn("14.1 Все рёбра правильной", repaired)
        self.assertEqual(changes, [("14.1", "пропущенный номер", "14.1")])

    def test_restored_block_keeps_ocr_words_over_pdf_text_layer(self) -> None:
        markdown = (
            "13.7 Решите уравнение.\n\n"
            "Все рёбра правильной четырёхугольной пирамиды расположены "
            "параллельно основанию соответственно условию.\n\n"
            "14.2 Дана правильная пирамида.\n"
        )
        pdf_text = (
            "13.7 Решите уравнение.\n"
            "14.1 Все рёбра правильной четырёхугольной пирамиды расположены "
            "параллельно основанию соотвественно условию.\n"
            "14.2 Дана правильная пирамида.\n"
        )

        repaired, changes = _repair_page(markdown, pdf_text)

        self.assertIn("14.1 Все рёбра", repaired)
        self.assertIn("соответственно условию", repaired)
        self.assertNotIn("соотвественно", repaired)
        self.assertEqual(changes, [("14.1", "пропущенный номер", "14.1")])

    def test_does_not_restore_heading_without_unique_condition_anchor(self) -> None:
        markdown = "B8 Решите предыдущую задачу.\n"
        pdf_text = "B8*\nРешите предыдущую задачу.\nB9*\nНайдите число.\n"

        repaired, changes = _restore_missing_task_headings(markdown, pdf_text)

        self.assertEqual(repaired, markdown)
        self.assertEqual(changes, [])

    def test_does_not_restore_isolated_composite_heading(self) -> None:
        markdown = "Все рёбра правильной четырёхугольной пирамиды равны.\n"
        pdf_text = "14.1 Все рёбра правильной четырёхугольной пирамиды равны.\n"

        repaired, changes = _restore_missing_task_headings(markdown, pdf_text)

        self.assertEqual(repaired, markdown)
        self.assertEqual(changes, [])

    def test_repairs_reordered_angle_name(self) -> None:
        markdown = (
            "В треугольнике АВС проведены высоты АА1 и ВВ1. "
            "Известно, что А1В1 = 4. Найдите угол $ABC$."
        )
        pdf_text = (
            "В треугольнике АВС проведены высоты АА1 и ВВ1. "
            "Известно, что А1В1 = 4. Найдите угол АСВ."
        )

        repaired, changes = _reconcile_block_symbols(markdown, pdf_text)

        self.assertIn("$ACB$", repaired)
        self.assertNotIn("$ABC$.", repaired)
        self.assertEqual(changes, [("ABC", "ACB")])

    def test_repairs_replaced_point_in_triangle(self) -> None:
        markdown = (
            "Найдите отношение площади треугольника $APO$ к площади "
            "трапеции $ABCD$, если Q — точка пересечения диагоналей."
        )
        pdf_text = (
            "Найдите отношение площади треугольника APQ к площади "
            "трапеции ABCD, если Q — точка пересечения диагоналей."
        )

        repaired, changes = _reconcile_block_symbols(markdown, pdf_text)

        self.assertIn("$APQ$", repaired)
        self.assertEqual(changes, [("APO", "APQ")])

    def test_repairs_one_missing_character_in_long_geometry_label(self) -> None:
        markdown = (
            "В кубе $ABCD_1B_1C_1D_1$ точка K — середина ребра $CC_1$."
        )
        pdf_text = (
            "В кубе ABCDA1B1C1D1 точка K — середина ребра CC1."
        )

        repaired, changes = _reconcile_block_symbols(markdown, pdf_text)

        self.assertIn("$ABCDA_1B_1C_1D_1$", repaired)
        self.assertEqual(changes, [("ABCD1B1C1D1", "ABCDA1B1C1D1")])

    def test_pdf_layout_rebuilds_separate_upper_indices(self) -> None:
        symbols = _pdf_geometry_symbols(_FakePdfPage())

        self.assertEqual(symbols, ["ABCDA1B1C1D1"])

    def test_pdf_math_italic_geometry_letters_are_normalized(self) -> None:
        symbols = _pdf_geometry_symbols(_MathItalicPdfPage())

        self.assertEqual(symbols, ["ABCDA1B1C1D1"])

    def test_repairs_single_character_spelling_error_from_pdf(self) -> None:
        repaired, changes = _reconcile_block_words(
            "Окружность пересекает гипотензу треугольника.",
            "Окружность пересекает гипотенузу треугольника.",
        )

        self.assertEqual(
            repaired,
            "Окружность пересекает гипотенузу треугольника.",
        )
        self.assertEqual(changes, [("гипотензу", "гипотенузу")])

    def test_page_fallback_repairs_only_missing_character(self) -> None:
        markdown = "В кубе $ABCD_1B_1C_1D_1$."

        repaired, changes = _reconcile_reference_symbols(
            markdown,
            _pdf_geometry_symbols(_FakePdfPage()),
        )

        self.assertEqual(repaired, "В кубе $ABCDA_1B_1C_1D_1$.")
        self.assertEqual(changes, [("ABCD1B1C1D1", "ABCDA1B1C1D1")])

    def test_page_fallback_does_not_make_equal_length_replacement(self) -> None:
        markdown = "Найдите угол $ABC$."

        repaired, changes = _reconcile_reference_symbols(markdown, ["ACB"])

        self.assertEqual(repaired, markdown)
        self.assertEqual(changes, [])

    def test_does_not_expand_short_geometry_label(self) -> None:
        markdown = "Рассмотрите отрезок $AB$."
        pdf_text = "Рассмотрите треугольник ABC."

        repaired, changes = _reconcile_block_symbols(markdown, pdf_text)

        self.assertEqual(repaired, markdown)
        self.assertEqual(changes, [])

    def test_keeps_matching_geometry_labels(self) -> None:
        markdown = (
            "Основание призмы $ABCA_1B_1C_1$, стороны $AB=BC=2$, "
            "точка M лежит на $AA_1$."
        )
        pdf_text = (
            "Основание призмы ABCA1B1C1, стороны AB=BC=2, "
            "точка M лежит на AA1."
        )

        repaired, changes = _reconcile_block_symbols(markdown, pdf_text)

        self.assertEqual(repaired, markdown)
        self.assertEqual(changes, [])

    def test_does_not_replace_unrelated_different_labels(self) -> None:
        markdown = "Рассмотрите треугольник ABC и отрезок MN."
        pdf_text = "Рассмотрите четырёхугольник PQRS и отрезок KL."

        repaired, changes = _reconcile_block_symbols(markdown, pdf_text)

        self.assertEqual(repaired, markdown)
        self.assertEqual(changes, [])

    def test_pdf_formula_number_is_not_mistaken_for_task_heading(self) -> None:
        pdf_text = (
            "1. В треугольнике АВС площадь круга равна\n"
            "π\n"
            "16\n"
            ". Найдите угол АСВ.\n"
            "2. Следующая задача."
        )

        blocks = _task_blocks(
            pdf_text,
            heading_pattern=PDF_TASK_HEADING_PATTERN,
        )

        self.assertEqual([block.task_num for block in blocks], ["1", "2"])
        self.assertIn("Найдите угол АСВ", blocks[0].text)

    def test_single_standalone_page_number_is_not_a_task_heading(self) -> None:
        blocks = _task_blocks(
            "Справочные материалы.\n1\n",
            heading_pattern=PDF_TASK_HEADING_PATTERN,
        )

        self.assertEqual(blocks, [])

    def test_repairs_angle_after_split_formula_number_in_pdf(self) -> None:
        markdown = (
            "1. В треугольнике АВС площадь круга равна 16π. "
            "Найдите угол $ABC$.\n"
            "2. Следующая задача."
        )
        pdf_text = (
            "1. В треугольнике АВС площадь круга равна\n"
            "π\n"
            "16\n"
            ". Найдите угол АСВ.\n"
            "2. Следующая задача."
        )

        repaired, changes = _repair_page(markdown, pdf_text)

        self.assertIn("Найдите угол $ACB$", repaired)
        self.assertEqual(changes, [("1", "ABC", "ACB")])


if __name__ == "__main__":
    unittest.main()
