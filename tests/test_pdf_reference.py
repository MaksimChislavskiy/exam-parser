from __future__ import annotations

import unittest

from exam_parser.pdf_reference import (
    PDF_TASK_HEADING_PATTERN,
    _pdf_geometry_symbols,
    _reconcile_block_symbols,
    _reconcile_reference_symbols,
    _repair_page,
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


class PdfReferenceTests(unittest.TestCase):
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
