from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from exam_parser.cli import _remove_legacy_single_result, main
from exam_parser.markdown_boundaries import normalize_task_boundaries
from exam_parser.models import TaskRecord
from exam_parser.variants import (
    detect_document_variants,
    materialize_variant_markdown,
    variant_page_paths,
)


def _write_page(root: Path, page_num: int, markdown: str) -> Path:
    page_dir = root / f"page_{page_num}"
    page_dir.mkdir(parents=True)
    page_path = page_dir / f"page_{page_num}.md"
    page_path.write_text(markdown, encoding="utf-8")
    return page_path


class VariantDetectionTests(unittest.TestCase):
    def test_detects_four_labeled_variants_without_fixed_page_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            markdown_dir = Path(temp)
            pages = [
                (
                    1,
                    "Вариант МА2510509\nИнструкция по выполнению работы\n"
                    "Часть 1\n1 Первая",
                ),
                (2, "5 Пятая\n12 Двенадцатая"),
                (3, "Часть 2\n13 Тринадцатая\n19 Девятнадцатая"),
                (
                    4,
                    "Вариант MA2510510\nИнструкция по выполнению работы\n"
                    "Часть 1\n1 Первая",
                ),
                (5, "12 Двенадцатая"),
                (6, "Часть 2\n19 Девятнадцатая"),
                (
                    7,
                    "Вариант MA2510511\nИнструкция по выполнению работы\n"
                    "Часть 1\n1 Первая",
                ),
                (8, "Часть 2\n13 Тринадцатая\n19 Девятнадцатая"),
                (
                    9,
                    "Вариант MA2510512\nИнструкция по выполнению работы\n"
                    "Часть 1\n1 Первая",
                ),
                (10, "5 Пятая"),
                (11, "12 Двенадцатая"),
                (12, "Часть 2\n19 Девятнадцатая"),
            ]
            for page_num, markdown in pages:
                _write_page(markdown_dir, page_num, markdown)

            variants = detect_document_variants(markdown_dir)

        self.assertEqual(
            [item.identifier for item in variants],
            ["MA2510509", "MA2510510", "MA2510511", "MA2510512"],
        )
        self.assertEqual(
            [item.page_numbers for item in variants],
            [(1, 2, 3), (4, 5, 6), (7, 8), (9, 10, 11, 12)],
        )

    def test_markdown_formatting_and_cyrillic_letters_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            markdown_dir = Path(temp)
            _write_page(
                markdown_dir,
                1,
                "## Вариант **МА 2510509**\n\nИнструкция по выполнению работы",
            )

            variants = detect_document_variants(markdown_dir)

        self.assertEqual(variants[0].identifier, "MA2510509")
        self.assertEqual(variants[0].output_name, "MA2510509")

    def test_unlabeled_variants_use_conservative_task_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            markdown_dir = Path(temp)
            _write_page(markdown_dir, 1, "Часть 1\n1 Первая\n12 Двенадцатая")
            _write_page(markdown_dir, 2, "Часть 2\n13 Тринадцатая\n19 Последняя")
            _write_page(markdown_dir, 3, "Часть 1\n1 Первая нового варианта")
            _write_page(markdown_dir, 4, "Часть 2\n19 Последняя нового варианта")

            variants = detect_document_variants(markdown_dir)

        self.assertEqual(len(variants), 2)
        self.assertEqual(
            [item.output_name for item in variants],
            ["variant_1", "variant_2"],
        )
        self.assertEqual(
            [item.page_numbers for item in variants],
            [(1, 2), (3, 4)],
        )

    def test_00558_instructions_split_when_ocr_loses_b1_and_b2(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            markdown_dir = Path(temp)
            _write_page(
                markdown_dir,
                1,
                "Инструкция по выполнению работы\nЧасть 1\nВ1 Первая\nВ3 Третья",
            )
            _write_page(markdown_dir, 2, "В14 Последняя краткая")
            _write_page(markdown_dir, 3, "Часть 2\nC1 Первая сложная\nC6 Последняя")
            _write_page(
                markdown_dir,
                4,
                "Инструкция по выполнению работы\nЧасть 1\n"
                "Первое задание без номера\nВ3 Третья нового варианта",
            )
            _write_page(markdown_dir, 5, "В14 Последняя краткая нового варианта")
            _write_page(markdown_dir, 6, "Часть 2\nC1 Новая сложная\nC6 Последняя")

            variants = detect_document_variants(markdown_dir)

        self.assertEqual(
            [item.page_numbers for item in variants],
            [(1, 2, 3), (4, 5, 6)],
        )

    def test_merges_shuffled_final_section_suffix_into_legacy_variant(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            markdown_dir = Path(temp)
            _write_page(
                markdown_dir,
                1,
                "Для записи ответов на задания C3-C5 используйте бланк №2.\n"
                "СЗ Найдите параметры.\n"
                "*C4 Докажите утверждение.\n"
                "C5 Найдите значение.",
            )
            _write_page(
                markdown_dir,
                2,
                "Ответом на задания B1-B11 должно быть число.\n"
                + "\n".join(
                    f"B{number} Найдите значение."
                    for number in (*range(1, 5), *range(6, 12))
                )
                + "\nДля записи решений заданий C1 и C2 используйте бланк.\n"
                "C1 Решите уравнение.\nC2 Решите неравенство.",
            )
            _write_page(
                markdown_dir,
                3,
                "При выполнении заданий A1-A10 выберите ответ.\n"
                + "\n".join(
                    f"A{number} Выберите ответ."
                    for number in range(2, 11)
                ),
            )

            variants = detect_document_variants(markdown_dir)

        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].page_numbers, (1, 2, 3))

    def test_does_not_merge_shuffled_suffix_when_task_sets_overlap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            markdown_dir = Path(temp)
            _write_page(
                markdown_dir,
                1,
                "Для записи ответов на задания C3-C5 используйте бланк №2.\n"
                "C3 Найдите параметры.\nC4 Докажите.\nC5 Найдите.",
            )
            _write_page(
                markdown_dir,
                2,
                "Ответом на задания B1-B11 должно быть число.\n"
                "B1 Найдите.\nB2 Найдите.\nB3 Найдите.",
            )
            _write_page(
                markdown_dir,
                3,
                "При выполнении заданий A1-A10 выберите ответ.\n"
                + "\n".join(
                    f"A{number} Выберите ответ."
                    for number in range(1, 11)
                )
                + "\nДля записи решений заданий C1-C5 используйте бланк.\n"
                "C1 Решите.\nC2 Решите.\nC3 Другая задача.",
            )

            variants = detect_document_variants(markdown_dir)

        self.assertEqual(len(variants), 2)
        self.assertEqual(
            [variant.page_numbers for variant in variants],
            [(1,), (2, 3)],
        )

    def test_numbered_list_without_part_one_does_not_split_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            markdown_dir = Path(temp)
            _write_page(markdown_dir, 1, "13 Сложная задача")
            _write_page(
                markdown_dir,
                2,
                "19 Продолжение условия\n1. Первый случай\n2. Второй случай",
            )

            variants = detect_document_variants(markdown_dir)

        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].page_numbers, (1, 2))

    def test_repeated_identifier_gets_unique_output_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            markdown_dir = Path(temp)
            _write_page(
                markdown_dir,
                1,
                "Вариант 7\nИнструкция по выполнению работы\n1 Первая\n19 Последняя",
            )
            _write_page(
                markdown_dir,
                2,
                "Вариант 7\nИнструкция по выполнению работы\n1 Новая первая",
            )

            variants = detect_document_variants(markdown_dir)

        self.assertEqual([item.output_name for item in variants], ["7", "7_2"])

    def test_variant_page_paths_keep_original_page_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            markdown_dir = Path(temp)
            first = _write_page(markdown_dir, 1, "Часть 1\n1 Первая")
            second = _write_page(markdown_dir, 2, "19 Последняя")
            variant = detect_document_variants(markdown_dir)[0]

            paths = variant_page_paths(markdown_dir, variant)

        self.assertEqual(paths, [first, second])

    def test_33925_partial_exam_sets_and_two_variants_on_one_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            markdown_dir = Path(temp)
            long_instruction = (
                "Для записи решений и ответов на задания 13–19 используйте "
                "бланк ответов № 2."
            )
            _write_page(
                markdown_dir,
                1,
                f"{long_instruction}\n13 Первая\n19 Последняя",
            )
            _write_page(
                markdown_dir,
                2,
                f"{long_instruction}\n13 Новая первая\n19 Новая последняя",
            )
            _write_page(
                markdown_dir,
                3,
                "## Часть 2\nC1 Первая\nC6 Последняя\n"
                "<div>Часть 2</div>\nC1 Другая первая\nC6 Другая последняя",
            )

            variants = detect_document_variants(markdown_dir)

        self.assertEqual(len(variants), 4)
        self.assertEqual(
            [item.page_numbers for item in variants],
            [(1,), (2,), (3,), (3,)],
        )
        self.assertFalse(variants[1].has_partial_pages)
        self.assertTrue(variants[2].has_partial_pages)
        self.assertTrue(variants[3].has_partial_pages)

    def test_materializes_exact_page_fragments_and_keeps_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            markdown_dir = root / "source"
            source = (
                "## Часть 2\nC1 Первая\n"
                '<img src="imgs/first.jpg">\nC6 Последняя\n'
                "<div>Часть 2</div>\nC1 Другая первая\n"
                '<img src="imgs/second.jpg">\nC6 Другая последняя'
            )
            page_path = _write_page(markdown_dir, 10, source)
            images_dir = page_path.parent / "imgs"
            images_dir.mkdir()
            (images_dir / "first.jpg").write_bytes(b"first")
            (images_dir / "second.jpg").write_bytes(b"second")
            variants = detect_document_variants(markdown_dir)

            first_dir = materialize_variant_markdown(
                markdown_dir,
                root / "first",
                variants[0],
            )
            second_dir = materialize_variant_markdown(
                markdown_dir,
                root / "second",
                variants[1],
            )
            first_path = variant_page_paths(
                first_dir,
                variants[0],
                materialized_fragments=True,
            )[0]
            second_path = variant_page_paths(
                second_dir,
                variants[1],
                materialized_fragments=True,
            )[0]

            first = first_path.read_text(encoding="utf-8")
            second = second_path.read_text(encoding="utf-8")
            boundary = variants[0].page_fragments[0].end

            assert boundary is not None
            self.assertEqual(first, source[:boundary])
            self.assertEqual(second, source[boundary:])
            self.assertTrue(
                (first_path.parent / "imgs" / "first.jpg").is_file()
            )
            self.assertTrue(
                (second_path.parent / "imgs" / "second.jpg").is_file()
            )


class VariantBoundaryTests(unittest.TestCase):
    def test_task_number_state_is_reset_between_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            normalized_dir = root / "normalized"
            _write_page(
                source,
                1,
                "Часть 1\n1 Первая\nОтвет: ___.\nВторая без номера\nОтвет: ___.",
            )
            _write_page(
                source,
                2,
                "Часть 1\n1 Первая другого варианта\nОтвет: ___.\n"
                "Вторая другого варианта без номера\nОтвет: ___.",
            )

            result_dir = normalize_task_boundaries(
                source,
                normalized_dir,
                page_groups=[(1,), (2,)],
            )
            second = (result_dir / "page_2" / "page_2.md").read_text(
                encoding="utf-8"
            )

        self.assertIn("2. Вторая другого варианта", second)
        self.assertNotIn("3. Вторая другого варианта", second)

    def test_page_must_belong_to_exactly_one_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            _write_page(source, 1, "1 Первая")
            _write_page(source, 2, "2 Вторая")

            with self.assertRaisesRegex(ValueError, "не распределены"):
                normalize_task_boundaries(
                    source,
                    root / "normalized",
                    page_groups=[(1,)],
                )


class MultiVariantCliTests(unittest.TestCase):
    def test_successful_multi_variant_run_removes_only_legacy_root_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            (output_dir / "tasks.xlsx").write_bytes(b"old workbook")
            images_dir = output_dir / "images"
            images_dir.mkdir()
            (images_dir / "task_12.png").write_bytes(b"old image")
            (images_dir / "keep.txt").write_text("keep", encoding="utf-8")

            _remove_legacy_single_result(output_dir)

            self.assertFalse((output_dir / "tasks.xlsx").exists())
            self.assertFalse((images_dir / "task_12.png").exists())
            self.assertTrue((images_dir / "keep.txt").is_file())

    def test_each_variant_is_processed_into_its_own_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            markdown_dir = root / "markdown"
            output_dir = root / "result"
            input_path = root / "document.pdf"
            input_path.write_bytes(b"pdf")
            for page_num, identifier, tasks in (
                (1, "MA2510509", "1 Первая\n19 Последняя"),
                (2, "MA2510510", "1 Первая\n19 Последняя"),
                (3, "MA2510511", "1 Первая\n19 Последняя"),
                (4, "MA2510512", "1 Первая\n19 Последняя"),
            ):
                _write_page(
                    markdown_dir,
                    page_num,
                    f"Вариант {identifier}\nИнструкция по выполнению работы\n{tasks}",
                )

            fake_records = [
                TaskRecord(task_num=str(number), condition="Условие")
                for number in range(1, 20)
            ]
            with (
                patch(
                    "sys.argv",
                    [
                        "main.py",
                        "document.pdf",
                        "--reuse-markdown",
                        "--no-solutions",
                        "--no-answers",
                        "--markdown-dir",
                        str(markdown_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                ),
                patch(
                    "exam_parser.cli.resolve_input_path",
                    return_value=input_path,
                ),
                patch(
                    "exam_parser.cli.repair_markdown_from_pdf",
                    return_value=markdown_dir,
                ),
                patch(
                    "exam_parser.cli.normalize_task_boundaries",
                    return_value=markdown_dir,
                ),
                patch(
                    "exam_parser.cli.process_markdown",
                    return_value=fake_records,
                ) as process,
                redirect_stdout(StringIO()),
            ):
                main()

        self.assertEqual(process.call_count, 4)
        self.assertEqual(
            [call.args[1] for call in process.call_args_list],
            [
                output_dir / "MA2510509",
                output_dir / "MA2510510",
                output_dir / "MA2510511",
                output_dir / "MA2510512",
            ],
        )
        self.assertEqual(
            [
                [path.name for path in call.kwargs["page_paths"]]
                for call in process.call_args_list
            ],
            [["page_1.md"], ["page_2.md"], ["page_3.md"], ["page_4.md"]],
        )

    def test_variants_inside_one_page_use_separate_exact_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            markdown_dir = root / "markdown"
            output_dir = root / "result"
            input_path = root / "document.pdf"
            input_path.write_bytes(b"pdf")
            source = (
                "## Часть 2\nC1 Первая\nC6 Последняя\n"
                "<div>Часть 2</div>\nC1 Другая первая\nC6 Другая последняя"
            )
            _write_page(markdown_dir, 10, source)
            fake_records = [TaskRecord(task_num="C1", condition="Условие")]
            with (
                patch(
                    "sys.argv",
                    [
                        "main.py",
                        "document.pdf",
                        "--reuse-markdown",
                        "--no-solutions",
                        "--no-answers",
                        "--markdown-dir",
                        str(markdown_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                ),
                patch(
                    "exam_parser.cli.resolve_input_path",
                    return_value=input_path,
                ),
                patch(
                    "exam_parser.cli.repair_markdown_from_pdf",
                    return_value=markdown_dir,
                ),
                patch(
                    "exam_parser.cli.normalize_task_boundaries",
                    side_effect=lambda source_dir, target_dir, **kwargs: Path(
                        source_dir
                    ),
                ),
                patch(
                    "exam_parser.cli.process_markdown",
                    return_value=fake_records,
                ) as process,
                redirect_stdout(StringIO()),
            ):
                main()

            first_path = process.call_args_list[0].kwargs["page_paths"][0]
            second_path = process.call_args_list[1].kwargs["page_paths"][0]
            first = first_path.read_text(encoding="utf-8")
            second = second_path.read_text(encoding="utf-8")

        self.assertEqual(process.call_count, 2)
        self.assertNotEqual(first_path.parent.parent, second_path.parent.parent)
        self.assertEqual(first + second, source)
        self.assertIn("C1 Первая", first)
        self.assertNotIn("C1 Другая первая", first)
        self.assertIn("C1 Другая первая", second)

    def test_single_variant_keeps_original_output_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            markdown_dir = root / "markdown"
            output_dir = root / "result"
            input_path = root / "document.pdf"
            input_path.write_bytes(b"pdf")
            _write_page(
                markdown_dir,
                1,
                "Вариант 951\nИнструкция по выполнению работы\n1 Первая",
            )
            fake_records = [TaskRecord(task_num="1", condition="Условие")]
            with (
                patch(
                    "sys.argv",
                    [
                        "main.py",
                        "document.pdf",
                        "--reuse-markdown",
                        "--no-solutions",
                        "--no-answers",
                        "--markdown-dir",
                        str(markdown_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                ),
                patch(
                    "exam_parser.cli.resolve_input_path",
                    return_value=input_path,
                ),
                patch(
                    "exam_parser.cli.repair_markdown_from_pdf",
                    return_value=markdown_dir,
                ),
                patch(
                    "exam_parser.cli.normalize_task_boundaries",
                    return_value=markdown_dir,
                ),
                patch(
                    "exam_parser.cli.process_markdown",
                    return_value=fake_records,
                ) as process,
                redirect_stdout(StringIO()),
            ):
                main()

        self.assertEqual(process.call_count, 1)
        self.assertEqual(process.call_args.args[1], output_dir)


if __name__ == "__main__":
    unittest.main()
