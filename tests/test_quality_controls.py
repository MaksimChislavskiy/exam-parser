from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from exam_parser.deepseek_client import (
    DeepSeekResponseLengthError,
    DeepSeekTaskClient,
)
from exam_parser.markdown_pipeline import (
    OCRQualityError,
    _clean_extracted_task,
    _condition_fidelity_issues,
    _deduplicate_tasks,
    _ensure_condition_fidelity,
    _extract_page_tasks,
    _generate_solutions_and_answers,
    _is_evaluation_example_page,
    _normalize_condition_artifacts,
    _raise_unreadable_ocr_conditions,
    _reconcile_duplicate_task_images,
    _reconcile_lettered_source_tasks,
    _reconcile_verified_page_tasks,
    _recover_missing_expected_tasks,
    _remove_embedded_task_conditions,
    _restore_empty_model_condition,
    _restore_empty_model_task_number,
    _SourceTaskBlock,
    _task_condition_blocks,
    _task_extraction_markdown,
    _verified_condition_blocks,
)
from exam_parser.math_text import normalize_ege_short_answer
from exam_parser.models import (
    MODEL_EMPTY_CONDITION_MARKER,
    MODEL_EMPTY_TASK_NUM_MARKER,
    ExtractedTask,
    SolutionVerification,
    TaskRecord,
    TaskSolution,
)
from exam_parser.ocr_noise import (
    OCR_UNREADABLE_REPEAT_MARKER,
    OCR_VERIFIED_CONDITION_END,
    OCR_VERIFIED_CONDITION_START,
)


class _FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _response(content: str, *, finish_reason: str = "stop") -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=None,
                ),
            )
        ],
        usage=SimpleNamespace(completion_tokens=10, total_tokens=20),
    )


def _client_with_responses(
    responses: list[object],
    *,
    max_solution_chars: int,
) -> tuple[DeepSeekTaskClient, _FakeCompletions]:
    completions = _FakeCompletions(responses)
    client = object.__new__(DeepSeekTaskClient)
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    client.model = "test-model"
    client.max_tokens = 100
    client.max_solution_chars = max_solution_chars
    client.compact_max_tokens = 17
    client.minimal_max_tokens = 9
    client.verify_solutions = True
    return client, completions


class ShortAnswerTests(unittest.TestCase):
    def test_removes_units_and_variable(self) -> None:
        self.assertEqual(normalize_ege_short_answer("1", "45°"), "45")
        self.assertEqual(normalize_ege_short_answer("6", "x=-2"), "-2")
        self.assertEqual(normalize_ege_short_answer("9", "10%"), "10")

    def test_converts_finite_fraction_to_decimal(self) -> None:
        self.assertEqual(normalize_ege_short_answer("4", "1/8"), "0,125")
        self.assertEqual(
            normalize_ege_short_answer("4", r"$\frac{3}{20}$"),
            "0,15",
        )

    def test_rejects_nonterminating_fraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "не является конечной"):
            normalize_ege_short_answer("4", "1/3")

    def test_does_not_change_second_part_answer(self) -> None:
        answer = "А) нет; Б) да; В) 6"
        self.assertEqual(normalize_ege_short_answer("19", answer), answer)

    def test_normalizes_latin_second_part_labels(self) -> None:
        answer = "a) да;\nb) нет;\nc) 931"
        self.assertEqual(
            normalize_ege_short_answer("19", answer),
            "а) да;\nб) нет;\nв) 931",
        )

    def test_does_not_treat_single_latin_marker_as_subpart_sequence(self) -> None:
        answer = "a) — значение параметра"
        self.assertEqual(normalize_ege_short_answer("18", answer), answer)


class LengthLimitedPageRecoveryTests(unittest.TestCase):
    class _Client:
        provider_name = "Test"

        def __init__(self, responses: list[object]) -> None:
            self.responses = responses
            self.calls: list[tuple[str, list[str]]] = []

        def extract_markdown(
            self,
            markdown: str,
            image_ids: list[str],
        ) -> list[ExtractedTask]:
            self.calls.append((markdown, image_ids))
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response  # type: ignore[return-value]

    def test_splits_only_by_unique_source_task_boundaries(self) -> None:
        client = self._Client(
            [
                DeepSeekResponseLengthError("length"),
                [ExtractedTask(task_num="1", condition="Условие 1.")],
                [
                    ExtractedTask(
                        task_num="2",
                        condition="Условие 2.",
                        image_id="task_2.png",
                    )
                ],
            ]
        )

        result = _extract_page_tasks(
            client,
            "1. Условие 1.\n\n2. Условие 2.",
            ["task_2.png", "service.png"],
            source_by_task={"1": "Условие 1.", "2": "Условие 2."},
            image_by_task={"2": "task_2.png"},
            page_num=28,
        )

        self.assertEqual([task.task_num for task in result], ["1", "2"])
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(client.calls[1], ("1. Условие 1.", []))
        self.assertEqual(
            client.calls[2],
            ("2. Условие 2.", ["task_2.png"]),
        )

    def test_single_source_task_does_not_make_second_paid_request(self) -> None:
        client = self._Client([DeepSeekResponseLengthError("length")])

        result = _extract_page_tasks(
            client,
            "13. Длинное условие.",
            ["diagram.png"],
            source_by_task={"13": "Длинное условие."},
            image_by_task={"13": "diagram.png"},
            page_num=5,
        )

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result[0].condition, "Длинное условие.")
        self.assertEqual(result[0].image_id, "diagram.png")

    def test_isolated_length_uses_exact_source_without_third_request(self) -> None:
        client = self._Client(
            [
                DeepSeekResponseLengthError("page length"),
                DeepSeekResponseLengthError("task length"),
                [ExtractedTask(task_num="2", condition="Условие 2.")],
            ]
        )

        result = _extract_page_tasks(
            client,
            "1. Условие 1.\n\n2. Условие 2.",
            [],
            source_by_task={"1": "Условие 1.", "2": "Условие 2."},
            image_by_task={},
            page_num=28,
        )

        self.assertEqual(len(client.calls), 3)
        self.assertEqual(result[0].condition, "Условие 1.")
        self.assertEqual(result[1].condition, "Условие 2.")

    def test_refuses_split_when_task_numbers_repeat(self) -> None:
        client = self._Client([DeepSeekResponseLengthError("length")])

        with self.assertRaisesRegex(
            OCRQualityError,
            "уникальные OCR-границы.*не найдены",
        ):
            _extract_page_tasks(
                client,
                "1. Первый вариант.\n\n1. Второй вариант.",
                [],
                source_by_task={"1": "Второй вариант."},
                image_by_task={},
                page_num=7,
            )

        self.assertEqual(len(client.calls), 1)


class EvaluationExamplePageTests(unittest.TestCase):
    def test_detects_36169_style_expert_evaluation_page(self) -> None:
        markdown = (
            "Пример 13.1.1\n\n"
            '<img src="imgs/solution.jpg" />\n\n'
            "Доказательство утверждения в пункте а не обосновано.\n\n"
            "С использованием утверждения пункта а верно получен ответ.\n\n"
            "Оценка эксперта: 1 балл."
        )

        self.assertTrue(_is_evaluation_example_page(markdown, {}))

    def test_does_not_skip_unnumbered_task_containing_example(self) -> None:
        markdown = (
            "Пример использования функции приведён на рисунке.\n"
            "Найдите значение параметра."
        )

        self.assertFalse(_is_evaluation_example_page(markdown, {}))

    def test_strict_task_heading_prevents_skip(self) -> None:
        markdown = (
            "13. Докажите утверждение.\n\n"
            "Пример 13.1.1\n\n"
            "Комментарий.\nОценка эксперта: 1 балл."
        )

        self.assertFalse(
            _is_evaluation_example_page(
                markdown,
                {"13": "Докажите утверждение."},
            )
        )


class VerifiedOCRConditionTests(unittest.TestCase):
    def test_verified_condition_rejects_only_low_quality_neighbour_headings(
        self,
    ) -> None:
        verified_condition = (
            "15 декабря планируется взять кредит на 17 месяцев. "
            "Какую сумму планируется взять в кредит?"
        )
        markdown = (
            f"{OCR_VERIFIED_CONDITION_START}\n"
            f"17. {verified_condition}\n"
            f"{OCR_VERIFIED_CONDITION_END}\n\n"
            "12. $x^2-y^2=10a-24$\n\n"
            "Bce a, uode uocuouuuu4peneue\n\n"
            "19. Централическая источная, внуклетого умомента 吋"
        )
        source_by_task = _task_condition_blocks(markdown)
        verified_by_task = _verified_condition_blocks(markdown)
        tasks = [
            ExtractedTask(task_num="12", condition="Ложный фрагмент 12"),
            ExtractedTask(task_num="17", condition="Изменённое условие"),
            ExtractedTask(task_num="19", condition="Ложный фрагмент 19"),
        ]

        result = _reconcile_verified_page_tasks(
            tasks,
            source_by_task,
            verified_by_task,
            provider_name="Test",
            page_num=2,
        )

        self.assertEqual([task.task_num for task in result], ["17"])
        self.assertEqual(result[0].condition, verified_condition)

    def test_verified_condition_ends_before_following_solution(self) -> None:
        verified_condition = (
            "Дано трёхзначное число $A$ и сумма его цифр $S$.\n\n"
            "а) Может ли $A\\cdot S=1105$?\n"
            "б) Может ли $A\\cdot S=1106$?"
        )
        markdown = (
            "## Задание 19\n\n19 номер\n\n"
            f"{OCR_VERIFIED_CONDITION_START}\n"
            f"19. {verified_condition}\n"
            f"{OCR_VERIFIED_CONDITION_END}\n\n"
            "а) $A\\cdot S=1105=5\\cdot221$\n"
            "б) Вычисления решения"
        )

        source_by_task = _task_condition_blocks(markdown)

        self.assertEqual(source_by_task["19"], verified_condition)
        self.assertNotIn("221", source_by_task["19"])
        extraction_markdown = _task_extraction_markdown(markdown)
        self.assertNotIn(OCR_VERIFIED_CONDITION_START, extraction_markdown)
        self.assertNotIn(OCR_VERIFIED_CONDITION_END, extraction_markdown)

    def test_missing_verified_task_is_restored_without_model_retry(self) -> None:
        result = _reconcile_verified_page_tasks(
            [],
            {"19": "Проверенное условие"},
            {"19": "Проверенное условие"},
            provider_name="Test",
            page_num=8,
        )

        self.assertEqual(
            [(task.task_num, task.condition) for task in result],
            [("19", "Проверенное условие")],
        )


class ConditionFidelityTests(unittest.TestCase):
    def test_extraction_markdown_drops_only_heading_metadata(self) -> None:
        markdown = (
            "### Задачи №15. Условия\n\n"
            "№15.1 (Дальний восток)\n\n"
            "Решите неравенство $x>0$.\n\n"
            "15.2. (а) Решите уравнение $x=1$.\n"
        )

        prepared = _task_extraction_markdown(markdown)

        self.assertIn("15.1.\n\nРешите неравенство", prepared)
        self.assertNotIn("Дальний восток", prepared)
        self.assertIn("15.2. (а) Решите уравнение", prepared)

    def test_unreadable_ocr_marker_uses_source_without_paid_retry(self) -> None:
        source = f"Условие {OCR_UNREADABLE_REPEAT_MARKER}"
        changed = ExtractedTask(
            task_num="19",
            condition="Модель попыталась восстановить условие",
            image_id="task.jpg",
        )

        class _NoRetryClient:
            provider_name = "Test"

            def extract_markdown(
                self,
                markdown: str,
                image_ids: list[str],
            ) -> list[ExtractedTask]:
                raise AssertionError("изолированный повтор не должен вызываться")

        result = _ensure_condition_fidelity(
            _NoRetryClient(),
            changed,
            source,
        )

        self.assertEqual(result.condition, source)
        self.assertEqual(result.image_id, "task.jpg")

    def test_detects_omitted_single_letter_prose_word(self) -> None:
        source = (
            "Плотность жидкости равна $\\rho=1000$, а $g=9,8$ — "
            "ускорение свободного падения."
        )
        candidate = (
            "Плотность жидкости равна $\\rho=1000$, $g=9,8$ — "
            "ускорение свободного падения."
        )

        self.assertIn(
            "изменен текст: утрачено а",
            _condition_fidelity_issues(source, candidate),
        )

    def test_rejects_twice_repeated_single_letter_word_omission(self) -> None:
        source = (
            "Плотность жидкости равна $\\rho=1000$, а $g=9,8$ — "
            "ускорение свободного падения."
        )
        changed = ExtractedTask(
            task_num="9",
            condition=(
                "Плотность жидкости равна $\\rho=1000$, $g=9,8$ — "
                "ускорение свободного падения."
            ),
        )
        client = SimpleNamespace(
            provider_name="Test",
            extract_markdown=lambda markdown, image_ids: [changed],
        )

        result = _ensure_condition_fidelity(client, changed, source)

        self.assertEqual(result.condition, source)

    def test_spelling_fix_cannot_hide_single_letter_word_omission(self) -> None:
        source = "Рассмотрите функию, а затем найдите её значение."
        changed = ExtractedTask(
            task_num="8",
            condition="Рассмотрите функцию, затем найдите её значение.",
        )
        client = SimpleNamespace(
            provider_name="Test",
            extract_markdown=lambda markdown, image_ids: [changed],
        )

        result = _ensure_condition_fidelity(client, changed, source)

        self.assertEqual(result.condition, source)

    def test_ignores_single_letter_inside_latex_as_prose(self) -> None:
        source = "Пусть $а=1$. Найдите значение выражения."
        candidate = "Пусть $a=1$. Найдите значение выражения."

        issues = _condition_fidelity_issues(source, candidate)

        self.assertFalse(
            any(issue.startswith("изменен текст:") for issue in issues)
        )

    def test_separately_checks_unchanged_two_letter_angle_notation(self) -> None:
        source = (
            "В тупоугольном треугольнике $ ABC $ угол C тупой. "
            "Докажите, что острые углы $ AB $ и $ ACH $ равны."
        )
        task = ExtractedTask(task_num="17", condition=source)

        class _AngleCheckClient:
            provider_name = "Test"

            def __init__(self) -> None:
                self.marked_conditions: list[str] = []

            def check_angle_notation(self, marked_condition: str) -> object:
                self.marked_conditions.append(marked_condition)
                return SimpleNamespace(corrected_notation="ABC")

        client = _AngleCheckClient()

        result = _ensure_condition_fidelity(client, task, source)

        self.assertEqual(
            result.condition,
            "В тупоугольном треугольнике $ ABC $ угол $C$ тупой. "
            "Докажите, что острые углы $ ABC $ и $ ACH $ равны.",
        )
        self.assertEqual(len(client.marked_conditions), 2)
        self.assertTrue(
            all(
                "<angle_to_check>AB</angle_to_check>" in condition
                for condition in client.marked_conditions
            )
        )

    def test_checks_angle_after_accepting_unrelated_spelling_fix(self) -> None:
        source = (
            "Рассмотрите функию. Докажите, что углы $PR$ и $QST$ равны."
        )
        corrected = ExtractedTask(
            task_num="14",
            condition=(
                "Рассмотрите функцию. Докажите, что углы $PR$ и $QST$ равны."
            ),
        )

        class _SpellingAndAngleClient:
            provider_name = "Test"

            def extract_markdown(
                self,
                markdown: str,
                image_ids: list[str],
            ) -> list[ExtractedTask]:
                return [corrected]

            def check_angle_notation(self, marked_condition: str) -> object:
                return SimpleNamespace(corrected_notation="PQR")

        result = _ensure_condition_fidelity(
            _SpellingAndAngleClient(),
            corrected,
            source,
        )

        self.assertEqual(
            result.condition,
            "Рассмотрите функцию. Докажите, что углы $PQR$ и $QST$ равны.",
        )

    def test_preserves_angle_when_separate_checks_disagree(self) -> None:
        source = "Докажите, что углы $PR$ и $QST$ равны."
        task = ExtractedTask(task_num="14", condition=source)
        replies = iter(("PQR", "PSR"))
        client = SimpleNamespace(
            provider_name="Test",
            check_angle_notation=lambda marked_condition: SimpleNamespace(
                corrected_notation=next(replies)
            ),
        )

        result = _ensure_condition_fidelity(client, task, source)

        self.assertEqual(result.condition, source)

    def test_preserves_angle_when_separate_check_fails(self) -> None:
        source = "Докажите, что углы $PR$ и $QST$ равны."
        task = ExtractedTask(task_num="14", condition=source)

        def fail(marked_condition: str) -> object:
            raise RuntimeError("service unavailable")

        client = SimpleNamespace(
            provider_name="Test",
            check_angle_notation=fail,
        )

        result = _ensure_condition_fidelity(client, task, source)

        self.assertEqual(result.condition, source)

    def test_does_not_check_two_letter_side_name_as_angle(self) -> None:
        source = "В треугольнике $ABC$ сторона $AB$ равна 5."
        task = ExtractedTask(task_num="3", condition=source)
        client = SimpleNamespace(provider_name="Test")

        result = _ensure_condition_fidelity(client, task, source)

        self.assertEqual(result.condition, source)

    def test_detects_reordered_angle_letters(self) -> None:
        issues = _condition_fidelity_issues(
            "Найдите угол АСВ, если А1В1 = 4.",
            "Найдите угол $ABC$, если $A_1B_1=4$.",
        )
        self.assertTrue(any("ACB" in issue for issue in issues))

    def test_detects_replaced_point_in_triangle(self) -> None:
        issues = _condition_fidelity_issues(
            "Найдите площадь треугольника APQ.",
            "Найдите площадь треугольника $APO$.",
        )
        self.assertTrue(any("APQ" in issue for issue in issues))

    def test_accepts_cyrillic_and_latin_confusables(self) -> None:
        self.assertEqual(
            _condition_fidelity_issues(
                "В треугольнике АВС сторона АВ = 2.",
                "В треугольнике $ABC$ сторона $AB=2$.",
            ),
            [],
        )

    def test_does_not_retry_equivalent_display_formula(self) -> None:
        source = (
            "а) Решите уравнение\n\n"
            r"$$2\cos^{2}x-\sin\left(x-\pi\right)-1=0$$"
            "\n\nб) Найдите корни на отрезке $[-1;1]$."
        )
        extracted = ExtractedTask(
            task_num="13.6",
            condition=(
                "<p>а) Решите уравнение "
                r"$2\cos^{2}x-\sin\left(x-\pi\right)-1=0$</p>"
                "\n<p>б) Найдите корни на отрезке $[-1;1]$.</p>"
            ),
        )
        client = SimpleNamespace(
            provider_name="Test",
            extract_markdown=lambda *_: self.fail("unexpected retry"),
        )

        result = _ensure_condition_fidelity(client, extracted, source)

        self.assertEqual(result, extracted)

    def test_accepts_ocr_spaces_around_geometry_indices(self) -> None:
        self.assertEqual(
            _condition_fidelity_issues(
                "Проведены высоты АА 1 и ВВ 1. Известно, что А 1 В 1 = 4.",
                "Проведены высоты $AA_1$ и $BB_1$. Известно, что $A_1B_1=4$.",
            ),
            [],
        )

    def test_accepts_cyrillic_d_in_latin_geometry_name(self) -> None:
        self.assertEqual(
            _condition_fidelity_issues(
                "Трапеция ABCД имеет основания BC и AD.",
                "Трапеция $ABCD$ имеет основания $BC$ и $AD$.",
            ),
            [],
        )

    def test_detects_added_geometry_letter(self) -> None:
        issues = _condition_fidelity_issues(
            "В призме ABCA1B1C1 проведена плоскость.",
            "В призме $ABCDA_1B_1C_1$ проведена плоскость.",
        )

        self.assertTrue(issues)

    def test_detects_replaced_russian_words(self) -> None:
        issues = _condition_fidelity_issues(
            "В треугольнике ABC угол C тупой. Сумма выплат указана в млн рублей.",
            "В треугольнике $ABC$ утопил C тупой. Сумма вышлат указана в мин рублей.",
        )

        self.assertTrue(any("текст" in issue for issue in issues))

    def test_accepts_yo_and_e_as_the_same_russian_letter(self) -> None:
        self.assertEqual(
            _condition_fidelity_issues(
                "На рисунке изображён график.",
                "На рисунке изображен график.",
            ),
            [],
        )

    def test_detects_changed_formula_variables(self) -> None:
        issues = _condition_fidelity_issues(
            r"Сила равна $F_A=\alpha\rho gr^3$.",
            r"Сила равна $F_A=\frac{a}{r}r^3$.",
        )

        self.assertTrue(any("формул" in issue for issue in issues))

    def test_detects_added_derivative_prime(self) -> None:
        issues = _condition_fidelity_issues(
            r"Найдите промежутки возрастания функции $f(x)$.",
            r"Найдите промежутки возрастания функции $f'(x)$.",
        )

        self.assertTrue(any("формул" in issue for issue in issues))

    def test_detects_changed_coordinate_separator(self) -> None:
        issues = _condition_fidelity_issues(
            r"Дан вектор $a(-1;-3)$.",
            r"Дан вектор $a(-1,-3)$.",
        )

        self.assertTrue(any("формул" in issue for issue in issues))

    def test_detects_number_added_by_misread_subpart_label(self) -> None:
        issues = _condition_fidelity_issues(
            "а) Первый вопрос. б) Второй вопрос.",
            "a) Первый вопрос. 6) Второй вопрос.",
        )

        self.assertTrue(any("числа" in issue for issue in issues))

    def test_detects_dropped_sentence_terminator(self) -> None:
        issues = _condition_fidelity_issues(
            "а) Докажите равенство. б) Найдите длину.",
            "a) Докажите равенство б) Найдите длину.",
        )

        self.assertTrue(any("пунктуац" in issue for issue in issues))

    def test_falls_back_to_source_when_retry_repeats_corruption(self) -> None:
        source = r"Сила равна $F_A=\alpha\rho gr^3$."
        corrupted = ExtractedTask(
            task_num="9",
            condition=r"Сила равна $F_A=\frac{a}{r}r^3$.",
        )
        client = SimpleNamespace(
            provider_name="Test",
            extract_markdown=lambda markdown, image_ids: [corrupted],
        )

        result = _ensure_condition_fidelity(client, corrupted, source)

        self.assertEqual(result.condition, source)

    def test_stops_when_isolated_retry_finds_tasks_without_numbers(self) -> None:
        source = (
            "$9^{x-1} = 81$\n\n"
            "Угол DBC равен 36 градусам. Найдите угол BAD."
        )
        extracted = ExtractedTask(task_num="5", condition="$9^{x-1} = 81$")
        client = SimpleNamespace(
            provider_name="Test",
            extract_markdown=lambda markdown, image_ids: [
                extracted,
                ExtractedTask(
                    task_num=MODEL_EMPTY_TASK_NUM_MARKER,
                    condition="Угол DBC равен 36 градусам. Найдите угол BAD.",
                ),
            ],
        )

        with self.assertRaisesRegex(
            OCRQualityError,
            "изолированной проверке задачи 5.*без task_num: 1",
        ):
            _ensure_condition_fidelity(client, extracted, source)

    def test_accepts_obvious_text_only_ocr_correction(self) -> None:
        source = "В основании ширамиды лежит треугольник."
        corrected = ExtractedTask(
            task_num="3",
            condition="В основании пирамиды лежит треугольник.",
        )
        client = SimpleNamespace(
            provider_name="Test",
            extract_markdown=lambda markdown, image_ids: [corrected],
        )

        result = _ensure_condition_fidelity(client, corrected, source)

        self.assertEqual(result.condition, corrected.condition)

    def test_accepts_twice_confirmed_single_character_spelling_fix(self) -> None:
        source = "Исследуйте функию на монотонность."
        corrected = ExtractedTask(
            task_num="8",
            condition="Исследуйте функцию на монотонность.",
        )
        client = SimpleNamespace(
            provider_name="Test",
            extract_markdown=lambda markdown, image_ids: [corrected],
        )

        result = _ensure_condition_fidelity(client, corrected, source)

        self.assertEqual(result.condition, corrected.condition)

    def test_accepts_twice_confirmed_missing_angle_vertex(self) -> None:
        source = (
            "В тупоугольном треугольнике $ ABC $ угол C тупой. "
            "Докажите, что острые углы $ AB $ и $ ACH $ равны."
        )
        corrected = ExtractedTask(
            task_num="17",
            condition=(
                "В тупоугольном треугольнике $ABC$ угол C тупой. "
                "Докажите, что острые углы $ABC$ и $ACH$ равны."
            ),
        )
        client = SimpleNamespace(
            provider_name="Test",
            extract_markdown=lambda markdown, image_ids: [corrected],
        )

        result = _ensure_condition_fidelity(client, corrected, source)

        self.assertEqual(
            result.condition,
            (
                "В тупоугольном треугольнике $ABC$ угол $C$ тупой. "
                "Докажите, что острые углы $ABC$ и $ACH$ равны."
            ),
        )

    def test_accepts_missing_middle_angle_vertex_for_other_labels(self) -> None:
        source = "Докажите, что углы $PR$ и $QST$ равны."
        corrected = ExtractedTask(
            task_num="14",
            condition="Докажите, что углы $PQR$ и $QST$ равны.",
        )
        client = SimpleNamespace(
            provider_name="Test",
            extract_markdown=lambda markdown, image_ids: [corrected],
        )

        result = _ensure_condition_fidelity(client, corrected, source)

        self.assertEqual(result.condition, corrected.condition)

    def test_does_not_guess_missing_angle_vertex_without_confirmation(
        self,
    ) -> None:
        condition = "Докажите, что углы $AB$ и $ACH$ равны."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="17"),
            condition,
        )

    def test_rejects_twice_confirmed_added_letter_in_side_name(self) -> None:
        source = "В треугольнике $ABC$ сторона $AB$ равна 5."
        changed = ExtractedTask(
            task_num="17",
            condition="В треугольнике $ABC$ сторона $ABC$ равна 5.",
        )
        client = SimpleNamespace(
            provider_name="Test",
            extract_markdown=lambda markdown, image_ids: [changed],
        )

        result = _ensure_condition_fidelity(client, changed, source)

        self.assertEqual(result.condition, source)

    def test_rejects_twice_confirmed_replaced_angle_points(self) -> None:
        source = "Докажите, что углы $AB$ и $ACH$ равны."
        changed = ExtractedTask(
            task_num="17",
            condition="Докажите, что углы $ACD$ и $ACH$ равны.",
        )
        client = SimpleNamespace(
            provider_name="Test",
            extract_markdown=lambda markdown, image_ids: [changed],
        )

        result = _ensure_condition_fidelity(client, changed, source)

        self.assertEqual(result.condition, source)

    def test_rejects_twice_repeated_meaning_sensitive_word_change(self) -> None:
        source = "Укажите длину наибольшого из промежутков."
        changed = ExtractedTask(
            task_num="8",
            condition="Укажите длину наибольшего из промежутков.",
        )
        client = SimpleNamespace(
            provider_name="Test",
            extract_markdown=lambda markdown, image_ids: [changed],
        )

        result = _ensure_condition_fidelity(client, changed, source)

        self.assertEqual(result.condition, source)

    def test_rejects_twice_repeated_number_change(self) -> None:
        source = "Сила не превосходит 264 600 Н."
        changed = ExtractedTask(
            task_num="9",
            condition="Сила не превосходит 264600 Н.",
        )
        client = SimpleNamespace(
            provider_name="Test",
            extract_markdown=lambda markdown, image_ids: [changed],
        )

        result = _ensure_condition_fidelity(client, changed, source)

        self.assertEqual(result.condition, source)

    def test_repairs_real_ocr_defects_from_all_four_variants(self) -> None:
        cases = [
            (
                "0509-3",
                "В основании ширамиды SABC. Объём ширамиды равен 24.",
                ("пирамиды",),
                ("ширамиды",),
            ),
            (
                "0509-8",
                (
                    "На рисунке изображен график функции $y=f'(x)$ — "
                    "производной функции $f(x)$. Найдите промежутки "
                    "возрастания функции $f'(x)$."
                ),
                ("возрастания функции $f(x)$",),
                ("возрастания функции $f'(x)$",),
            ),
            (
                "0509-9",
                (
                    "Аппарат имеет форму сферы, выталкивающая (архимедова) "
                    "сила определяется по формуле $F_A=\\frac{a}{r}r^3$, "
                    "где a = 4, 2 — постоянная, r — радиус, $\\rho=1000$, "
                    "a = 10 Н/кг — ускорение свободного падения."
                ),
                (r"F_A=\alpha\rho gr^3", r"$\alpha=4,2$", r"$g=10$"),
                (r"\frac{a}{r}", "a = 4, 2", "a = 10"),
            ),
            (
                "0509-14",
                (
                    "В правильной треугольной призме $ABC_1B_1C_1$ "
                    "проведена плоскость. Сечением будет трапения."
                ),
                (r"$ABCA_1B_1C_1$", "трапеция"),
                (r"$ABC_1B_1C_1$", "трапения"),
            ),
            (
                "0509-16",
                "<td>Долг (в мн рублей)</td>",
                ("в млн рублей",),
                ("в мн рублей",),
            ),
            (
                "0510-4",
                "Взятая из стопки тетрады окажется в косую линейку.",
                ("тетрадь",),
                ("тетрады",),
            ),
            (
                "0510-2",
                (
                    "Даны векторы $\\vec{a}(-1,-3)$, "
                    "$\\vec{b}(5,-3)$ и $\\vec{c}(1,6)$."
                ),
                ("(-1;-3)", "(5;-3)", "(1;6)"),
                ("(-1,-3)", "(5,-3)", "(1,6)"),
            ),
            (
                "0510-8",
                "Функция определена на интервале $(-15, 4)$.",
                ("$(-15;4)$",),
                ("$(-15, 4)$",),
            ),
            (
                "0510-9",
                (
                    "Аппарат имеет форму сферы, выталкивающая (архимедова) "
                    "сила определяется по формуле $F_A=\\frac{a}{r_gr^3}$, "
                    "где a = 3,1 — постоянная, r — радиус, $\\rho=1000$, "
                    "a = 10 Н/кг — ускорение свободного падения."
                ),
                (r"F_A=\alpha\rho gr^3", r"$\alpha=3,1$", r"$g=10$"),
                (r"\frac{a}{r_gr^3}", "a = 3,1", "a = 10"),
            ),
            (
                "0510-14",
                "В правильной треугольной призме $ABCDA_1B_1C_1$.",
                (r"$ABCA_1B_1C_1$",),
                (r"$ABCDA_1B_1C_1$",),
            ),
            (
                "0510-16",
                (
                    "<td>Долг (в мин рублей)</td> Общая сумма вышлат "
                    "по кредиту."
                ),
                ("в млн рублей", "сумма выплат"),
                ("в мин рублей", "сумма вышлат"),
            ),
            (
                "0510-17",
                (
                    "В треугольнике $ABC$ утопил С тупой. "
                    "Докажите, что острые углы $AB$ и $ACH$ равны."
                ),
                ("угол $C$ тупой",),
                ("утопил", "угол С"),
            ),
            (
                "0510-19",
                (
                    "<p>а) Может ли получиться число 3?  "
                    "6) Может ли получиться число 17,5?</p>\n"
                    "<p>в) Какое число наибольшее?</p>"
                ),
                ("<p>а) Может ли получиться число 3?</p>",
                 "<p>б) Может ли получиться число 17,5?</p>"),
                ("6)",),
            ),
            (
                "0511-9",
                "где $\\rho=1000$, $a$ $g = 9,8$ Н/кг — ускорение.",
                ("а $g = 9,8$",),
                ("$a$ $g",),
            ),
            (
                "0511-14",
                "В прямой треугольной призме $ABC_1B_1C_1$.",
                (r"$ABCA_1B_1C_1$",),
                (r"$ABC_1B_1C_1$",),
            ),
            (
                "0512-14",
                "Сечением призмы будет прямоугольная трапейка.",
                ("прямоугольная трапеция",),
                ("трапейка",),
            ),
        ]

        for name, source, expected, forbidden in cases:
            with self.subTest(case=name):
                normalized = _normalize_condition_artifacts(
                    source,
                    task_num=name.split("-")[1],
                )
                for fragment in expected:
                    self.assertIn(fragment, normalized)
                for fragment in forbidden:
                    self.assertNotIn(fragment, normalized)

    def test_coordinate_cleanup_does_not_change_decimal_fraction(self) -> None:
        condition = r"Найдите корень уравнения $(0,125)^{x+5}=4^{x+4}$."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="6"),
            condition,
        )

    def test_coordinate_cleanup_does_not_change_spaced_decimal_fraction(self) -> None:
        condition = "Вероятность равна $(0, 125)$ при указанном условии."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="5"),
            condition,
        )

    def test_buoyancy_cleanup_keeps_other_valid_sphere_formula(self) -> None:
        condition = (
            "Тело имеет форму сферы, архимедова сила определяется по формуле "
            r"$F_A=\frac{4}{3}\pi\rho gr^3$."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="9"),
            condition,
        )

    def test_extracts_clean_source_block(self) -> None:
        blocks = _task_condition_blocks(
            '1. Найдите угол АСВ.\n<img src="imgs/one.jpg" />\n'
            'Ответ: __________\n2. Найдите число 5.\nОтвет: ___'
        )
        self.assertEqual(blocks["1"], "Найдите угол АСВ.")
        self.assertEqual(blocks["2"], "Найдите число 5.")

    def test_extracts_legacy_a_task_source_block(self) -> None:
        blocks = _task_condition_blocks(
            "А1 Найдите значение выражения.\n"
            "Решение: вычислим выражение.\n"
            "А2 Найдите число."
        )

        self.assertEqual(blocks["A1"], "Найдите значение выражения.")
        self.assertEqual(blocks["A2"], "Найдите число.")

    def test_restores_legacy_a8_heading_misread_as_as(self) -> None:
        blocks = _task_condition_blocks(
            "A7 Найдите значение функции.\n\n"
            "AS Решите неравенство.\n\n"
            "A9 Решите уравнение."
        )

        self.assertEqual(blocks["A7"], "Найдите значение функции.")
        self.assertEqual(blocks["A8"], "Решите неравенство.")
        self.assertEqual(blocks["A9"], "Решите уравнение.")

    def test_stops_condition_at_next_lettered_section_instruction(self) -> None:
        blocks = _task_condition_blocks(
            "A10 Решите неравенство.\n\n"
            "Ответом на задания B1-B11 должно быть некоторое число.\n\n"
            "B1 Найдите значение выражения.\n\n"
            "C2 Решите систему.\n\n"
            "## ЧАСТЬ 3\n\n"
            "Для записи ответов на задания C3-C5 используйте бланк №2.\n\n"
            "C3 Докажите утверждение."
        )

        self.assertEqual(blocks["A10"], "Решите неравенство.")
        self.assertEqual(blocks["C2"], "Решите систему.")

    def test_cached_condition_stops_at_next_lettered_section(self) -> None:
        task = _clean_extracted_task(
            ExtractedTask(
                task_num="A10",
                condition=(
                    "Решите неравенство.\n\n"
                    "Ответом на задания B1-B11 должно быть некоторое число."
                ),
            )
        )

        self.assertEqual(task.condition, "Решите неравенство.")

    def test_removes_explicit_solution_from_source_condition(self) -> None:
        blocks = _task_condition_blocks(
            "B1\nНайдите значение выражения.\n\n"
            "Решение: выполним вычисления.\n\n"
            "Записать ответ 5."
        )

        self.assertEqual(blocks["B1"], "Найдите значение выражения.")

    def test_uses_first_paragraph_before_unlabeled_worked_solution(self) -> None:
        blocks = _task_condition_blocks(
            "B2\nНайдите точку касания.\n\n"
            "Значение производной равно коэффициенту касательной.\n\n"
            "Записать ответ -6."
        )

        self.assertEqual(blocks["B2"], "Найдите точку касания.")

    def test_removes_same_paragraph_solution_transition(self) -> None:
        blocks = _task_condition_blocks(
            "B4\nВычислите площадь фигуры. Построим графики этих функций.\n\n"
            "Записать ответ 30."
        )

        self.assertEqual(blocks["B4"], "Вычислите площадь фигуры.")

    def test_removes_unlabeled_solution_paragraph_without_answer_field(self) -> None:
        blocks = _task_condition_blocks(
            "C2\nНайдите наименьшую площадь треугольника.\n\n"
            "Построим систему координат и выполним вычисления."
        )

        self.assertEqual(
            blocks["C2"],
            "Найдите наименьшую площадь треугольника.",
        )

    def test_keeps_only_first_system_when_ocr_appends_its_solution(self) -> None:
        blocks = _task_condition_blocks(
            "C1\nРешите систему уравнений\n\n"
            r"$$ \begin{aligned}&\left\{\begin{aligned}"
            r"&x+y=2,\\&x-y=0.\end{aligned}\right.\\"
            r"&x=1\Rightarrow y=1.\end{aligned} $$"
            "\n\nРешением будет пара (1; 1)."
        )

        self.assertEqual(
            blocks["C1"],
            "Решите систему уравнений\n\n"
            r"$$ \left\{\begin{aligned}&x+y=2,\\&x-y=0."
            r"\end{aligned}\right. $$",
        )

    def test_keeps_legitimate_multi_system_condition_without_solution_evidence(self) -> None:
        source = (
            "C1\nРешите систему уравнений\n\n"
            r"$$ \left\{\begin{aligned}x+y&=2\\x-y&=0"
            r"\end{aligned}\right. $$"
        )

        blocks = _task_condition_blocks(source)

        self.assertIn(r"x+y&=2", blocks["C1"])

    def test_removes_unlabeled_multiple_choice_solution(self) -> None:
        blocks = _task_condition_blocks(
            "A8\nУкажите убывающую функцию.\n\n"
            "1) $y=2^x$\n\n2) $y=3^x$\n\n"
            "3) $y=0.5^{-x}$\n\n4) $y=0.5^x$\n\n"
            "Показательная функция убывает при основании от нуля до единицы.\n\n"
            "Верный ответ 4)."
        )

        self.assertEqual(
            blocks["A8"],
            "Укажите убывающую функцию.\n\n"
            "1) $y=2^x$\n\n2) $y=3^x$\n\n"
            "3) $y=0.5^{-x}$\n\n4) $y=0.5^x$",
        )

    def test_keeps_multiple_choice_options_without_correct_answer_tail(self) -> None:
        source = (
            "A2\nНайдите значение выражения.\n"
            "1) 1 2) 2 3) 3 4) 4"
        )

        blocks = _task_condition_blocks(source)

        self.assertIn("4) 4", blocks["A2"])

    def test_removes_checkbox_and_repeated_task_number(self) -> None:
        blocks = _task_condition_blocks(
            "2. ☐ 2 Даны три вектора.\nОтвет: ___"
        )

        self.assertEqual(blocks["2"], "Даны три вектора.")

    def test_keeps_legitimate_number_at_start_of_condition(self) -> None:
        blocks = _task_condition_blocks(
            "2. 2 рабочих выполняют заказ.\nОтвет: ___"
        )

        self.assertEqual(blocks["2"], "2 рабочих выполняют заказ.")

    def test_replaces_markdown_nonbreaking_space_before_formula(self) -> None:
        blocks = _task_condition_blocks(
            r"15. Решите неравенство~$2^x\leq2$."
        )

        self.assertEqual(blocks["15"], r"Решите неравенство $2^x\leq2$.")

    def test_extracts_legacy_and_late_numeric_source_blocks(self) -> None:
        blocks = _task_condition_blocks(
            "## 20 Найдите параметр.\n"
            "21 Докажите утверждение.\n"
            "23.06.14 Образец варианта Часть С\n"
            "C1 Решите уравнение.\n"
            "С2 Найдите расстояние.\n"
        )

        self.assertEqual(blocks["20"], "Найдите параметр.")
        self.assertEqual(blocks["21"], "Докажите утверждение.\n23.06.14 Образец варианта Часть С")
        self.assertEqual(blocks["C1"], "Решите уравнение.")
        self.assertEqual(blocks["C2"], "Найдите расстояние.")
        self.assertNotIn("23.06", blocks)

    def test_extracts_boxed_legacy_source_block(self) -> None:
        blocks = _task_condition_blocks(
            r"$$ \boxed{\mathrm{C3}}\quad\mathrm{Решите неравенство~}"
            "$x>0$. $$\nC4 Найдите радиус."
        )

        self.assertIn("Решите неравенство", blocks["C3"])
        self.assertEqual(blocks["C4"], "Найдите радиус.")

    def test_canonicalizes_cyrillic_legacy_task_number(self) -> None:
        task = ExtractedTask(task_num="С6", condition="Найдите число.")

        self.assertEqual(_clean_extracted_task(task).task_num, "C6")

    def test_canonicalizes_cyrillic_a_task_number(self) -> None:
        task = ExtractedTask(task_num="А6", condition="Найдите число.")

        self.assertEqual(_clean_extracted_task(task).task_num, "A6")

    def test_removes_transliterated_answer_field_and_ocr_artifacts(self) -> None:
        condition = (
            "<p>a) Найдите вероятность того, что в течение года в течение года "
            "лампа перегорит， а другая не перегорит..\n\n"
            "<p>b) Запишите результат.</p>\n"
            "Otvet: ___"
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="5"),
            (
                "<p>а) Найдите вероятность того, что в течение года лампа "
                "перегорит, а другая не перегорит.\n\n"
                "<p>б) Запишите результат.</p>"
            ),
        )

    def test_removes_variant_footer_from_source_block(self) -> None:
        blocks = _task_condition_blocks(
            "8. Найдите параметр a.\n"
            "Тренировочный вариант № 540\n"
            "9. Найдите процент.\n"
        )
        self.assertEqual(blocks["8"], "Найдите параметр a.")
        self.assertEqual(blocks["9"], "Найдите процент.")

    def test_numeric_condition_start_does_not_merge_next_task(self) -> None:
        blocks = _task_condition_blocks(
            "15 Решите неравенство $3^x \\leq 3$.\n\n"
            "16 15 января планируется взять кредит на 6 месяцев.\n"
            "Найдите наибольшее значение r.\n\n"
            "17 В треугольнике ABC угол C тупой.\n"
        )

        self.assertEqual(blocks["15"], "Решите неравенство $3^x \\leq 3$.")
        self.assertEqual(
            blocks["16"],
            (
                "15 января планируется взять кредит на 6 месяцев.\n"
                "Найдите наибольшее значение r."
            ),
        )
        self.assertEqual(
            blocks["17"],
            "В треугольнике $ABC$ угол $C$ тупой.",
        )


class _MissingTaskClient:
    provider_name = "Test"

    def __init__(self, retry_tasks: list[ExtractedTask]) -> None:
        self.retry_tasks = retry_tasks
        self.calls: list[tuple[str, list[str]]] = []

    def extract_markdown(
        self,
        markdown: str,
        image_ids: list[str],
    ) -> list[ExtractedTask]:
        self.calls.append((markdown, image_ids))
        return self.retry_tasks


class EmptyModelConditionTests(unittest.TestCase):
    def test_uses_exact_source_block_without_model_call(self) -> None:
        result = _restore_empty_model_condition(
            ExtractedTask(
                task_num="15.6",
                condition=MODEL_EMPTY_CONDITION_MARKER,
            ),
            "Решите неравенство\n\n$$ 5^x - 1 \\leq 0. $$",
            provider_name="Test",
            page_num=15,
        )

        self.assertEqual(result.task_num, "15.6")
        self.assertEqual(
            result.condition,
            "Решите неравенство\n\n$ 5^x - 1 \\leq 0. $",
        )

    def test_fails_quality_check_when_source_block_is_missing(self) -> None:
        with self.assertRaisesRegex(
            OCRQualityError,
            "15.6.*странице 15.*OCR-блок не найден",
        ):
            _restore_empty_model_condition(
                ExtractedTask(
                    task_num="15.6",
                    condition=MODEL_EMPTY_CONDITION_MARKER,
                ),
                None,
                provider_name="Test",
                page_num=15,
            )

    def test_restores_empty_task_num_from_unique_exact_source_block(self) -> None:
        result = _restore_empty_model_task_number(
            ExtractedTask(
                task_num=MODEL_EMPTY_TASK_NUM_MARKER,
                condition="Найдите площадь трапеции.",
            ),
            {
                "B8": "Найдите площадь пирамиды.",
                "B9": "Найдите площадь трапеции.",
            },
            provider_name="Test",
            page_num=6,
        )

        self.assertEqual(result.task_num, "B9")
        self.assertEqual(result.condition, "Найдите площадь трапеции.")

    def test_fails_quality_check_for_unresolved_empty_task_num(self) -> None:
        with self.assertRaisesRegex(
            OCRQualityError,
            "без task_num.*странице 1.*совпадений: 0",
        ):
            _restore_empty_model_task_number(
                ExtractedTask(
                    task_num=MODEL_EMPTY_TASK_NUM_MARKER,
                    condition="Неразборчивое рукописное условие.",
                ),
                {"5": "Решите уравнение."},
                provider_name="Test",
                page_num=1,
            )

    def test_does_not_guess_task_num_from_reordered_condition(self) -> None:
        with self.assertRaisesRegex(OCRQualityError, "совпадений: 0"):
            _restore_empty_model_task_number(
                ExtractedTask(
                    task_num=MODEL_EMPTY_TASK_NUM_MARKER,
                    condition="Найдите площадь трапеции равнобедренной.",
                ),
                {"B9": "Найдите площадь равнобедренной трапеции."},
                provider_name="Test",
                page_num=6,
            )


class MissingTaskRecoveryTests(unittest.TestCase):
    def test_sorts_lettered_task_numbers_by_section_and_number(self) -> None:
        page = Path("page_1.md")
        extracted = [
            (ExtractedTask(task_num="C1", condition="C1."), page),
            (ExtractedTask(task_num="B11", condition="B11."), page),
            (ExtractedTask(task_num="A10", condition="A10."), page),
            (ExtractedTask(task_num="B5", condition="B5."), page),
            (ExtractedTask(task_num="A2", condition="A2."), page),
        ]

        result = _deduplicate_tasks(extracted)

        self.assertEqual(
            [task.task_num for task, _page_path in result],
            ["A2", "A10", "B5", "B11", "C1"],
        )

    def test_restores_declared_lettered_tail_from_unlabeled_page_blocks(
        self,
    ) -> None:
        client = _MissingTaskClient([])
        page_2 = Path("page_2.md")
        page_3 = Path("page_3.md")
        page_2_markdown = (
            "B4. Решите уравнение.\n\n"
            "15. Функция задана графиком. Найдите число точек максимума."
        )
        page_3_markdown = (
            "Найдите значение выражения\n\n$$ x+1 $$\n\n"
            "Функция периодическая. Найдите её значение.\n\n"
            "Найдите все значения x.\n\n"
            "(Если их несколько, запишите наибольшее.)\n\n"
            "Магазин продал товар. Сколько процентов составила прибыль?\n\n"
            "Угол конуса равен 60°. Найдите другой угол.\n\n"
            "В параллелограмме проведена биссектриса. Найдите периметр.\n\n"
            "Для записи ответов на задания C1 и C2 используйте бланк №2.\n\n"
            "C1. Решите уравнение."
        )
        extracted = [
            *[
                (
                    ExtractedTask(
                        task_num=f"B{number}",
                        condition=f"Условие B{number}.",
                    ),
                    page_2,
                )
                for number in range(1, 5)
            ],
            (
                ExtractedTask(
                    task_num="15",
                    condition=(
                        "Функция задана графиком. "
                        "Найдите число точек максимума."
                    ),
                ),
                page_2,
            ),
            (
                ExtractedTask(task_num="C1", condition="Решите уравнение."),
                page_3,
            ),
        ]
        source_blocks = {
            "15": [
                _SourceTaskBlock(
                    condition=(
                        "Функция задана графиком. "
                        "Найдите число точек максимума."
                    ),
                    page_path=page_2,
                    image_id="b5.png",
                    available_image_ids=("b5.png",),
                )
            ]
        }

        result = _reconcile_lettered_source_tasks(
            client,
            extracted,
            source_blocks,
            document_markdown=(
                "Ответом на задания B1-B11 должно быть некоторое число.\n"
                "Для записи решений заданий C1-C5 используйте бланк №2."
            ),
            source_pages=[
                (page_2, page_2_markdown),
                (page_3, page_3_markdown),
            ],
        )

        by_number = {task.task_num: task for task, _page_path in result}
        self.assertTrue(
            {f"B{number}" for number in range(1, 12)}.issubset(by_number)
        )
        self.assertNotIn("15", by_number)
        self.assertEqual(by_number["B5"].image_id, "b5.png")
        self.assertIn("$ x+1 $", by_number["B6"].condition)
        self.assertIn("Если их несколько", by_number["B8"].condition)
        self.assertNotIn("Для записи ответов", by_number["B11"].condition)
        self.assertEqual(client.calls, [])

    def test_restores_locally_numbered_tail_of_declared_lettered_range(
        self,
    ) -> None:
        client = _MissingTaskClient([])
        page_2 = Path("page_2.md")
        page_3 = Path("page_3.md")
        extracted = [
            *[
                (
                    ExtractedTask(
                        task_num=f"B{number}",
                        condition=f"Условие B{number}.",
                    ),
                    page_2,
                )
                for number in range(1, 6)
            ],
            *[
                (
                    ExtractedTask(
                        task_num=str(number),
                        condition=f"Безымянное условие {number}.",
                    ),
                    page_3,
                )
                for number in range(1, 7)
            ],
            (
                ExtractedTask(task_num="C1", condition="Условие C1."),
                page_3,
            ),
        ]
        source_blocks = {
            **{
                f"B{number}": [
                    _SourceTaskBlock(
                        condition=f"Условие B{number}.",
                        page_path=page_2,
                        image_id=None,
                        available_image_ids=(),
                    )
                ]
                for number in range(1, 5)
            },
            "15": [
                _SourceTaskBlock(
                    condition="OCR ошибочно прочитал B5 как 15.",
                    page_path=page_2,
                    image_id=None,
                    available_image_ids=(),
                )
            ],
            "C1": [
                _SourceTaskBlock(
                    condition="Условие C1.",
                    page_path=page_3,
                    image_id=None,
                    available_image_ids=(),
                )
            ],
        }

        result = _reconcile_lettered_source_tasks(
            client,
            extracted,
            source_blocks,
            document_markdown=(
                "Ответом на задания B1-B11 должно быть некоторое число.\n"
                "Для записи решений заданий C1-C5 используйте бланк №2."
            ),
        )

        by_number = {task.task_num: task for task, _page_path in result}
        self.assertTrue(
            {f"B{number}" for number in range(1, 12)}.issubset(by_number)
        )
        self.assertNotIn("1", by_number)
        self.assertEqual(by_number["B6"].condition, "Безымянное условие 1.")
        self.assertEqual(by_number["B11"].condition, "Безымянное условие 6.")
        self.assertEqual(client.calls, [])

    def test_does_not_restore_single_numeric_task_from_declared_range(self) -> None:
        client = _MissingTaskClient([])
        page_path = Path("page_3.md")
        extracted = [
            (ExtractedTask(task_num="B5", condition="Условие B5."), page_path),
            (ExtractedTask(task_num="1", condition="Числовая задача."), page_path),
            (ExtractedTask(task_num="C1", condition="Условие C1."), page_path),
        ]

        result = _reconcile_lettered_source_tasks(
            client,
            extracted,
            {},
            document_markdown=(
                "Ответом на задания B1-B6 должно быть некоторое число."
            ),
        )

        self.assertEqual(result, extracted)
        self.assertEqual(client.calls, [])

    def test_does_not_restore_when_unlabeled_block_count_is_ambiguous(self) -> None:
        client = _MissingTaskClient([])
        page_2 = Path("page_2.md")
        page_3 = Path("page_3.md")
        extracted = [
            *[
                (
                    ExtractedTask(task_num=f"B{i}", condition=f"B{i}."),
                    page_2,
                )
                for i in range(1, 5)
            ],
            (
                ExtractedTask(task_num="15", condition="Найдите значение."),
                page_2,
            ),
            (
                ExtractedTask(task_num="C1", condition="Решите уравнение."),
                page_3,
            ),
        ]
        source_blocks = {
            "15": [
                _SourceTaskBlock(
                    condition="Найдите значение.",
                    page_path=page_2,
                    image_id=None,
                    available_image_ids=(),
                )
            ]
        }

        result = _reconcile_lettered_source_tasks(
            client,
            extracted,
            source_blocks,
            document_markdown="Ответом на задания B1-B11 является число.",
            source_pages=[
                (page_2, "B4. Условие.\n\n15. Найдите значение."),
                (page_3, "Найдите только одно значение.\n\nC1. Решите."),
            ],
        )

        self.assertEqual(result, extracted)
        self.assertEqual(client.calls, [])

    def test_isolated_late_task_does_not_imply_one_to_n_numbering(self) -> None:
        client = _MissingTaskClient(
            [ExtractedTask(task_num="1", condition="Ложная первая задача")]
        )
        page_path = Path("page_2.md")
        extracted = [
            (
                ExtractedTask(task_num="17", condition="Условие задачи 17."),
                page_path,
            )
        ]
        source_blocks = {
            "1": [
                _SourceTaskBlock(
                    condition="Служебный фрагмент с номером 1.",
                    page_path=page_path,
                    image_id=None,
                    available_image_ids=(),
                )
            ]
        }

        result = _recover_missing_expected_tasks(
            client,
            extracted,
            source_blocks,
            expected_tasks=1,
        )

        self.assertEqual(result, extracted)
        self.assertEqual(client.calls, [])

    def test_lettered_scheme_drops_numeric_solution_fragments_and_recovers_source(self) -> None:
        client = _MissingTaskClient([])
        page_path = Path("page_8.md")
        source_blocks = {
            task_num: [
                _SourceTaskBlock(
                    condition=f"Точное условие {task_num}.",
                    page_path=page_path,
                    image_id=f"{task_num}.jpg" if task_num == "C3" else None,
                    available_image_ids=(),
                )
            ]
            for task_num in ("C1", "C2", "C3", "C4")
        }
        extracted = [
            (ExtractedTask(task_num="C1", condition="Точное условие C1."), page_path),
            (ExtractedTask(task_num="1", condition="Фрагмент решения."), page_path),
            (ExtractedTask(task_num="2", condition="Другой фрагмент."), page_path),
        ]

        result = _reconcile_lettered_source_tasks(
            client,
            extracted,
            source_blocks,
        )

        by_number = {task.task_num: task for task, _ in result}
        self.assertEqual(set(by_number), {"C1", "C2", "C3", "C4"})
        self.assertEqual(by_number["C3"].condition, "Точное условие C3.")
        self.assertEqual(by_number["C3"].image_id, "C3.jpg")
        self.assertEqual(client.calls, [])

    def test_lettered_scheme_is_not_assumed_from_isolated_heading(self) -> None:
        client = _MissingTaskClient([])
        page_path = Path("page_1.md")
        extracted = [
            (ExtractedTask(task_num="1", condition="Первая задача."), page_path)
        ]
        source_blocks = {
            "C1": [
                _SourceTaskBlock(
                    condition="Единственная буквенная задача.",
                    page_path=page_path,
                    image_id=None,
                    available_image_ids=(),
                )
            ]
        }

        result = _reconcile_lettered_source_tasks(
            client,
            extracted,
            source_blocks,
        )

        self.assertEqual(result, extracted)

    def test_recovers_unreadable_ocr_source_without_paid_retry(self) -> None:
        client = _MissingTaskClient([])
        page_path = Path("page_1.md")
        source = f"Условие {OCR_UNREADABLE_REPEAT_MARKER}"

        result = _recover_missing_expected_tasks(
            client,
            [
                (ExtractedTask(task_num="1", condition="Первая"), page_path),
                (ExtractedTask(task_num="3", condition="Третья"), page_path),
            ],
            {
                "2": [
                    _SourceTaskBlock(
                        condition=source,
                        page_path=page_path,
                        image_id=None,
                        available_image_ids=(),
                    )
                ]
            },
            expected_tasks=3,
        )

        recovered = next(task for task, _ in result if task.task_num == "2")
        self.assertEqual(recovered.condition, source)
        self.assertEqual(client.calls, [])

    def test_recovers_model_omission_from_isolated_source_block(self) -> None:
        client = _MissingTaskClient(
            [ExtractedTask(task_num="2", condition="Найдите число $5$.")]
        )
        page_path = Path("page_1.md")
        extracted = [
            (ExtractedTask(task_num="1", condition="Первая"), page_path),
            (ExtractedTask(task_num="3", condition="Третья"), page_path),
        ]
        source_blocks = {
            "2": [
                _SourceTaskBlock(
                    condition="Найдите число 5.",
                    page_path=page_path,
                    image_id="two.jpg",
                    available_image_ids=("two.jpg",),
                )
            ]
        }

        result = _recover_missing_expected_tasks(
            client,
            extracted,
            source_blocks,
            expected_tasks=3,
        )

        by_number = {task.task_num: task for task, _ in result}
        self.assertEqual(set(by_number), {"1", "2", "3"})
        self.assertEqual(by_number["2"].condition, "Найдите число $5$.")
        self.assertEqual(by_number["2"].image_id, "two.jpg")
        self.assertEqual(client.calls, [("2. Найдите число 5.", [])])

    def test_uses_source_block_when_isolated_retry_also_omits_task(self) -> None:
        client = _MissingTaskClient([])
        page_path = Path("page_1.md")
        source_blocks = {
            "2": [
                _SourceTaskBlock(
                    condition="Точное условие 2.",
                    page_path=page_path,
                    image_id=None,
                    available_image_ids=(),
                )
            ]
        }

        result = _recover_missing_expected_tasks(
            client,
            [
                (ExtractedTask(task_num="1", condition="Первая"), page_path),
                (ExtractedTask(task_num="3", condition="Третья"), page_path),
            ],
            source_blocks,
            expected_tasks=3,
        )

        recovered = next(task for task, _ in result if task.task_num == "2")
        self.assertEqual(recovered.condition, "Точное условие 2.")

    def test_does_not_guess_for_nonstandard_task_numbering(self) -> None:
        client = _MissingTaskClient(
            [ExtractedTask(task_num="2", condition="Вторая")]
        )
        page_path = Path("page_1.md")
        extracted = [
            (ExtractedTask(task_num="1.1", condition="Подзадача"), page_path)
        ]

        result = _recover_missing_expected_tasks(
            client,
            extracted,
            {},
            expected_tasks=1,
        )

        self.assertEqual(result, extracted)
        self.assertEqual(client.calls, [])

    def test_ignores_duplicate_number_from_later_page_header(self) -> None:
        client = _MissingTaskClient([])
        extracted = [
            (
                ExtractedTask(task_num=str(number), condition=f"Задача {number}"),
                Path(f"page_{7 if number <= 4 else 8}.md"),
            )
            for number in (1, 2, 3, 4, 6)
        ]
        source_blocks = {
            "5": [
                _SourceTaskBlock(
                    condition="Изготовление стеклянных колб завершается отжигом.",
                    page_path=Path("page_8.md"),
                    image_id=None,
                    available_image_ids=(),
                ),
                _SourceTaskBlock(
                    condition="Служебный колонтитул страницы.",
                    page_path=Path("page_9.md"),
                    image_id=None,
                    available_image_ids=(),
                ),
            ]
        }

        result = _recover_missing_expected_tasks(
            client,
            extracted,
            source_blocks,
            expected_tasks=6,
        )

        recovered = next(task for task, _ in result if task.task_num == "5")
        self.assertEqual(
            recovered.condition,
            "Изготовление стеклянных колб завершается отжигом.",
        )
        self.assertEqual(
            client.calls[0][0],
            "5. Изготовление стеклянных колб завершается отжигом.",
        )

    def test_rejects_only_candidate_when_it_is_after_next_task(self) -> None:
        client = _MissingTaskClient([])
        extracted = [
            (ExtractedTask(task_num="4", condition="Четвёртая"), Path("page_7.md")),
            (ExtractedTask(task_num="6", condition="Шестая"), Path("page_8.md")),
        ]
        source_blocks = {
            "5": [
                _SourceTaskBlock(
                    condition="Служебный колонтитул страницы.",
                    page_path=Path("page_9.md"),
                    image_id=None,
                    available_image_ids=(),
                )
            ]
        }

        result = _recover_missing_expected_tasks(
            client,
            extracted,
            source_blocks,
            expected_tasks=6,
        )

        self.assertEqual(result, extracted)
        self.assertEqual(client.calls, [])


class EmbeddedTaskConditionTests(unittest.TestCase):
    def test_removes_exact_condition_of_another_task_from_suffix(self) -> None:
        page_path = Path("page_9.md")
        task_19 = (
            "Юра и Полина играют в числа. "
            "а) Может ли получиться число 2? "
            "б) Какое число наибольшее?"
        )
        extracted = [
            (
                ExtractedTask(
                    task_num="18",
                    condition="Найдите значения параметра.\n\n" + task_19,
                ),
                page_path,
            ),
            (ExtractedTask(task_num="19", condition=task_19), page_path),
        ]

        cleaned = _remove_embedded_task_conditions(extracted)
        by_number = {task.task_num: task for task, _ in cleaned}

        self.assertEqual(by_number["18"].condition, "Найдите значения параметра.")
        self.assertEqual(by_number["19"].condition, task_19)

    def test_does_not_remove_short_common_phrase(self) -> None:
        page_path = Path("page_1.md")
        extracted = [
            (
                ExtractedTask(
                    task_num="1",
                    condition="Найдите значение функции.",
                ),
                page_path,
            ),
            (
                ExtractedTask(task_num="2", condition="Найдите значение."),
                page_path,
            ),
        ]

        self.assertEqual(_remove_embedded_task_conditions(extracted), extracted)

    def test_removes_same_condition_with_different_html_and_latex_formatting(
        self,
    ) -> None:
        page_path = Path("page_6.md")
        task_14 = (
            "Точки $A$, $B$ и $C$ лежат на окружности основания конуса.\n"
            "<p>а) Докажите утверждение.</p>\n"
            "<p>б) Найдите высоту, если угол равен $60^\\circ$.</p>"
        )
        task_13 = (
            "<p>а) Решите уравнение.</p>\n"
            "<p>б) Найдите корни.  Точки $A$, B и C лежат на окружности "
            "основания конуса.</p>\n"
            "<p>а) Докажите утверждение.</p>\n"
            "<p>б) Найдите высоту, если угол равен 60°.</p>"
        )
        extracted = [
            (ExtractedTask(task_num="13", condition=task_13), page_path),
            (ExtractedTask(task_num="14", condition=task_14), page_path),
        ]

        cleaned = _remove_embedded_task_conditions(extracted)
        by_number = {task.task_num: task for task, _ in cleaned}

        self.assertEqual(
            by_number["13"].condition,
            "<p>а) Решите уравнение.</p>\n<p>б) Найдите корни.</p>",
        )
        self.assertEqual(by_number["14"].condition, task_14)

    def test_removes_condition_with_corrupted_heading_from_new_paragraph(
        self,
    ) -> None:
        page_path = Path("page_2/page_2.md")
        task_b5 = (
            "Функция $y=f(x)$ определена на промежутке $(a;b)$. "
            "На рисунке изображен график ее производной. Найдите число "
            "точек максимума функции $y=f(x)$ на промежутке $(a;b)$."
        )
        task_b4 = (
            "Решите уравнение $12^x-9\\cdot4^x=8\\cdot3^x-72$.\n\n"
            "(Если уравнение имеет более одного корня, запишите сумму "
            "корней).\n\n"
            "15 ∂$y=f(x)$ определена на промежутке $(a;b)$. На рисунке "
            "изображен график ее производной. Найдите число точек максимума "
            "функции $y=f(x)$ на промежутке $(a;b)$."
        )
        extracted = [
            (ExtractedTask(task_num="B4", condition=task_b4), page_path),
            (ExtractedTask(task_num="B5", condition=task_b5), page_path),
        ]

        cleaned = _remove_embedded_task_conditions(extracted)
        by_number = {task.task_num: task for task, _ in cleaned}

        self.assertEqual(
            by_number["B4"].condition,
            "Решите уравнение $12^x-9\\cdot4^x=8\\cdot3^x-72$.\n\n"
            "(Если уравнение имеет более одного корня, запишите сумму "
            "корней).",
        )
        self.assertEqual(by_number["B5"].condition, task_b5)


class ConditionArtifactRepairTests(unittest.TestCase):
    def test_removes_image_tags_and_their_layout_numbers(self) -> None:
        condition = (
            "Укажите рисунок. 1) "
            '<div><img src="imgs/one.jpg" width="24%" /></div> '
            "2) "
            '<div><img src="imgs/two.jpg" width="24%" /></div>'
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="A5"),
            "Укажите рисунок. 1)   2)",
        )

    def test_repairs_plain_geometry_line_label(self) -> None:
        condition = "Плоскость параллельна прямой АС."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="B8"),
            "Плоскость параллельна прямой $AC$.",
        )

    def test_repairs_cyrillic_single_geometry_letter_in_math_context(self) -> None:
        condition = (
            "В треугольнике $ABC$ угол С тупой. "
            "Точка Р лежит вне треугольника."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="17"),
            (
                "В треугольнике $ABC$ угол $C$ тупой. "
                "Точка $P$ лежит вне треугольника."
            ),
        )

    def test_keeps_same_cyrillic_letter_in_ordinary_prose(self) -> None:
        condition = "С вершиной пирамиды соединена середина ребра."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="14"),
            condition,
        )

    def test_repairs_geometry_labels_lost_outside_latex(self) -> None:
        condition = (
            "Докажите, что острые углы АВC и АСН равны. "
            "Найдите длину стороны АС, если АК = 10, ВК = 30."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="17"),
            (
                "Докажите, что острые углы $ABC$ и $ACH$ равны. "
                "Найдите длину стороны $AC$, если $AK=10$, $BK=30$."
            ),
        )

    def test_repairs_coordinated_sides_and_intersecting_segment(self) -> None:
        condition = (
            "$BP$ и $CP$ — перпендикуляры к сторонам $AB$ и АС "
            "соответственно, причём СР пересекает сторону $AB$."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="17"),
            (
                "$BP$ и $CP$ — перпендикуляры к сторонам $AB$ и $AC$ "
                "соответственно, причём $CP$ пересекает сторону $AB$."
            ),
        )

    def test_keeps_labels_outside_matching_geometry_context(self) -> None:
        condition = (
            "Компании ООО и РОМ соответственно завершили проверку, "
            "причём РОМ пересекает границу рынка."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="17"),
            condition,
        )

    def test_repairs_point_list_while_preserving_existing_latex(self) -> None:
        condition = (
            "Точки $K$, М и P — середины сторон $AB$, $BC$ и $AC$ "
            "соответственно."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="3"),
            (
                "Точки $K$, $M$ и $P$ — середины сторон $AB$, $BC$ и $AC$ "
                "соответственно."
            ),
        )

    def test_repairs_geometry_relations_and_indexed_point_list(self) -> None:
        condition = (
            "Основание прямой призмы $ABCA_1B_1C_1$ — треугольник "
            "$ABC$, в котором АВ = $BC$ = 8, $AC$ = 2. "
            "Вершины $A$₁, А, С₁ и точка Д ребра АВ лежат на сфере. "
            "Найдите радиус, если $AD$: $DB$ = 1:3."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="C3"),
            (
                "Основание прямой призмы $ABCA_1B_1C_1$ — треугольник "
                "$ABC$, в котором $AB=BC=8$, $AC=2$. "
                "Вершины $A_1$, $A$, $C_1$ и точка $D$ ребра $AB$ "
                "лежат на сфере. Найдите радиус, если $AD:DB=1:3$."
            ),
        )

    def test_geometry_assignment_normalization_avoids_false_fidelity_issue(
        self,
    ) -> None:
        source = _normalize_condition_artifacts(
            "В треугольнике $ABC$ сторона $AC$ = 2.",
            task_num="C3",
        )
        candidate = _normalize_condition_artifacts(
            "В треугольнике $ABC$ сторона $AC=2$.",
            task_num="C3",
        )

        self.assertEqual(source, candidate)
        self.assertEqual(_condition_fidelity_issues(source, candidate), [])

    def test_repairs_split_geometry_ratio(self) -> None:
        condition = (
            "Найдите площадь сечения, если $AT$$:TB$ =1:2, "
            "высота пирамиды равна 3."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="B8"),
            (
                "Найдите площадь сечения, если $AT:TB=1:2$, "
                "высота пирамиды равна 3."
            ),
        )

    def test_repairs_scaled_geometry_segment_relation(self) -> None:
        condition = (
            "Точки N и L на сторонах BC и SA расположены так, "
            "что LA = 4SL."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="14.2"),
            (
                "Точки $N$ и $L$ на сторонах $BC$ и $SA$ расположены так, "
                "что $LA=4SL$."
            ),
        )

    def test_repairs_already_split_scaled_geometry_relation(self) -> None:
        condition = "Известно, что $LA=4$$SL$."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="14.2"),
            "Известно, что $LA=4SL$.",
        )

    def test_repairs_two_point_list(self) -> None:
        condition = "Точки А и B лежат на окружности."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="14"),
            "Точки $A$ и $B$ лежат на окружности.",
        )

    def test_keeps_uppercase_abbreviations_outside_geometry_context(self) -> None:
        condition = "ООО РОМ зарегистрировано в Москве. С вершиной всё верно."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="5"),
            condition,
        )

    def test_final_task_cleanup_reapplies_all_condition_repairs(self) -> None:
        task = ExtractedTask(
            task_num="18",
            condition=(
                "Найдите все значения р, при каждом из которых уравнение "
                "$x^2-p=0$ имеет корень."
            ),
        )

        self.assertEqual(
            _clean_extracted_task(task).condition,
            (
                "Найдите все значения $p$, при каждом из которых уравнение "
                "$x^2-p=0$ имеет корень."
            ),
        )

    def test_restores_terminal_period_before_answer_field(self) -> None:
        blocks = _task_condition_blocks(
            "6. Найдите корень уравнения $x^2=4$\n"
            "Ответ: ____________________.\n"
            "7. Найдите значение выражения.\n"
            "Ответ: ____________________."
        )

        self.assertEqual(
            blocks["6"],
            "Найдите корень уравнения $x^2=4$.",
        )
        self.assertEqual(blocks["7"], "Найдите значение выражения.")

    def test_preserves_question_mark_before_answer_field(self) -> None:
        blocks = _task_condition_blocks(
            "4. Какова вероятность события?\nОтвет: ____________."
        )

        self.assertEqual(blocks["4"], "Какова вероятность события?")

    def test_restores_terminal_period_inside_html_paragraph(self) -> None:
        blocks = _task_condition_blocks(
            "6. <p>Найдите значение выражения $x+1$</p>\n"
            "Ответ: ____________."
        )

        self.assertEqual(
            blocks["6"],
            "<p>Найдите значение выражения $x+1$.</p>",
        )

    def test_does_not_add_terminal_period_without_answer_field(self) -> None:
        blocks = _task_condition_blocks(
            "17. <p>а) Докажите, что углы равны</p>\n"
            "<p>б) Найдите длину стороны.</p>"
        )

        self.assertEqual(
            blocks["17"],
            (
                "<p>а) Докажите, что углы равны</p>\n"
                "<p>б) Найдите длину стороны.</p>"
            ),
        )

    def test_moves_explanatory_prose_out_of_latex_span(self) -> None:
        condition = (
            "$y=f'(x) - \\text{производной функции } f(x)$, "
            "определённой на интервале."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="8"),
            (
                "$y=f'(x)$ — производной функции $f(x)$, "
                "определённой на интервале."
            ),
        )

    def test_repairs_transliterated_third_subpart_and_splits_html(self) -> None:
        condition = (
            "<p>a) Может ли результат удвоиться?</p>\n"
            "<p>b) Может ли результат увеличиться в пять раз?  "
            "v) В какое наибольшее число раз он может увеличиться?</p>"
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="19"),
            (
                "<p>а) Может ли результат удвоиться?</p>\n"
                "<p>б) Может ли результат увеличиться в пять раз?</p>\n"
                "<p>в) В какое наибольшее число раз он может увеличиться?</p>"
            ),
        )

    def test_removes_mixed_alphabet_answer_field(self) -> None:
        condition = "Найдите вероятность.\n\nOтвет: ___."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="4"),
            "Найдите вероятность.",
        )

    def test_removes_trailing_answer_instruction_inside_subpart(self) -> None:
        condition = (
            "<p>а) Может ли число встретиться?</p>\n"
            "<p>б) Найдите наибольшее число. Проверьте, чтобы каждый ответ "
            "был записан рядом с номером\n"
            "соответствующего задания.</p>"
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="19"),
            (
                "<p>а) Может ли число встретиться?</p>\n"
                "<p>б) Найдите наибольшее число.</p>"
            ),
        )

    def test_repairs_prism_name_from_repeated_vertex_structure(self) -> None:
        condition = (
            "В правильной треугольной призме $ABCD_{1}B_{1}C_{1}$ "
            "проведена плоскость."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="14"),
            (
                "В правильной треугольной призме $ABCA_1B_1C_1$ "
                "проведена плоскость."
            ),
        )

    def test_repairs_same_prism_defect_with_arbitrary_vertex_names(self) -> None:
        condition = "В призме $MNKP_2N_2K_2$ выбрана точка."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="14"),
            "В призме $MNKM_2N_2K_2$ выбрана точка.",
        )

    def test_does_not_change_non_triangular_prism_name(self) -> None:
        condition = "В призме $ABCDA_1B_1C_1D_1$ выбрана точка."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="14"),
            condition,
        )

    def test_preserves_already_correct_triangular_prism_formatting(self) -> None:
        condition = "В призме $ ABCA_1B_1C_1 $ выбрана точка."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="14"),
            condition,
        )

    def test_restores_period_before_new_task_instruction(self) -> None:
        condition = (
            "Период полураспада составляет 7 минут "
            "Найдите, через сколько минут останется половина массы."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="9"),
            (
                "Период полураспада составляет 7 минут. "
                "Найдите, через сколько минут останется половина массы."
            ),
        )

    def test_does_not_invent_punctuation_missing_in_source_subpart(self) -> None:
        condition = (
            "<p>а) Докажите, что углы равны</p>\n"
            "<p>б) Найдите длину стороны.</p>"
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="17"),
            condition,
        )

    def test_repairs_cyrillic_parameter_confirmed_by_formula(self) -> None:
        condition = (
            "Найдите все значения р, при каждом из которых уравнение "
            "$x^2-p=0$ имеет корень."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="18"),
            (
                "Найдите все значения $p$, при каждом из которых уравнение "
                "$x^2-p=0$ имеет корень."
            ),
        )

    def test_repairs_latin_parameter_confirmed_by_formula(self) -> None:
        condition = (
            "Долг увеличивается на $r\\%$. "
            "Найдите наибольшее значение r, при котором условие выполнено."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="16"),
            (
                "Долг увеличивается на $r\\%$. "
                "Найдите наибольшее значение $r$, при котором условие "
                "выполнено."
            ),
        )

    def test_repairs_plain_greek_plane_symbol(self) -> None:
        condition = (
            "Проведена плоскость α.\n"
            "<p>а) Докажите, что сечением плоскостью α будет трапеция.</p>\n"
            "<p>б) Найдите расстояние до плоскости α.</p>"
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="14"),
            (
                "Проведена плоскость $\\alpha$.\n"
                "<p>а) Докажите, что сечением плоскостью $\\alpha$ будет "
                "трапеция.</p>\n"
                "<p>б) Найдите расстояние до плоскости $\\alpha$.</p>"
            ),
        )

    def test_repairs_plain_radius_definition(self) -> None:
        condition = "Формула задана, где r — радиус сферы в метрах."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="9"),
            "Формула задана, где $r$ — радиус сферы в метрах.",
        )

    def test_keeps_greek_symbol_outside_plane_context(self) -> None:
        condition = "Коэффициент α указан в справочных материалах."

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="5"),
            condition,
        )

    def test_keeps_cyrillic_letter_without_matching_formula_variable(self) -> None:
        condition = (
            "Найдите все значения р, при каждом из которых уравнение "
            "$x^2-a=0$ имеет корень."
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="18"),
            condition,
        )

    def test_repairs_high_confidence_legacy_exam_ocr(self) -> None:
        cases = {
            "A1": ("Yпростите выражение $x^2$.", "Упростите выражение $x^2$."),
            "A2": ("BЫЧИСЛИТЕ: $2+2$.", "Вычислите: $2+2$."),
            "A9": (
                r"Peunrte ypaennne cos x = $\frac{1}{2}$.",
                r"Решите уравнение $\cos x=\frac{1}{2}$.",
            ),
            "B1": (
                (
                    "Haidntte zhaenie viyapxennia "
                    "cos 2α + 4·sin 2α, esni sin 2α = 0,3."
                ),
                (
                    "Найдите значение выражения "
                    r"$\cos^2\alpha+4\cdot\sin^2\alpha$, если "
                    r"$\sin^2\alpha=0,3$."
                ),
            ),
            "B4": (
                (
                    "Peunte urpahenne 12-9-4=8.3-72.\n\n"
                    "(Если уравнение имеет более одного корня, то в "
                    "бланке ответо защищает сумму корней).\n\n15 ∂"
                ),
                (
                    r"Решите уравнение $12^x-9\cdot4^x="
                    r"8\cdot3^x-72$."
                    "\n\n(Если уравнение имеет более одного корня, то в "
                    "бланке ответов запишите сумму корней)."
                ),
            ),
        }

        for task_num, (source, expected) in cases.items():
            with self.subTest(task_num=task_num):
                self.assertEqual(
                    _normalize_condition_artifacts(
                        source,
                        task_num=task_num,
                    ),
                    expected,
                )

    def test_repairs_legacy_visual_task_text_and_next_heading_debris(
        self,
    ) -> None:
        condition = (
            "Хозяйка установила на утоге режим «хлопоко». Утог остывает. "
            "На рисунке представлен график зависимости температуры 7 "
            "упога в промежутке времени 1 между размыканиями. Температура "
            "утога достигает максимума.\n\nAS"
        )

        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="A7"),
            (
                "Хозяйка установила на утюге режим «хлопок». Утюг остывает. "
                "На рисунке представлен график зависимости температуры "
                "$T$ утюга в промежутке времени $t$ между размыканиями. "
                "Температура утюга достигает максимума."
            ),
        )

    def test_repairs_power_scope_only_with_matching_answer_boundaries(
        self,
    ) -> None:
        source = (
            r"Решите неравенство $7^{4}x>7^{3}x+21$. "
            "1) (-∞, 21) 2) (3, +∞) 3) (-∞, 3) 4) (21, +∞)"
        )
        self.assertEqual(
            _normalize_condition_artifacts(source, task_num="A10"),
            (
                r"Решите неравенство $7^{4x}>7^{3x+21}$. "
                "1) (-∞; 21) 2) (3; +∞) 3) (-∞; 3) 4) (21; +∞)"
            ),
        )

        ambiguous = r"Сравните $7^{4}x>7^{3}x+21$."
        self.assertEqual(
            _normalize_condition_artifacts(ambiguous, task_num="1"),
            ambiguous,
        )

        decimal = "Вероятность равна (0,125)."
        self.assertEqual(
            _normalize_condition_artifacts(decimal, task_num="2"),
            decimal,
        )

    def test_unwraps_only_fraction_with_empty_denominator(self) -> None:
        source = (
            r"Найдите $\frac{\sqrt{35}-\frac{1}{a-b}}{}$ и "
            r"$\frac{1}{2}$."
        )
        self.assertEqual(
            _normalize_condition_artifacts(source, task_num="B6"),
            r"Найдите $\sqrt{35}-\frac{1}{a-b}$ и $\frac{1}{2}$.",
        )

    def test_repairs_missing_formula_choice_numbers(self) -> None:
        source = (
            "Укажите эту функцию.\n\n$y=2^x$\n\n$y=3^x$\n\n"
            "$y=4^x$\n\n$y=5^x$"
        )
        self.assertEqual(
            _normalize_condition_artifacts(source, task_num="A4"),
            (
                "Укажите эту функцию.\n\n1) $y=2^x$\n\n2) $y=3^x$\n\n"
                "3) $y=4^x$\n\n4) $y=5^x$"
            ),
        )

    def test_repairs_split_solid_notation_and_plain_line_equation(self) -> None:
        source = (
            r"Дан прямоугольный параллелепипед $ABCD_{1}B_{1}C_{1}D_{1}$, "
            r"$ $$AA_1$=6$\sqrt{5} $. Касательные параллельны прямой y=26x. "
            "Определите тангено угла."
        )
        self.assertEqual(
            _normalize_condition_artifacts(source, task_num="C4"),
            (
                r"Дан прямоугольный параллелепипед $ABCDA_1B_1C_1D_1$, "
                r"$AA_1=6\sqrt{5}$. Касательные параллельны прямой "
                r"$y=26x$. Определите тангенс угла."
            ),
        )

    def test_does_not_rewrite_complete_solid_name(self) -> None:
        condition = (
            r"Дан параллелепипед $ABCDA_1B_1C_1D_1$ и точка $M$."
        )
        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="14"),
            condition,
        )

    def test_repairs_split_solid_name_with_arbitrary_vertices(self) -> None:
        condition = r"Дан параллелепипед $WXYQ_2X_2Y_2Q_2$."
        self.assertEqual(
            _normalize_condition_artifacts(condition, task_num="14"),
            r"Дан параллелепипед $WXYQW_2X_2Y_2Q_2$.",
        )


class DuplicateTaskImageTests(unittest.TestCase):
    def test_keeps_shared_image_only_on_visual_task(self) -> None:
        page = Path("page_2/page_2.md")
        extracted = [
            (
                ExtractedTask(
                    task_num="B4",
                    condition="Решите уравнение $x=1$.",
                    image_id="graph.png",
                ),
                page,
            ),
            (
                ExtractedTask(
                    task_num="B5",
                    condition="На рисунке изображен график функции.",
                    image_id="graph.png",
                ),
                page,
            ),
        ]

        result = _reconcile_duplicate_task_images(extracted)

        self.assertIsNone(result[0][0].image_id)
        self.assertEqual(result[1][0].image_id, "graph.png")

    def test_keeps_ambiguous_shared_image_assignments(self) -> None:
        page = Path("page_2/page_2.md")
        extracted = [
            (
                ExtractedTask(
                    task_num="1",
                    condition="На рисунке изображена схема.",
                    image_id="diagram.png",
                ),
                page,
            ),
            (
                ExtractedTask(
                    task_num="2",
                    condition="На рисунке изображен график.",
                    image_id="diagram.png",
                ),
                page,
            ),
        ]

        result = _reconcile_duplicate_task_images(extracted)

        self.assertEqual(
            [task.image_id for task, _page in result],
            ["diagram.png", "diagram.png"],
        )

    def test_matches_same_image_content_saved_under_different_names(self) -> None:
        with TemporaryDirectory() as temp_dir:
            page = Path(temp_dir) / "page_2" / "page_2.md"
            image_dir = page.parent / "imgs"
            image_dir.mkdir(parents=True)
            (image_dir / "equation_crop.png").write_bytes(b"same image")
            (image_dir / "graph_crop.png").write_bytes(b"same image")
            extracted = [
                (
                    ExtractedTask(
                        task_num="B4",
                        condition="Решите уравнение $x=1$.",
                        image_id="equation_crop.png",
                    ),
                    page,
                ),
                (
                    ExtractedTask(
                        task_num="B5",
                        condition="На рисунке изображен график функции.",
                        image_id="graph_crop.png",
                    ),
                    page,
                ),
            ]

            result = _reconcile_duplicate_task_images(extracted)

        self.assertIsNone(result[0][0].image_id)
        self.assertEqual(result[1][0].image_id, "graph_crop.png")


class DeepSeekQualityTests(unittest.TestCase):
    def test_long_solution_retries_in_compact_mode(self) -> None:
        client, completions = _client_with_responses(
            [
                _response('{"solution":"очень длинное решение","answer":"1"}'),
                _response('{"solution":"кратко","answer":"1"}'),
            ],
            max_solution_chars=10,
        )

        result = client._request_structured(
            "prompt",
            TaskSolution,
            thinking=True,
        )

        self.assertEqual(result.solution, "кратко")
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(completions.calls[1]["max_tokens"], 17)
        self.assertEqual(
            completions.calls[1]["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_invalid_compact_response_gets_minimal_third_attempt(self) -> None:
        client, completions = _client_with_responses(
            [
                _response('{"solution":"оборвано'),
                _response('{"solution":"снова оборвано'),
                _response('{"solution":"готово","answer":"1"}'),
            ],
            max_solution_chars=100,
        )

        result = client._request_structured(
            "prompt",
            TaskSolution,
            thinking=False,
        )

        self.assertEqual(result.solution, "готово")
        self.assertEqual(len(completions.calls), 3)
        self.assertEqual(completions.calls[1]["max_tokens"], 17)
        self.assertEqual(completions.calls[2]["max_tokens"], 9)

    def test_solution_verification_returns_corrected_result(self) -> None:
        client = object.__new__(DeepSeekTaskClient)
        client.verify_solutions = True
        task = ExtractedTask(task_num="19", condition="Условие")
        candidate = TaskSolution(solution="ошибочное решение", answer="нет")
        verification = SolutionVerification(
            is_correct=False,
            issues=["найден контрпример"],
            solution="исправленное решение",
            answer="да",
        )

        with patch.object(
            client,
            "_request_task_result",
            return_value=candidate,
        ), patch.object(
            client,
            "_verify_task_solution",
            return_value=verification,
        ):
            result = client.solve_task(task)

        self.assertEqual(result.solution, "исправленное решение")
        self.assertEqual(result.answer, "да")


class _PartiallyFailingClient:
    provider_name = "Test"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def solve_task(self, task: ExtractedTask) -> TaskSolution:
        self.calls.append(task.task_num)
        if task.task_num == "1":
            raise ValueError("сбой первой задачи")
        return TaskSolution(solution="решено", answer="2")


class PerTaskFailureTests(unittest.TestCase):
    def test_later_tasks_continue_after_one_failure(self) -> None:
        records = [
            TaskRecord(task_num="1", condition="Условие 1"),
            TaskRecord(task_num="2", condition="Условие 2"),
        ]
        extracted = [
            (ExtractedTask(task_num="1", condition="Условие 1"), Path("1.md")),
            (ExtractedTask(task_num="2", condition="Условие 2"), Path("2.md")),
        ]

        client = _PartiallyFailingClient()
        with self.assertRaisesRegex(RuntimeError, "1: ValueError"):
            _generate_solutions_and_answers(
                records,
                extracted,
                client,
            )

        self.assertEqual(records[0].solution, "")
        self.assertEqual(records[1].solution, "решено")
        self.assertEqual(records[1].answer, "2")
        self.assertEqual(client.calls, ["1", "2"])

    def test_unreadable_ocr_condition_prevents_false_ok_status(self) -> None:
        records = [
            TaskRecord(
                task_num="C5",
                condition=OCR_UNREADABLE_REPEAT_MARKER,
            )
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "задачах C5.*нельзя считать качественно обработанным",
        ):
            _raise_unreadable_ocr_conditions(records)

    def test_unattached_ocr_noise_page_prevents_false_ok_status(self) -> None:
        records = [TaskRecord(task_num="1", condition="Читаемое условие")]

        with self.assertRaisesRegex(RuntimeError, "на страницах 5"):
            _raise_unreadable_ocr_conditions(records, affected_pages=[5])

    def test_unreadable_ocr_task_is_not_sent_for_paid_generation(self) -> None:
        records = [
            TaskRecord(
                task_num="1",
                condition=f"Условие {OCR_UNREADABLE_REPEAT_MARKER}",
            ),
            TaskRecord(task_num="2", condition="Условие 2"),
        ]
        extracted = [
            (
                ExtractedTask(
                    task_num="1",
                    condition=f"Условие {OCR_UNREADABLE_REPEAT_MARKER}",
                ),
                Path("1.md"),
            ),
            (ExtractedTask(task_num="2", condition="Условие 2"), Path("2.md")),
        ]
        client = _PartiallyFailingClient()

        with self.assertRaisesRegex(
            RuntimeError,
            "OCR_UNREADABLE_REPEAT.*Остальные результаты сохранены",
        ):
            _generate_solutions_and_answers(records, extracted, client)

        self.assertEqual(client.calls, ["2"])
        self.assertEqual(records[0].solution, "")
        self.assertEqual(records[1].solution, "решено")


if __name__ == "__main__":
    unittest.main()
