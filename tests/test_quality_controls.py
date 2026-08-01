from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from exam_parser.deepseek_client import DeepSeekTaskClient
from exam_parser.markdown_pipeline import (
    _SourceTaskBlock,
    _clean_extracted_task,
    _condition_fidelity_issues,
    _ensure_condition_fidelity,
    _generate_solutions_and_answers,
    _normalize_condition_artifacts,
    _recover_missing_expected_tasks,
    _remove_embedded_task_conditions,
    _task_condition_blocks,
)
from exam_parser.math_text import normalize_ege_short_answer
from exam_parser.models import (
    ExtractedTask,
    SolutionVerification,
    TaskRecord,
    TaskSolution,
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


class ConditionFidelityTests(unittest.TestCase):
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


class MissingTaskRecoveryTests(unittest.TestCase):
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


class ConditionArtifactRepairTests(unittest.TestCase):
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
                "Найдите длину стороны $AC$, если $AK$ = 10, $BK$ = 30."
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

    def solve_task(self, task: ExtractedTask) -> TaskSolution:
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

        with self.assertRaisesRegex(RuntimeError, "1: ValueError"):
            _generate_solutions_and_answers(
                records,
                extracted,
                _PartiallyFailingClient(),
            )

        self.assertEqual(records[0].solution, "")
        self.assertEqual(records[1].solution, "решено")
        self.assertEqual(records[1].answer, "2")


if __name__ == "__main__":
    unittest.main()
