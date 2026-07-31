from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from exam_parser.condition_audit import (
    ConditionCorrection,
    MistralConditionAuditor,
    PageConditionCorrections,
    _apply_verified_corrections,
    _exactly_confirmed_corrections,
    _page_image_data_urls,
    verify_markdown_conditions,
)


class _FakeAuditor:
    def __init__(self, corrections: list[ConditionCorrection]) -> None:
        self.corrections = corrections
        self.calls: list[tuple[str, Path]] = []

    def audit(
        self,
        markdown: str,
        page_image: Path,
    ) -> list[ConditionCorrection]:
        self.calls.append((markdown, page_image))
        return self.corrections


class _FakeChat:
    def __init__(self, responses: list[PageConditionCorrections]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(parsed=self.responses.pop(0))
                )
            ]
        )


def _write_page(root: Path, markdown: str) -> tuple[Path, Path]:
    markdown_dir = root / "markdown" / "page_1"
    markdown_dir.mkdir(parents=True)
    markdown_path = markdown_dir / "page_1.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    pages_dir = root / "pages"
    pages_dir.mkdir()
    page_image = pages_dir / "page_1.png"
    page_image.write_bytes(b"page image")
    return markdown_path, page_image


class ConditionAuditTests(unittest.TestCase):
    def test_applies_regressions_found_in_scanned_exam_pages(self) -> None:
        markdown = (
            "8 Найдите промежутки возрастания функции $f'(x)$.\n"
            r"9 Сила равна $F_A = \frac{a}{r} r^3$." "\n"
            "14 В призме $ABC_1B_1C_1$ сечением будет трапения.\n"
            "16 Долг (в мн рублей)."
        )
        corrections = [
            ConditionCorrection(
                task_num="8",
                source_fragment="возрастания функции $f'(x)$",
                replacement="возрастания функции $f(x)$",
                reason="Штрих отсутствует.",
            ),
            ConditionCorrection(
                task_num="9",
                source_fragment=r"$F_A = \frac{a}{r} r^3$",
                replacement=r"$F_A = \alpha\rho g r^3$",
                reason="Формула прочитана неверно.",
            ),
            ConditionCorrection(
                task_num="14",
                source_fragment=r"$ABC_1B_1C_1$",
                replacement=r"$ABCA_1B_1C_1$",
                reason="Потеряна вершина A1.",
            ),
            ConditionCorrection(
                task_num="14",
                source_fragment="трапения",
                replacement="трапеция",
                reason="Опечатка OCR.",
            ),
            ConditionCorrection(
                task_num="16",
                source_fragment="в мн рублей",
                replacement="в млн рублей",
                reason="Потеряна буква л.",
            ),
        ]

        corrected, applied = _apply_verified_corrections(
            markdown,
            corrections,
            page_num=1,
        )

        self.assertEqual(applied, 5)
        self.assertIn("возрастания функции $f(x)$", corrected)
        self.assertIn(r"$F_A = \alpha\rho g r^3$", corrected)
        self.assertIn(r"$ABCA_1B_1C_1$", corrected)
        self.assertIn("трапеция", corrected)
        self.assertIn("в млн рублей", corrected)

    def test_provider_requires_independent_confirmation(self) -> None:
        draft_correction = ConditionCorrection(
            task_num="16",
            source_fragment="мн рублей",
            replacement="млн рублей",
            reason="На изображении есть буква л.",
        )
        confirmed_correction = draft_correction.model_copy(
            update={"reason": "Исправление независимо подтверждено."}
        )
        chat = _FakeChat(
            [
                PageConditionCorrections(corrections=[draft_correction]),
                PageConditionCorrections(corrections=[confirmed_correction]),
            ]
        )
        auditor = object.__new__(MistralConditionAuditor)
        auditor.client = SimpleNamespace(chat=chat)
        auditor.model = "draft-model"
        auditor.confirmation_model = "confirmation-model"

        with tempfile.TemporaryDirectory() as temp:
            page_image = Path(temp) / "page_1.png"
            Image.new("RGB", (400, 200), "white").save(page_image)
            result = auditor.audit("16 Долг в мн рублей.", page_image)

        self.assertEqual(result, [confirmed_correction])
        self.assertEqual(len(chat.calls), 2)
        self.assertEqual(chat.calls[0]["model"], "draft-model")
        self.assertEqual(chat.calls[1]["model"], "confirmation-model")
        confirmation_prompt = chat.calls[1]["messages"][0]["content"][0]["text"]
        self.assertIn("независим", confirmation_prompt.lower())
        self.assertIn("мн рублей", confirmation_prompt)
        self.assertNotIn("Черновые исправления", confirmation_prompt)
        self.assertEqual(
            [item["type"] for item in chat.calls[0]["messages"][0]["content"]],
            ["text", "image_url", "image_url"],
        )

    def test_confirmation_cannot_change_or_introduce_correction(self) -> None:
        draft = [
            ConditionCorrection(
                task_num="8",
                source_fragment="$f'(x)$",
                replacement="$f(x)$",
                reason="В черновике штрих признан лишним.",
            )
        ]
        confirmed = [
            ConditionCorrection(
                task_num="8",
                source_fragment="$f'(x)$",
                replacement="$f''(x)$",
                reason="Проверяющая модель предложила другой текст.",
            ),
            ConditionCorrection(
                task_num="9",
                source_fragment="264 600",
                replacement="264600",
                reason="Новая стилистическая правка.",
            ),
        ]

        self.assertEqual(
            _exactly_confirmed_corrections(draft, confirmed),
            [],
        )

    def test_splits_wide_page_into_two_readable_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            page_image = Path(temp) / "page_1.png"
            Image.new("RGB", (800, 400), "white").save(page_image)

            data_urls = _page_image_data_urls(page_image)

        self.assertEqual(len(data_urls), 2)
        self.assertTrue(
            all(item.startswith("data:image/png;base64,") for item in data_urls)
        )

    def test_applies_only_unique_exact_visual_correction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            markdown_path, page_image = _write_page(
                root,
                "8 Найдите промежутки возрастания функции $f'(x)$.",
            )
            auditor = _FakeAuditor(
                [
                    ConditionCorrection(
                        task_num="8",
                        source_fragment="возрастания функции $f'(x)$",
                        replacement="возрастания функции $f(x)$",
                        reason="На странице у функции нет штриха.",
                    )
                ]
            )

            result_dir = verify_markdown_conditions(
                root / "markdown",
                root / "pages",
                root / "verified",
                root / "cache",
                auditor=auditor,
            )
            corrected = (
                result_dir / markdown_path.relative_to(root / "markdown")
            ).read_text(encoding="utf-8")

        self.assertEqual(
            corrected,
            "8 Найдите промежутки возрастания функции $f(x)$.",
        )
        self.assertEqual(auditor.calls[0][1], page_image)

    def test_skips_correction_when_source_fragment_is_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            markdown = "$f'(x)$ и ещё раз $f'(x)$"
            _write_page(root, markdown)
            auditor = _FakeAuditor(
                [
                    ConditionCorrection(
                        task_num="8",
                        source_fragment="$f'(x)$",
                        replacement="$f(x)$",
                        reason="Черновой фрагмент слишком короткий.",
                    )
                ]
            )

            result_dir = verify_markdown_conditions(
                root / "markdown",
                root / "pages",
                root / "verified",
                root / "cache",
                auditor=auditor,
            )

        self.assertEqual(result_dir, root / "markdown")

    def test_applies_nonoverlapping_corrections_against_original_task_block(self) -> None:
        markdown = (
            "8 На рисунке дан график $f'(x)$ на интервале $(-8, 7)$.\n"
            "Найдите промежутки возрастания функции $f'(x)$.\n"
            "9 Сила равна $F_A = a r^3$."
        )
        corrections = [
            ConditionCorrection(
                task_num="8",
                source_fragment="возрастания функции $f'(x)$",
                replacement="возрастания функции $f(x)$",
                reason="У искомой функции нет штриха.",
            ),
            ConditionCorrection(
                task_num="8",
                source_fragment="$(-8, 7)$",
                replacement="$(-8; 7)$",
                reason="В интервале стоит точка с запятой.",
            ),
        ]

        corrected, applied = _apply_verified_corrections(
            markdown,
            corrections,
            page_num=1,
        )

        self.assertEqual(applied, 2)
        self.assertIn("возрастания функции $f(x)$", corrected)
        self.assertIn("$(-8; 7)$", corrected)

    def test_rejects_correction_outside_declared_task(self) -> None:
        markdown = "8 Найдите $f'(x)$.\n9 Найдите значение."
        correction = ConditionCorrection(
            task_num="9",
            source_fragment="$f'(x)$",
            replacement="$f(x)$",
            reason="Фрагмент относится к другой задаче.",
        )

        corrected, applied = _apply_verified_corrections(
            markdown,
            [correction],
            page_num=1,
        )

        self.assertEqual(applied, 0)
        self.assertEqual(corrected, markdown)

    def test_uses_real_task_when_same_number_appears_in_page_header(self) -> None:
        markdown = (
            "5\nСлужебный колонтитул страницы.\n"
            "5 Линия подсветки состоит из четырёх ламп."
        )
        correction = ConditionCorrection(
            task_num="5",
            source_fragment="четырёх",
            replacement="4",
            reason="В условии напечатана цифра.",
        )

        corrected, applied = _apply_verified_corrections(
            markdown,
            [correction],
            page_num=1,
        )

        self.assertEqual(applied, 1)
        self.assertIn("состоит из 4 ламп", corrected)

    def test_rejects_styling_and_answer_instruction_changes(self) -> None:
        markdown = (
            "9 Сила не превосходит 264 600 Н.\n"
            "10 Найдите скорость. Ответ выразите в км/ч."
        )
        corrections = [
            ConditionCorrection(
                task_num="9",
                source_fragment="264 600",
                replacement="264600",
                reason="Убран пробел.",
            ),
            ConditionCorrection(
                task_num="10",
                source_fragment=" Ответ выразите в км/ч.",
                replacement="",
                reason="Ошибочно принято за поле ответа.",
            ),
        ]

        corrected, applied = _apply_verified_corrections(
            markdown,
            corrections,
            page_num=1,
        )

        self.assertEqual(applied, 0)
        self.assertEqual(corrected, markdown)

    def test_reuses_audit_cache_for_unchanged_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_page(root, "14 Сечением будет трапения.")
            correction = ConditionCorrection(
                task_num="14",
                source_fragment="трапения",
                replacement="трапеция",
                reason="На странице напечатано «трапеция».",
            )
            first = _FakeAuditor([correction])
            second = _FakeAuditor([])

            verify_markdown_conditions(
                root / "markdown",
                root / "pages",
                root / "verified",
                root / "cache",
                auditor=first,
            )
            result_dir = verify_markdown_conditions(
                root / "markdown",
                root / "pages",
                root / "verified",
                root / "cache",
                auditor=second,
            )
            corrected = (result_dir / "page_1" / "page_1.md").read_text(
                encoding="utf-8"
            )

        self.assertEqual(len(first.calls), 1)
        self.assertEqual(second.calls, [])
        self.assertIn("трапеция", corrected)

    def test_changed_markdown_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            markdown_path, _ = _write_page(root, "14 Сечением будет трапения.")
            first = _FakeAuditor([])
            second = _FakeAuditor([])

            verify_markdown_conditions(
                root / "markdown",
                root / "pages",
                root / "verified",
                root / "cache",
                auditor=first,
            )
            markdown_path.write_text(
                "14 Сечением будет прямоугольная трапейка.",
                encoding="utf-8",
            )
            verify_markdown_conditions(
                root / "markdown",
                root / "pages",
                root / "verified",
                root / "cache",
                auditor=second,
            )

        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 1)

    def test_previous_audit_version_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            markdown_path, page_image = _write_page(
                root,
                "14 Сечением будет трапения.",
            )
            cache_path = root / "cache" / "page_1.json"
            cache_path.parent.mkdir()
            cache_path.write_text(
                json.dumps(
                    {
                        "version": "exam-parser-condition-audit-v1",
                        "source_sha256": hashlib.sha256(
                            markdown_path.read_bytes()
                        ).hexdigest(),
                        "image_sha256": hashlib.sha256(
                            page_image.read_bytes()
                        ).hexdigest(),
                        "corrections": [],
                    }
                ),
                encoding="utf-8",
            )
            auditor = _FakeAuditor([])

            verify_markdown_conditions(
                root / "markdown",
                root / "pages",
                root / "verified",
                root / "cache",
                auditor=auditor,
            )

        self.assertEqual(len(auditor.calls), 1)


if __name__ == "__main__":
    unittest.main()
