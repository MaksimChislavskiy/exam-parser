from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from exam_parser.condition_audit import (
    ConditionCorrection,
    MistralConditionAuditor,
    PageConditionCorrections,
    _apply_verified_corrections,
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
            page_image.write_bytes(b"png")
            result = auditor.audit("16 Долг в мн рублей.", page_image)

        self.assertEqual(result, [confirmed_correction])
        self.assertEqual(len(chat.calls), 2)
        self.assertEqual(chat.calls[0]["model"], "draft-model")
        self.assertEqual(chat.calls[1]["model"], "confirmation-model")
        confirmation_prompt = chat.calls[1]["messages"][0]["content"][0]["text"]
        self.assertIn("независимо", confirmation_prompt.lower())
        self.assertIn("мн рублей", confirmation_prompt)

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


if __name__ == "__main__":
    unittest.main()
