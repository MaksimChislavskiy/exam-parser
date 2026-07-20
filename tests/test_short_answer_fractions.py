from __future__ import annotations

import unittest

from exam_parser.math_text import normalize_ege_short_answer


class ShortAnswerLatexFractionTests(unittest.TestCase):
    def test_converts_dfrac_to_decimal(self) -> None:
        self.assertEqual(
            normalize_ege_short_answer("4", r"$\dfrac{1}{8}$"),
            "0,125",
        )

    def test_converts_tfrac_to_decimal(self) -> None:
        self.assertEqual(
            normalize_ege_short_answer("4", r"$\tfrac{3}{20}$"),
            "0,15",
        )

    def test_keeps_standard_frac_support(self) -> None:
        self.assertEqual(
            normalize_ege_short_answer("4", r"$\frac{1}{4}$"),
            "0,25",
        )

    def test_rejects_nonterminating_dfrac(self) -> None:
        with self.assertRaisesRegex(ValueError, "не является конечной"):
            normalize_ege_short_answer("4", r"$\dfrac{1}{3}$")


if __name__ == "__main__":
    unittest.main()
