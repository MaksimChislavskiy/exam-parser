from __future__ import annotations

from exam_parser.data_store import DataStore
from exam_parser.verified_condition_corrections import (
    apply_verified_condition_correction,
    canonical_condition,
    condition_fingerprint,
    record_verified_condition_correction,
)


def test_canonical_condition_ignores_only_formatting_whitespace() -> None:
    left = "Решите:\n\n$ x+1=0 $"
    right = "Решите:  $x+1=0$"
    assert canonical_condition(left) == canonical_condition(right)
    assert condition_fingerprint(left) == condition_fingerprint(right)


def test_canonical_condition_keeps_math_content_strict() -> None:
    assert canonical_condition("Решите $x+1=0$") != canonical_condition(
        "Решите $x+2=0$"
    )
    assert condition_fingerprint("Решите $x+1=0$") != condition_fingerprint(
        "Решите $x+2=0$"
    )


def test_records_and_applies_verified_condition_correction(tmp_path) -> None:
    store = DataStore(tmp_path / "data")
    source = "Вычислите:\n\nB446\n\n1) 9"
    corrected = (
        "Вычислите: $\\frac{\\sqrt{486}}{\\sqrt{6}}$\n\n"
        "1) 9"
    )

    path = record_verified_condition_correction(
        source,
        corrected,
        note="verified against source PDF",
        data_store=store,
    )

    assert path.is_file()
    assert apply_verified_condition_correction(
        source,
        data_store=store,
    ) == corrected


def test_applies_saved_correction_when_only_math_edge_spaces_differ(tmp_path) -> None:
    store = DataStore(tmp_path / "data")
    source = (
        "B446\n\n"
        "1) 9\n\n"
        "2) $\\sqrt{480}$\n\n"
        "3) 81\n\n"
        "4) $\\sqrt{8}$"
    )
    runtime_source = (
        "B446\n\n"
        "1) 9\n\n"
        "2)  $ \\sqrt{480} $\n\n"
        "3) 81\n\n"
        "4)  $ \\sqrt{8} $"
    )
    corrected = (
        "Вычислите: $\\frac{\\sqrt{486}}{\\sqrt{6}}$\n\n"
        "1) 9\n\n"
        "2) $\\sqrt{480}$\n\n"
        "3) 81\n\n"
        "4) $\\sqrt{8}$"
    )

    record_verified_condition_correction(
        source,
        corrected,
        data_store=store,
    )

    assert apply_verified_condition_correction(
        runtime_source,
        data_store=store,
    ) == corrected


def test_does_not_apply_to_different_condition(tmp_path) -> None:
    store = DataStore(tmp_path / "data")
    record_verified_condition_correction(
        "Задача $x=1$.",
        "Задача $x=2$.",
        data_store=store,
    )

    untouched = "Задача $x=3$."
    assert apply_verified_condition_correction(
        untouched,
        data_store=store,
    ) == untouched


def test_updates_existing_fingerprint_instead_of_duplicating(tmp_path) -> None:
    store = DataStore(tmp_path / "data")
    source = "Найдите $x$."
    record_verified_condition_correction(
        source,
        "Найдите $y$.",
        data_store=store,
    )
    path = record_verified_condition_correction(
        source,
        "Найдите $z$.",
        data_store=store,
    )

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert apply_verified_condition_correction(
        source,
        data_store=store,
    ) == "Найдите $z$."


def test_rejects_noop_correction(tmp_path) -> None:
    store = DataStore(tmp_path / "data")
    try:
        record_verified_condition_correction(
            "Решите: $x=0$",
            "Решите:\n\n$x=0$",
            data_store=store,
        )
    except ValueError as error:
        assert "не изменяет" in str(error)
    else:
        raise AssertionError("noop correction must be rejected")
