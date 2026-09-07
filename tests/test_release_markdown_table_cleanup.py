from exam_parser.release_condition_cleanup import clean_release_condition


def test_repairs_third_label_in_three_provider_markdown_table() -> None:
    value = (
        "Строительной фирме нужно выбрать поставщика.\n\n"
        "| Поставщик | Цена | Доставка |\n"
        "| --- | --- | --- |\n"
        "| A | 3900 | 10000 |\n"
        "| B | 4400 | 8000 |\n"
        "| B | 4000 | 8000 |\n"
    )

    expected = (
        "Строительной фирме нужно выбрать поставщика.\n\n"
        "| Поставщик | Цена | Доставка |\n"
        "| --- | --- | --- |\n"
        "| A | 3900 | 10000 |\n"
        "| B | 4400 | 8000 |\n"
        "| C | 4000 | 8000 |"
    )

    assert clean_release_condition(value) == expected


def test_repairs_compact_duplicate_provider_markdown_label() -> None:
    value = (
        "| Фирма | Цена |\n"
        "|---|---|\n"
        "| A | 100 |\n"
        "| B | 200 |\n"
        "|B | 300 |"
    )

    expected = (
        "| Фирма | Цена |\n"
        "|---|---|\n"
        "| A | 100 |\n"
        "| B | 200 |\n"
        "|C | 300 |"
    )

    assert clean_release_condition(value) == expected


def test_preserves_duplicate_labels_in_unrelated_markdown_table() -> None:
    value = (
        "| Символ | Значение |\n"
        "| --- | --- |\n"
        "| A | 1 |\n"
        "| B | 2 |\n"
        "| B | 3 |"
    )

    assert clean_release_condition(value) == value


def test_preserves_already_correct_provider_markdown_table() -> None:
    value = (
        "| Поставщик | Цена |\n"
        "| --- | --- |\n"
        "| A | 1 |\n"
        "| B | 2 |\n"
        "| C | 3 |"
    )

    assert clean_release_condition(value) == value
