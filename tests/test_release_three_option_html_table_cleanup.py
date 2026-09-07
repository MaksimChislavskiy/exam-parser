from exam_parser.release_condition_cleanup import clean_release_condition


def test_repairs_three_option_taxi_table_label() -> None:
    value = (
        "<table>"
        "<tr><td>Фирма такси</td><td>Цена</td></tr>"
        "<tr><td>A</td><td>100</td></tr>"
        "<tr><td>B</td><td>200</td></tr>"
        "<tr><td>B</td><td>300</td></tr>"
        "</table>"
    )

    expected = (
        "<table>"
        "<tr><td>Фирма такси</td><td>Цена</td></tr>"
        "<tr><td>A</td><td>100</td></tr>"
        "<tr><td>B</td><td>200</td></tr>"
        "<tr><td>C</td><td>300</td></tr>"
        "</table>"
    )

    assert clean_release_condition(value) == expected


def test_repairs_three_option_store_table_label() -> None:
    value = (
        "<table>"
        "<tr><td>Интернет-магазин</td><td>Цена</td></tr>"
        "<tr><td>A</td><td>100</td></tr>"
        "<tr><td>B</td><td>200</td></tr>"
        "<tr><td>B</td><td>300</td></tr>"
        "</table>"
    )

    result = clean_release_condition(value)

    assert "<tr><td>C</td><td>300</td></tr>" in result


def test_repairs_three_option_carrier_table_label() -> None:
    value = (
        "<table>"
        "<tr><td>Перевозчик</td><td>Цена</td></tr>"
        "<tr><td>A</td><td>100</td></tr>"
        "<tr><td>B</td><td>200</td></tr>"
        "<tr><td>B</td><td>300</td></tr>"
        "</table>"
    )

    result = clean_release_condition(value)

    assert "<tr><td>C</td><td>300</td></tr>" in result


def test_preserves_four_option_model_table_with_ambiguous_third_label() -> None:
    value = (
        "<table>"
        "<tr><td>Модель чайника</td><td>Цена</td></tr>"
        "<tr><td>A</td><td>100</td></tr>"
        "<tr><td>B</td><td>200</td></tr>"
        "<tr><td>B</td><td>300</td></tr>"
        "<tr><td>Г</td><td>400</td></tr>"
        "</table>"
    )

    assert clean_release_condition(value) == value


def test_preserves_three_row_symbol_table() -> None:
    value = (
        "<table>"
        "<tr><td>Символ</td><td>Значение</td></tr>"
        "<tr><td>A</td><td>1</td></tr>"
        "<tr><td>B</td><td>2</td></tr>"
        "<tr><td>B</td><td>3</td></tr>"
        "</table>"
    )

    assert clean_release_condition(value) == value
