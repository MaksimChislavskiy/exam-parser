"""Структурные правила восстановления номеров длинных заданий 13–19."""

from __future__ import annotations

import re


TASK_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*((?:1[0-9]|[1-9])(?:\.\d+)*)"
    r"(?:\.[ \t]+|[ \t]+(?=[A-Za-zА-Яа-яЁё0-9])|[ \t]*$)"
)
EXPLICIT_LABEL_PATTERN = re.compile(
    r"^[ \t]*#{0,6}[ \t]*Задание[ \t]*(?:№[ \t]*)?"
    r"(?P<num>(?:1[0-9]|[1-9]))[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
PART_TWO_PATTERN = re.compile(
    r"^\s*(?:#{0,6}\s*Часть\s*2\b|"
    r"<div[^>]*>\s*Часть\s*2\s*</div>)",
    re.IGNORECASE | re.MULTILINE,
)
LONG_HEADING_PATTERN = re.compile(
    r"^[ \t]*(?P<num>1[3-9])"
    r"(?:\.[ \t]*(?=\n|[А-ЯЁA-Zа-яёa-z])|"
    r"[ \t]+(?=[А-ЯЁA-Zа-яёa-z]))",
    re.MULTILINE,
)
SUBPART_A_PATTERN = re.compile(
    r"^[ \t]*(?:а|a)\)[ \t]+(?=[А-ЯЁ])",
    re.IGNORECASE | re.MULTILINE,
)
SOLUTION_SECTION_PATTERN = re.compile(
    r"^[ \t]*#{0,6}[ \t]*(?:Ответ|Решение)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
SERVICE_START_PATTERN = re.compile(
    r"^(?:Инструкция|Часть\s*[12]|Тренировочная\s+работа|"
    r"Единый\s+государственный\s+экзамен|Справочные\s+материалы)\b",
    re.IGNORECASE | re.DOTALL,
)
CONTINUATION_PATTERN = re.compile(
    r"^(?:Тогда|Следовательно|Получаем|Отсюда|Так\s+как|Поэтому|"
    r"Пусть\s+планируется|Воспользуемся|Зная)\b",
    re.IGNORECASE | re.DOTALL,
)
TASK_START_PATTERN = re.compile(
    r"^(?:Найдите|Решите|Докажите|Определите|Вычислите|Укажите|"
    r"Постройте|Исследуйте|Окружность|В\s+параллелограмме|"
    r"Каждый|Шесть|Восемь|Юра\s+и\s+Полина|Точки|Прямая|"
    r"Функция|Критики|В\s+июле)\b",
    re.IGNORECASE | re.DOTALL,
)


def repair_page_group(source_texts: list[str]) -> list[str]:
    texts = _normalize_explicit_labels(source_texts)
    texts = _restore_task_13(texts)
    texts = _restore_prefix_task(texts)
    return _restore_complete_solution_task(texts)


def _normalize_explicit_labels(source_texts: list[str]) -> list[str]:
    texts = list(source_texts)
    carry: int | None = None
    for index, source in enumerate(texts):
        text = source
        if carry is not None:
            if not _starts_with_number(text, carry):
                text = _prepend_number(text, carry)
            carry = None

        edits: list[tuple[int, int, str]] = []
        for match in EXPLICIT_LABEL_PATTERN.finditer(text):
            number = int(match.group("num"))
            if not text[match.end() :].strip() and index + 1 < len(texts):
                edits.append((match.start(), match.end(), ""))
                carry = number
            else:
                edits.append((match.start(), match.end(), f"{number}. "))
        for start, end, replacement in reversed(edits):
            text = text[:start] + replacement + text[end:]
        texts[index] = text

    if carry is not None and texts:
        texts[-1] = texts[-1].rstrip() + f"\n\n{carry}. \n"
    return texts


def _restore_task_13(source_texts: list[str]) -> list[str]:
    texts = list(source_texts)
    for index, source in enumerate(texts):
        headings = _headings(source)
        if any(int(match.group("num")) == 13 for match in headings):
            continue
        part_two = PART_TWO_PATTERN.search(source)
        if part_two is None:
            continue
        following = [m for m in headings if m.start() > part_two.end()]
        if not following or int(following[0].group("num")) != 14:
            continue
        between = source[part_two.end() : following[0].start()]
        subparts = list(SUBPART_A_PATTERN.finditer(between))
        if len(subparts) != 1:
            continue
        position = part_two.end() + subparts[0].start()
        texts[index] = source[:position] + "13. " + source[position:]
    return texts


def _restore_prefix_task(source_texts: list[str]) -> list[str]:
    texts = list(source_texts)
    previous: int | None = None
    for index, source in enumerate(texts):
        text = source
        headings = _headings(text)
        if headings:
            first = headings[0]
            first_num = int(first.group("num"))
            if (
                previous is not None
                and first_num == previous + 2
                and _looks_like_task(text[: first.start()])
            ):
                text = _prepend_number(text, previous + 1)
                texts[index] = text
                headings = _headings(text)
            if headings:
                previous = max(int(m.group("num")) for m in headings)
    return texts


def _restore_complete_solution_task(source_texts: list[str]) -> list[str]:
    texts = list(source_texts)
    previous: int | None = None
    for index, source in enumerate(texts):
        headings = _headings(source)
        if headings:
            previous = max(int(m.group("num")) for m in headings)
            continue
        if previous is None or index + 1 >= len(texts):
            continue
        following = _headings(texts[index + 1])
        if not following or int(following[0].group("num")) != previous + 2:
            continue
        if not _looks_like_task(source) or not SOLUTION_SECTION_PATTERN.search(source):
            continue
        texts[index] = _prepend_number(source, previous + 1)
        previous += 1
    return texts


def _headings(value: str) -> list[re.Match[str]]:
    result: list[re.Match[str]] = []
    for match in LONG_HEADING_PATTERN.finditer(value):
        following = HTML_TAG_PATTERN.sub(
            " ", value[match.end() : match.end() + 250]
        ).strip()
        if not following or re.match(
            r"^(?:класс|года|вариант|профиль)", following, re.IGNORECASE
        ):
            continue
        if re.match(
            r"^(?:[а-вa-c]\)\s+|[А-ЯЁA-Z]|\$)",
            following,
            re.IGNORECASE | re.DOTALL,
        ):
            result.append(match)
    return result


def _looks_like_task(value: str) -> bool:
    visible = re.sub(r"\s+", " ", HTML_TAG_PATTERN.sub(" ", value)).strip()
    if len(visible) < 80 or SERVICE_START_PATTERN.match(visible):
        return False
    if CONTINUATION_PATTERN.match(visible):
        return False
    return bool(
        TASK_START_PATTERN.match(visible)
        or re.match(r"^(?:а|a)\)\s+", visible, re.IGNORECASE)
    )


def _starts_with_number(value: str, number: int) -> bool:
    match = TASK_HEADING_PATTERN.match(value.lstrip())
    return bool(match and int(match.group(1).split(".", 1)[0]) == number)


def _prepend_number(value: str, number: int) -> str:
    leading = len(value) - len(value.lstrip())
    return value[:leading] + f"{number}. " + value[leading:]
