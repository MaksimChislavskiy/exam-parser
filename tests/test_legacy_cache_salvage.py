import json
from pathlib import Path

from exam_parser import markdown_pipeline as pipeline
from exam_parser.legacy_cache_salvage import _legacy_payloads, _salvage_explicit_tasks


class Cache:
    provider = "deepseek"
    model = "deepseek-v4-pro"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir


def _write_legacy(cache_dir: Path, page: int, tasks: list[dict]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"page_{page}_legacy.json").write_text(
        json.dumps(
            {
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "tasks": tasks,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_salvage_rebuilds_condition_from_current_explicit_ocr(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    image_dir = tmp_path / "imgs"
    image_dir.mkdir()
    _write_legacy(
        cache_dir,
        5,
        [
            {"task_num": "10", "condition": "Старый ответ модели", "image_id": None},
            {"task_num": "O_{T\\beta t}:", "condition": "Мусор", "image_id": None},
        ],
    )
    markdown = "10 Найдите ускорение автомобиля, если известны путь и скорость."
    candidates = _legacy_payloads(Cache(cache_dir), 5)

    salvaged = _salvage_explicit_tasks(
        pipeline,
        Cache(cache_dir),
        markdown,
        [],
        image_dir=image_dir,
        candidates=candidates,
    )

    assert salvaged is not None
    assert [task.task_num for task in salvaged] == ["10"]
    assert salvaged[0].condition == "Найдите ускорение автомобиля, если известны путь и скорость."


def test_salvage_does_not_trust_unlabelled_legacy_task(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    image_dir = tmp_path / "imgs"
    image_dir.mkdir()
    _write_legacy(
        cache_dir,
        6,
        [{"task_num": "14", "condition": "Придуманное условие", "image_id": None}],
    )
    markdown = (
        "13 Решите уравнение.\n\n"
        "В пирамиде дана сторона основания. Докажите равенство и найдите объём.\n\n"
        "15 Решите неравенство."
    )
    candidates = _legacy_payloads(Cache(cache_dir), 6)

    salvaged = _salvage_explicit_tasks(
        pipeline,
        Cache(cache_dir),
        markdown,
        [],
        image_dir=image_dir,
        candidates=candidates,
    )

    assert salvaged is None
