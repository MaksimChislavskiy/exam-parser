from pathlib import Path
from tempfile import TemporaryDirectory

from exam_parser.extraction_cache import PageExtractionCache
from exam_parser.models import ExtractedTask


def test_prompt_version_change_is_a_cache_miss() -> None:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        image_dir = root / "imgs"
        image_dir.mkdir()
        cache_dir = root / "cache"
        markdown = "8. Условие.\nБезномерное условие.\n10. Следующее условие."
        tasks = [ExtractedTask(task_num="8", condition="Условие.")]

        first = PageExtractionCache(
            cache_dir,
            provider="deepseek",
            model="deepseek-test",
            prompt_version="v1",
        )
        first.save(
            1,
            markdown,
            [],
            tasks,
            image_dir=image_dir,
        )

        same = PageExtractionCache(
            cache_dir,
            provider="deepseek",
            model="deepseek-test",
            prompt_version="v1",
        )
        assert same.load(1, markdown, [], image_dir=image_dir) == tasks

        changed = PageExtractionCache(
            cache_dir,
            provider="deepseek",
            model="deepseek-test",
            prompt_version="v2",
        )
        assert changed.load(1, markdown, [], image_dir=image_dir) is None
