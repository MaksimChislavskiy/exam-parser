from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageOps

from .pipeline_runtime_v4 import install_runtime_repairs as install_v4_repairs


_INSTALLED = False


def copy_task_image_without_resampling(
    markdown_path: Path,
    image_id: str | None,
    images_dir: Path,
    task_num: str,
) -> str | None:
    """Сохраняет прямой кроп страницы без увеличения и фильтрации.

    Paddle по-прежнему определяет блок изображения и записывает ссылку в
    Markdown. Базовый runtime заменяет Paddle-кроп прямым кропом PNG-страницы.
    На этом этапе мы лишь переводим найденный файл в итоговый lossless PNG:
    размеры и значения пикселей не меняются, резкость и бинаризация не
    применяются.
    """

    if not image_id:
        return None

    source = markdown_path.parent / "imgs" / Path(image_id).name
    if not source.is_file():
        print(f"Картинка не найдена: {source}", flush=True)
        return None

    safe_num = re.sub(
        r"[^0-9A-Za-zА-Яа-я._-]+",
        "_",
        task_num,
    ).strip("._-")
    filename = f"task_{safe_num}.png"
    images_dir.mkdir(parents=True, exist_ok=True)
    destination = images_dir / filename

    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
            image = image.convert("RGB")
        image.save(destination, "PNG", optimize=True)

    return filename


def install_runtime_repairs() -> None:
    """Подключает безопасное сохранение изображений поверх исправлений v4."""

    global _INSTALLED
    if _INSTALLED:
        return

    # Намеренно пропускаем v5: его пороговая очистка уничтожала сглаживание
    # линий низкоразрешённых растров. v4 включает все проверенные текстовые
    # исправления и базовую замену Paddle-кропа прямым кропом страницы.
    install_v4_repairs()

    from . import markdown_pipeline as pipeline

    pipeline._copy_task_image = copy_task_image_without_resampling
    _INSTALLED = True
