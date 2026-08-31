from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from .data_store import DataStore, resolve_data_store
from .ocr_noise import (
    OCR_UNREADABLE_REPEAT_MARKER,
    sanitize_pathological_ocr_repetitions,
)


_TASK_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?(?:№[ \t]*)?"
    r"(?P<num>(?:[1-9]\d?)(?:\.\d+)?|[AАBВCС](?:1[0-9]|[1-9]))"
    r"(?P<suffix>[.)]?[ \t]+|[.)]?[ \t]*$)"
)
_IMAGE_REFERENCE_PATTERN = re.compile(
    r"(?:!\[[^]]*]\([^)]+\)|<img\b)",
    re.IGNORECASE,
)
_SOLUTION_HEADING_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:Решение|Ответ)[ \t]*:"
)
_WORD_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")
_CYRILLIC_LETTER_PATTERN = re.compile(r"[А-Яа-яЁё]")
_ALPHABETIC_LETTER_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё]")
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


class PaddleDeviceError(RuntimeError):
    """Ошибка выбора устройства для PaddleOCR."""


class CpuFallbackDeclined(PaddleDeviceError):
    """Пользователь отказался продолжать обработку на CPU."""


def recognize_pages(
    pages: list[Path],
    markdown_dir: str | Path,
    *,
    device: str = "gpu:0",
    allow_cpu_fallback: bool = False,
    data_store: DataStore | None = None,
) -> list[Path]:
    import os

    dll_directory_handles = []

    if sys.platform == "win32":
        nvidia_packages_dir = (
            Path(sys.prefix)
            / "Lib"
            / "site-packages"
            / "nvidia"
        )

        if nvidia_packages_dir.is_dir():
            for bin_dir in nvidia_packages_dir.glob("*/bin"):
                if bin_dir.is_dir():
                    dll_directory_handles.append(
                        os.add_dll_directory(str(bin_dir))
                    )

    # На Windows Torch нужно загрузить раньше Paddle,
    # иначе возможен конфликт нативных DLL.
    import torch  # noqa: F401

    import paddle
    from paddleocr import PaddleOCRVL

    selected_device = configure_paddle_device(
        device,
        paddle,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    markdown_dir = Path(markdown_dir)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    print(f"Загрузка PaddleOCR-VL; устройство: {selected_device}", flush=True)
    pipeline = PaddleOCRVL(pipeline_version="v1", device=selected_device)
    review_root = (data_store or resolve_data_store()).ocr_review_dir
    markdown_files: list[Path] = []

    for page_num, page_path in enumerate(pages, start=1):
        print(f"PaddleOCR: страница {page_num}/{len(pages)}", flush=True)
        page_dir = markdown_dir / f"page_{page_num}"
        if page_dir.exists():
            shutil.rmtree(page_dir)
        page_dir.mkdir(parents=True)

        results = list(pipeline.predict(str(page_path)))
        for result in results:
            result.save_to_markdown(save_path=str(page_dir))

        markdown_path = page_dir / f"page_{page_num}.md"
        if not markdown_path.is_file():
            raise RuntimeError(f"PaddleOCR не создал {markdown_path}")
        _recover_pathological_ocr_blocks(
            pipeline,
            page_path,
            markdown_path,
            results,
            page_num=page_num,
            review_root=review_root,
        )
        markdown_files.append(markdown_path)
    return markdown_files


def _recover_pathological_ocr_blocks(
    pipeline: Any,
    page_path: Path,
    markdown_path: Path,
    results: list[Any],
    *,
    page_num: int,
    review_root: Path | None = None,
) -> int:
    """Повторно распознаёт только layout-блок с патологическим OCR-повтором.

    Полная страница уже прошла обычный PaddleOCR-VL. Для доказанно повреждённого
    блока используются его штатные координаты ``block_bbox``: область берётся
    из исходной PNG, мягко увеличивается и ещё раз проходит тот же загруженный
    pipeline. Если повтор не исчез либо изменился номер задания, исходный
    Markdown остаётся без изменений и позднее срабатывает ``OCRQualityError``.
    """

    if not page_path.is_file() or not markdown_path.is_file():
        return 0

    markdown = markdown_path.read_text(encoding="utf-8")
    recovered_count = 0

    with tempfile.TemporaryDirectory(prefix="exam_parser_ocr_recovery_") as raw:
        temp_dir = Path(raw)
        source_blocks = _prediction_blocks(results, temp_dir, prefix="source")
        noisy_blocks = [
            block
            for block in source_blocks
            if _has_pathological_repetition(block.get("block_content"))
        ]
        if not noisy_blocks:
            return 0

        with Image.open(page_path) as opened_page:
            page = opened_page.convert("RGB")
            for index, block in enumerate(noisy_blocks, start=1):
                original = block.get("block_content")
                bbox = _block_bbox(block.get("block_bbox"), page.size)
                if not isinstance(original, str) or bbox is None:
                    continue

                print(
                    f"PaddleOCR: страница {page_num}, повторное распознавание "
                    f"повреждённой области {index}/{len(noisy_blocks)}",
                    flush=True,
                )
                crop = _recovery_crop(page, bbox)
                review_item_dir = (
                    _ocr_review_item_dir(review_root, crop)
                    if review_root is not None
                    else None
                )
                recovered = None
                if review_item_dir is not None:
                    correction_path = review_item_dir / "correction.md"
                    if correction_path.is_file():
                        try:
                            correction = correction_path.read_text(
                                encoding="utf-8"
                            )
                        except (OSError, UnicodeError):
                            correction = ""
                        recovered = _safe_recovered_block(original, correction)
                        if recovered is not None:
                            print(
                                f"PaddleOCR: страница {page_num}, загружена "
                                "проверенная OCR-правка из Дата-центра",
                                flush=True,
                            )
                        else:
                            print(
                                f"PaddleOCR: страница {page_num}, файл "
                                f"{correction_path} отклонён проверкой качества",
                                flush=True,
                            )
                if recovered is None:
                    recovered = _retry_recovery_variants(
                        pipeline,
                        crop,
                        original,
                        temp_dir,
                        block_index=index,
                        page_num=page_num,
                    )
                if recovered is None:
                    if review_item_dir is not None:
                        correction_path = _write_ocr_review_item(
                            review_item_dir,
                            crop,
                            original,
                            bbox=bbox,
                            page_size=page.size,
                        )
                        print(
                            f"PaddleOCR: страница {page_num}, требуется ручная "
                            f"сверка OCR: {correction_path}",
                            flush=True,
                        )
                    continue

                replaced = _replace_block_once(markdown, original, recovered)
                if replaced is None:
                    continue
                markdown = replaced
                recovered_count += 1
                print(
                    f"PaddleOCR: страница {page_num}, повреждённая область "
                    "восстановлена повторным OCR",
                    flush=True,
                )

    if recovered_count:
        markdown_path.write_text(markdown, encoding="utf-8")
    return recovered_count


def _ocr_review_item_dir(review_root: Path, crop: Image.Image) -> Path:
    digest = hashlib.sha256()
    digest.update(f"{crop.mode}:{crop.width}x{crop.height}\0".encode("ascii"))
    digest.update(crop.tobytes())
    return review_root / digest.hexdigest()


def _write_ocr_review_item(
    item_dir: Path,
    crop: Image.Image,
    original: str,
    *,
    bbox: tuple[int, int, int, int],
    page_size: tuple[int, int],
) -> Path:
    item_dir.mkdir(parents=True, exist_ok=True)
    source_path = item_dir / "source.png"
    if not source_path.is_file():
        crop.save(source_path, "PNG", optimize=True, dpi=(300, 300))

    cleaned_original, _replacements = sanitize_pathological_ocr_repetitions(
        original
    )
    (item_dir / "original_ocr.md").write_text(
        cleaned_original.strip() + "\n",
        encoding="utf-8",
    )

    heading = _TASK_HEADING_PATTERN.search(original)
    metadata = {
        "schema_version": 1,
        "fingerprint": item_dir.name,
        "task_num": (
            _canonical_task_num(heading.group("num"))
            if heading is not None
            else None
        ),
        "bbox": list(bbox),
        "page_size": list(page_size),
        "crop_size": [crop.width, crop.height],
    }
    (item_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme_path = item_dir / "README.txt"
    if not readme_path.is_file():
        readme_path.write_text(
            "Проверьте source.png по исходному документу.\n"
            "Создайте correction.md только после ручной сверки.\n"
            "В correction.md запишите номер и полное условие задачи без решения.\n"
            "Следующий OCR-запуск проверит номер и качество текста автоматически.\n",
            encoding="utf-8",
        )
    return item_dir / "correction.md"


def _retry_recovery_variants(
    pipeline: Any,
    crop: Image.Image,
    original: str,
    temp_dir: Path,
    *,
    block_index: int,
    page_num: int,
) -> str | None:
    variants = _recovery_crop_variants(crop)
    for variant_index, (variant_name, variant) in enumerate(variants, start=1):
        if variant_index > 1:
            print(
                f"PaddleOCR: страница {page_num}, уточнённый crop "
                f"{variant_index - 1}/{len(variants) - 1} "
                f"({variant_name})",
                flush=True,
            )
        crop_path = temp_dir / f"crop_{block_index}_{variant_index}.png"
        variant.save(crop_path, "PNG", optimize=True, dpi=(300, 300))
        try:
            retry_results = list(pipeline.predict(str(crop_path)))
        except (OSError, RuntimeError, ValueError, TypeError):
            continue

        retry_blocks = _prediction_blocks(
            retry_results,
            temp_dir,
            prefix=f"retry_{block_index}_{variant_index}",
        )
        recovered = _safe_recovered_block(
            original,
            _blocks_markdown(retry_blocks),
        )
        if recovered is not None:
            return recovered
    return None


def _prediction_blocks(
    results: list[Any],
    temp_dir: Path,
    *,
    prefix: str,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        save_to_json = getattr(result, "save_to_json", None)
        if not callable(save_to_json):
            continue
        json_path = temp_dir / f"{prefix}_{index}.json"
        try:
            save_to_json(save_path=str(json_path))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError):
            continue
        blocks.extend(_first_parsing_blocks(payload))
    return blocks


def _first_parsing_blocks(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        parsing = value.get("parsing_res_list")
        if isinstance(parsing, list):
            return [item for item in parsing if isinstance(item, dict)]
        for nested in value.values():
            found = _first_parsing_blocks(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _first_parsing_blocks(nested)
            if found:
                return found
    return []


def _has_pathological_repetition(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    _cleaned, replacements = sanitize_pathological_ocr_repetitions(value)
    return bool(replacements)


def _block_bbox(
    value: Any,
    page_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        if len(value) == 4 and all(
            not isinstance(item, (list, tuple)) for item in value
        ):
            x1, y1, x2, y2 = (int(round(float(item))) for item in value)
        else:
            points = [
                (float(point[0]), float(point[1]))
                for point in value
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
            if len(points) < 2:
                return None
            x1 = int(round(min(point[0] for point in points)))
            y1 = int(round(min(point[1] for point in points)))
            x2 = int(round(max(point[0] for point in points)))
            y2 = int(round(max(point[1] for point in points)))
    except (TypeError, ValueError, OverflowError):
        return None
    width, height = page_size
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def _recovery_crop(
    page: Image.Image,
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    x1, y1, x2, y2 = bbox
    margin_x = max(12, round((x2 - x1) * 0.06))
    margin_y = max(12, round((y2 - y1) * 0.10))
    expanded = (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(page.width, x2 + margin_x),
        min(page.height, y2 + margin_y),
    )
    crop = page.crop(expanded)
    longest_side = max(crop.size)
    if longest_side and longest_side < 1800:
        scale = min(3, max(2, math.ceil(1800 / longest_side)))
        crop = crop.resize(
            (crop.width * scale, crop.height * scale),
            Image.Resampling.LANCZOS,
        )
    return crop


def _recovery_crop_variants(
    crop: Image.Image,
) -> list[tuple[str, Image.Image]]:
    variants = [("полный", crop)]
    for name, height_ratio in (("верхние 60%", 0.60), ("верхние 42%", 0.42)):
        height = max(1, round(crop.height * height_ratio))
        upper = crop.crop((0, 0, crop.width, height))
        gray = ImageOps.grayscale(upper)
        enhanced = ImageOps.autocontrast(gray, cutoff=1).filter(
            ImageFilter.UnsharpMask(radius=2, percent=170, threshold=3)
        )
        variants.append((name, enhanced.convert("RGB")))
    return variants


def _blocks_markdown(blocks: list[dict[str, Any]]) -> str:
    ordered = sorted(
        enumerate(blocks),
        key=lambda item: (
            item[1].get("block_order") is None,
            item[1].get("block_order")
            if isinstance(item[1].get("block_order"), int)
            else item[0],
        ),
    )
    contents = [
        str(block.get("block_content", "")).strip()
        for _index, block in ordered
        if str(block.get("block_content", "")).strip()
    ]
    return "\n\n".join(contents).strip()


def _safe_recovered_block(original: str, recovered: str) -> str | None:
    if not recovered or _IMAGE_REFERENCE_PATTERN.search(recovered):
        return None

    original_heading = _TASK_HEADING_PATTERN.search(original)
    recovered_headings = list(_TASK_HEADING_PATTERN.finditer(recovered))
    if original_heading is not None:
        original_num = _canonical_task_num(original_heading.group("num"))
        matching_heading = next(
            (
                heading
                for heading in recovered_headings
                if _canonical_task_num(heading.group("num")) == original_num
            ),
            None,
        )
        if matching_heading is not None:
            recovered = recovered[matching_heading.start() :]
            for heading in _TASK_HEADING_PATTERN.finditer(
                recovered,
                matching_heading.end() - matching_heading.start(),
            ):
                if _canonical_task_num(heading.group("num")) != original_num:
                    recovered = recovered[: heading.start()]
                    break
        elif recovered_headings:
            return None
        else:
            prefix = original[original_heading.start() : original_heading.end()]
            recovered = prefix + recovered.lstrip()

    solution_heading = _SOLUTION_HEADING_PATTERN.search(recovered)
    if solution_heading is not None:
        recovered = recovered[: solution_heading.start()]
    recovered = recovered.strip()

    cleaned, replacements = sanitize_pathological_ocr_repetitions(recovered)
    if replacements or OCR_UNREADABLE_REPEAT_MARKER in cleaned:
        return None
    if len(re.findall(r"[0-9A-Za-zА-Яа-яЁё]", recovered)) < 8:
        return None
    if _looks_like_ocr_hallucination(recovered):
        return None
    return recovered


def _looks_like_ocr_hallucination(value: str) -> bool:
    if _CJK_PATTERN.search(value):
        return True

    letters = _ALPHABETIC_LETTER_PATTERN.findall(value)
    if len(letters) >= 20:
        cyrillic = len(_CYRILLIC_LETTER_PATTERN.findall(value))
        if cyrillic / len(letters) < 0.35:
            return True

    words = [word.casefold() for word in _WORD_PATTERN.findall(value)]
    if len(words) < 12:
        return False
    counts = Counter(words)
    most_common = counts.most_common(1)[0][1]
    if most_common >= 6 and most_common / len(words) >= 0.20:
        return True
    return False


def _canonical_task_num(value: str) -> str:
    return value.translate(str.maketrans({"А": "A", "В": "B", "С": "C"}))


def _replace_block_once(
    markdown: str,
    original: str,
    recovered: str,
) -> str | None:
    if original in markdown:
        return markdown.replace(original, recovered, 1)
    stripped = original.strip()
    if stripped and stripped in markdown:
        return markdown.replace(stripped, recovered, 1)
    return None


def configure_paddle_device(
    requested: str,
    paddle_module: Any,
    *,
    allow_cpu_fallback: bool = False,
    interactive: bool | None = None,
    input_func: Callable[[str], str] = input,
) -> str:
    """Выбирает устройство и не переключается на CPU без согласия пользователя."""

    requested = requested.strip().lower()
    if requested == "cpu":
        selected = "cpu"
    elif requested in {"auto"} or requested.startswith("gpu"):
        if _cuda_is_available(paddle_module):
            selected = "gpu:0" if requested == "auto" else requested
        else:
            selected = _request_cpu_fallback(
                allow_cpu_fallback=allow_cpu_fallback,
                interactive=interactive,
                input_func=input_func,
            )
    else:
        raise ValueError("--device должен быть gpu:0, cpu или auto")

    paddle_module.set_device(selected)
    actual = str(paddle_module.get_device()).lower()
    if selected.startswith("gpu") and not actual.startswith("gpu"):
        raise PaddleDeviceError(
            f"Paddle сообщил устройство {actual!r} вместо запрошенного {selected!r}"
        )
    print(
        "Paddle CUDA: "
        f"compiled={_compiled_with_cuda(paddle_module)}, "
        f"gpu_count={_gpu_count(paddle_module)}, actual={actual}",
        flush=True,
    )
    return selected


def _request_cpu_fallback(
    *,
    allow_cpu_fallback: bool,
    interactive: bool | None,
    input_func: Callable[[str], str],
) -> str:
    print(
        "GPU недоступен: установленная сборка Paddle не видит CUDA.\n"
        "PaddleOCR-VL может обрабатывать одну страницу на CPU десятки минут.",
        flush=True,
    )

    if allow_cpu_fallback:
        print(
            "Переход на CPU разрешён параметром --allow-cpu-fallback.",
            flush=True,
        )
        return "cpu"

    if interactive is None:
        interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
    if not interactive:
        raise PaddleDeviceError(
            "GPU недоступен, а запрос подтверждения невозможен в неинтерактивном "
            "запуске. Укажите --allow-cpu-fallback для автоматического перехода "
            "или --device cpu для явного запуска на CPU."
        )

    try:
        answer = input_func("Продолжить на CPU? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt) as error:
        raise CpuFallbackDeclined("Запуск на CPU отменён пользователем.") from error

    if answer in {"y", "yes", "д", "да"}:
        return "cpu"
    raise CpuFallbackDeclined("Запуск на CPU отменён пользователем.")


def _compiled_with_cuda(paddle_module: Any) -> bool:
    checker = getattr(paddle_module, "is_compiled_with_cuda", None)
    return bool(checker and checker())


def _gpu_count(paddle_module: Any) -> int:
    try:
        return int(paddle_module.device.cuda.device_count())
    except (AttributeError, RuntimeError):
        return 0


def _cuda_is_available(paddle_module: Any) -> bool:
    return _compiled_with_cuda(paddle_module) and _gpu_count(paddle_module) > 0
