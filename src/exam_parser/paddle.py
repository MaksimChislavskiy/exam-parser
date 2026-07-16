from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def recognize_pages(
    pages: list[Path],
    markdown_dir: str | Path,
    *,
    device: str = "gpu:0",
) -> list[Path]:
    import paddle
    from paddleocr import PaddleOCRVL

    selected_device = configure_paddle_device(device, paddle)
    markdown_dir = Path(markdown_dir)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    print(f"Загрузка PaddleOCR-VL; устройство: {selected_device}", flush=True)
    pipeline = PaddleOCRVL(pipeline_version="v1", device=selected_device)
    markdown_files: list[Path] = []

    for page_num, page_path in enumerate(pages, start=1):
        print(f"PaddleOCR: страница {page_num}/{len(pages)}", flush=True)
        page_dir = markdown_dir / f"page_{page_num}"
        if page_dir.exists():
            shutil.rmtree(page_dir)
        page_dir.mkdir(parents=True)

        results = pipeline.predict(str(page_path))
        for result in results:
            result.save_to_markdown(save_path=str(page_dir))

        markdown_path = page_dir / f"page_{page_num}.md"
        if not markdown_path.is_file():
            raise RuntimeError(f"PaddleOCR не создал {markdown_path}")
        markdown_files.append(markdown_path)
    return markdown_files


def configure_paddle_device(requested: str, paddle_module: Any) -> str:
    """Выбирает устройство без скрытого отката с GPU на CPU."""

    requested = requested.strip().lower()
    if requested == "auto":
        selected = "gpu:0" if _cuda_is_available(paddle_module) else "cpu"
    elif requested == "cpu":
        selected = "cpu"
    elif requested.startswith("gpu"):
        if not _cuda_is_available(paddle_module):
            raise RuntimeError(
                "Запрошен GPU, но установленная сборка Paddle не видит CUDA. "
                "Проверьте, что установлен paddlepaddle-gpu, а не CPU-пакет "
                "paddlepaddle, и что версии драйвера/CUDA совместимы. "
                "Для осознанного запуска на CPU укажите --device cpu."
            )
        selected = requested
    else:
        raise ValueError("--device должен быть gpu:0, cpu или auto")

    paddle_module.set_device(selected)
    actual = str(paddle_module.get_device()).lower()
    if selected.startswith("gpu") and not actual.startswith("gpu"):
        raise RuntimeError(
            f"Paddle сообщил устройство {actual!r} вместо запрошенного {selected!r}"
        )
    print(
        "Paddle CUDA: "
        f"compiled={_compiled_with_cuda(paddle_module)}, "
        f"gpu_count={_gpu_count(paddle_module)}, actual={actual}",
        flush=True,
    )
    return selected


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
