from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


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
