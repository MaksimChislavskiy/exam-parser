from __future__ import annotations

import sys
from typing import Any


def configure_utf8_console() -> None:
    """Переключает текстовые потоки на UTF-8, если они это поддерживают.

    На Windows при перенаправлении вывода через ``tee`` Python иногда
    выбирает системную однобайтовую кодировку. Тогда обычные математические
    символы вроде ``≤`` могут вызвать ``UnicodeEncodeError`` прямо во время
    диагностического ``print``. Ошибки замены экранируются, чтобы служебный
    вывод никогда не прерывал обработку документа.
    """

    for stream_name in ("stdout", "stderr"):
        stream: Any = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            # Некоторые обёртки терминала запрещают менять параметры потока.
            # В таком окружении оставляем исходную настройку.
            continue


def main() -> None:
    """Запускает CLI после подключения детерминированных исправлений."""

    configure_utf8_console()

    from .pipeline_runtime_v7 import install_runtime_repairs

    install_runtime_repairs()

    from .cli import main as cli_main

    cli_main()
