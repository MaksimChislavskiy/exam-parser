from __future__ import annotations


def main() -> None:
    """Запускает CLI после подключения детерминированных исправлений."""

    from .pipeline_runtime_v7 import install_runtime_repairs

    install_runtime_repairs()

    from .cli import main as cli_main

    cli_main()
