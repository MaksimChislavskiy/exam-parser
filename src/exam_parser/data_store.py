from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_DATA_ROOT = Path.home() / ".exam-parser-data"
DATA_ROOT_ENV = "EXAM_PARSER_DATA_ROOT"


@dataclass(frozen=True)
class DataStore:
    """Пути внешнего хранилища данных, не привязанные к диску или машине."""

    root: Path

    @property
    def references_dir(self) -> Path:
        return self.root / "references"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def ocr_review_dir(self) -> Path:
        return self.dataset_dir / "ocr_review"

    def reference_path(self, filename: str) -> Path:
        if Path(filename).name != filename:
            raise ValueError(
                "Имя справочника должно быть именем файла без пути: "
                f"{filename!r}"
            )
        return self.references_dir / filename

    def ensure_layout(self) -> None:
        self.references_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def resolve_data_store(explicit_root: str | Path | None = None) -> DataStore:
    """Возвращает внешний data root или безопасный локальный fallback.

    Приоритет конфигурации: явный аргумент, переменная окружения процесса,
    корневой ``.env`` проекта, затем локальный fallback вне репозитория.
    DataStore не зависит от того, создавался ли до него LLM-клиент.
    """

    raw_root: str | Path | None = explicit_root or os.getenv(DATA_ROOT_ENV)
    if not raw_root:
        env_values = dotenv_values(PROJECT_DIR / ".env")
        raw_root = env_values.get(DATA_ROOT_ENV)

    if raw_root:
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            root = PROJECT_DIR / root
    else:
        root = DEFAULT_LOCAL_DATA_ROOT
    return DataStore(root=root.resolve())
