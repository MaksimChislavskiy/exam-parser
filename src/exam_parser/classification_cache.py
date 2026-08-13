from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .data_store import DataStore, resolve_data_store
from .reference_catalogs import ReferenceCatalog


CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CachedClassification:
    catalog_id: int
    catalog_name: str


class ClassificationCache:
    """Персистентный кэш успешной классификации во внешнем data root."""

    def __init__(
        self,
        *,
        model: str,
        prompt_version: str,
        data_store: DataStore | None = None,
    ) -> None:
        self.model = model
        self.prompt_version = prompt_version
        self.data_store = data_store or resolve_data_store()

    def load(
        self,
        condition: str,
        catalog: ReferenceCatalog,
    ) -> CachedClassification | None:
        path = self._path(condition, catalog)
        if not path.is_file():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None
        if payload.get("format_version") != CACHE_FORMAT_VERSION:
            return None
        if payload.get("prompt_version") != self.prompt_version:
            return None
        if payload.get("model") != self.model:
            return None
        if payload.get("catalog_key") != catalog.spec.key:
            return None
        if payload.get("catalog_fingerprint") != catalog_fingerprint(catalog):
            return None

        catalog_id = payload.get("catalog_id")
        catalog_name = payload.get("catalog_name")
        if not isinstance(catalog_id, int) or not isinstance(catalog_name, str):
            return None

        item = catalog.by_id().get(catalog_id)
        if item is None or item.name != catalog_name:
            return None
        return CachedClassification(
            catalog_id=catalog_id,
            catalog_name=catalog_name,
        )

    def save(
        self,
        condition: str,
        catalog: ReferenceCatalog,
        *,
        catalog_id: int,
    ) -> Path:
        item = catalog.by_id().get(catalog_id)
        if item is None:
            raise ValueError(
                f"Нельзя кэшировать отсутствующий id={catalog_id} "
                f"из справочника {catalog.spec.key!r}"
            )

        path = self._path(condition, catalog)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "catalog_key": catalog.spec.key,
            "catalog_fingerprint": catalog_fingerprint(catalog),
            "condition_hash": _condition_hash(condition),
            "catalog_id": item.item_id,
            "catalog_name": item.name,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def _path(self, condition: str, catalog: ReferenceCatalog) -> Path:
        key = classification_cache_key(
            condition,
            catalog,
            model=self.model,
            prompt_version=self.prompt_version,
        )
        catalog_dir = _safe_component(catalog.spec.key)
        return (
            self.data_store.cache_dir
            / "classification"
            / catalog_dir
            / f"{key}.json"
        )


def classification_cache_key(
    condition: str,
    catalog: ReferenceCatalog,
    *,
    model: str,
    prompt_version: str,
) -> str:
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "prompt_version": prompt_version,
        "model": model,
        "catalog_key": catalog.spec.key,
        "catalog_fingerprint": catalog_fingerprint(catalog),
        "condition": _normalize_condition(condition),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def catalog_fingerprint(catalog: ReferenceCatalog) -> str:
    payload = {
        "spec": {
            "key": catalog.spec.key,
            "filename": catalog.spec.filename,
            "id_column": catalog.spec.id_column,
            "name_column": catalog.spec.name_column,
            "parent_column": catalog.spec.parent_column,
        },
        "headers": catalog.headers,
        "prompt_text": catalog.prompt_text(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _condition_hash(condition: str) -> str:
    return hashlib.sha256(_normalize_condition(condition).encode("utf-8")).hexdigest()


def _normalize_condition(condition: str) -> str:
    return " ".join(condition.split())


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "catalog"
