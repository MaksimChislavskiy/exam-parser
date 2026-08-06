"""Извлечение математических задач из распознанных страниц."""

from .models import ExtractedTask, TaskRecord, TaskSolution
from .condition_repairs import install_condition_repairs

install_condition_repairs()
del install_condition_repairs

__all__ = ["ExtractedTask", "TaskRecord", "TaskSolution"]
