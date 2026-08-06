"""Извлечение математических задач из распознанных страниц."""

from .models import ExtractedTask, TaskRecord, TaskSolution
from .condition_repairs import install_condition_repairs
from .ocr_context_repairs import install_ocr_context_repairs
from .source_repairs import install_source_repairs
from .boundary_repairs import install_boundary_repairs

install_condition_repairs()
install_ocr_context_repairs()
install_source_repairs()
install_boundary_repairs()

del install_condition_repairs
del install_ocr_context_repairs
del install_source_repairs
del install_boundary_repairs

__all__ = ["ExtractedTask", "TaskRecord", "TaskSolution"]
