"""Извлечение математических задач из распознанных страниц."""

from .models import ExtractedTask, TaskRecord, TaskSolution
from .condition_repairs import install_condition_repairs
from .ocr_context_repairs import install_ocr_context_repairs
from .source_repairs import install_source_repairs
from .boundary_repairs import install_boundary_repairs
from .image_repairs import install_image_repairs
from .release_quality_repairs import install_release_quality_repairs
from .release_quality_repairs_v2 import install_release_quality_repairs_v2
from .pdf_reference_repairs import install_pdf_reference_repairs

install_condition_repairs()
install_ocr_context_repairs()
install_source_repairs()
install_boundary_repairs()
install_image_repairs()
install_release_quality_repairs()
install_release_quality_repairs_v2()
install_pdf_reference_repairs()

del install_condition_repairs
del install_ocr_context_repairs
del install_source_repairs
del install_boundary_repairs
del install_image_repairs
del install_release_quality_repairs
del install_release_quality_repairs_v2
del install_pdf_reference_repairs

__all__ = ["ExtractedTask", "TaskRecord", "TaskSolution"]
