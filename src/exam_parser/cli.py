from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal

from .documents import prepare_pages
from .llm_client import LLMProvider
from .markdown_boundaries import normalize_task_boundaries
from .markdown_pipeline import process_markdown
from .models import TaskRecord
from .paddle import PaddleDeviceError, recognize_pages
from .pdf_reference import repair_markdown_from_pdf
from .variants import detect_document_variants, variant_page_paths


PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_DIR / "output" / "input"
AnswerSource = Literal["generated", "document", "none"]
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "mistral").strip().lower()
PROVIDER_LABELS = {
    "mistral": "Mistral",
    "gigachat": "GigaChat",
    "deepseek": "DeepSeek",
}

HELP_EPILOG = """
Примеры:
  uv run python main.py trvar540.pdf
      Полный цикл через Mistral.

  uv run python main.py trvar540.pdf --provider deepseek
      Полный цикл через DeepSeek.

  uv run python main.py variant_951.pdf --provider deepseek \
      --no-solutions --document-answers --reuse-markdown
      Готовый Markdown; условия и ответы из документа обрабатывает DeepSeek.

  uv run python main.py trvar540.pdf --no-answers
      Подробные решения без отдельного столбца коротких ответов.

Входной файл всегда ищется в output/input. Указывайте только имя файла.
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Извлечение математических задач, изображений, решений и ответов "
            "из одного документа"
        ),
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        metavar="FILE",
        help=(
            "Имя одного PDF или изображения из output/input. "
            "Папка автоматически не перебирается."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("mistral", "gigachat", "deepseek"),
        default=DEFAULT_PROVIDER,
        help=(
            "LLM-провайдер: mistral, gigachat или deepseek. "
            "По умолчанию берётся LLM_PROVIDER или mistral."
        ),
    )
    parser.add_argument(
        "--no-solutions",
        action="store_true",
        help="Не генерировать подробные решения через выбранную LLM.",
    )
    answer_mode = parser.add_mutually_exclusive_group()
    answer_mode.add_argument(
        "--document-answers",
        action="store_true",
        help=(
            "Брать короткие ответы из раздела ответов самого документа. "
            "Без этого флага ответы генерирует выбранная LLM."
        ),
    )
    answer_mode.add_argument(
        "--no-answers",
        action="store_true",
        help="Не генерировать и не извлекать короткие ответы.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--markdown-dir", type=Path, default=None)
    parser.add_argument("--pages-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--device",
        default="gpu:0",
        help="Устройство Paddle: gpu:0 (по умолчанию), cpu или auto.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help=(
            "Автоматически продолжить на CPU, если GPU недоступен. "
            "Без этого флага в обычном терминале программа запросит подтверждение."
        ),
    )
    ocr_mode = parser.add_mutually_exclusive_group()
    ocr_mode.add_argument(
        "--reuse-markdown",
        action="store_true",
        help="Использовать готовый Markdown без повторного OCR.",
    )
    ocr_mode.add_argument(
        "--run-ocr",
        action="store_true",
        help="Явно запустить PaddleOCR заново; по умолчанию OCR и так запускается.",
    )
    parser.add_argument(
        "--resume-results",
        action="store_true",
        help=(
            "Продолжить по существующему tasks.xlsx: не извлекать задания "
            "повторно и запросить только отсутствующие решения или ответы."
        ),
    )
    parser.add_argument(
        "--expected-tasks",
        type=int,
        default=19,
        help="Ожидаемое число задач; 0 отключает проверку.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Имя модели выбранного LLM-провайдера.",
    )
    return parser


def resolve_input_path(filename: str, input_dir: Path = INPUT_DIR) -> Path:
    """Возвращает путь к одному файлу из стандартной входной папки."""

    candidate = Path(filename)
    if candidate.is_absolute() or candidate.name != filename:
        raise ValueError(
            "Укажите только имя файла из output/input, например: trvar540.pdf"
        )

    input_path = input_dir / candidate.name
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Входной файл не найден: {input_path}. "
            "Поместите его в output/input и укажите только имя файла."
        )
    return input_path.resolve()


def _answer_source(args: argparse.Namespace) -> AnswerSource:
    if args.no_answers:
        return "none"
    if args.document_answers:
        return "document"
    return "generated"


def main() -> None:
    args = build_parser().parse_args()
    if args.provider not in PROVIDER_LABELS:
        raise SystemExit(
            "LLM_PROVIDER должен быть mistral, gigachat или deepseek"
        )

    provider: LLMProvider = args.provider
    provider_label = PROVIDER_LABELS[provider]
    try:
        input_path = resolve_input_path(args.input)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from None

    answer_source = _answer_source(args)
    include_solutions = not args.no_solutions
    workspace = PROJECT_DIR / "output" / "work" / input_path.stem
    output_dir = args.output_dir or PROJECT_DIR / "output" / "result" / input_path.stem
    markdown_dir = args.markdown_dir or workspace / "markdown"
    pages_dir = args.pages_dir or workspace / "pages"

    markdown_is_ready = any(markdown_dir.glob("page_*/page_*.md"))
    if args.reuse_markdown and not markdown_is_ready:
        raise SystemExit(
            f"Нельзя использовать --reuse-markdown: в {markdown_dir} нет готовых страниц."
        )
    run_ocr = not args.reuse_markdown

    answer_description = {
        "generated": f"генерируются {provider_label}",
        "document": f"извлекаются из документа через {provider_label}",
        "none": "не создаются",
    }[answer_source]
    print(f"Входной документ: {input_path}", flush=True)
    print(f"LLM-провайдер: {provider_label}", flush=True)
    print(
        "Подробные решения: "
        + (f"генерируются {provider_label}" if include_solutions else "не создаются"),
        flush=True,
    )
    print(f"Короткие ответы: {answer_description}", flush=True)

    if run_ocr:
        pages = prepare_pages(input_path, pages_dir, dpi=args.dpi)
        try:
            recognize_pages(
                pages,
                markdown_dir,
                device=args.device,
                allow_cpu_fallback=args.allow_cpu_fallback,
            )
        except PaddleDeviceError as error:
            raise SystemExit(str(error)) from None
    else:
        print(f"Используется готовый Markdown: {markdown_dir}", flush=True)

    processing_markdown_dir = repair_markdown_from_pdf(
        input_path,
        markdown_dir,
        workspace / "markdown_verified",
    )
    if processing_markdown_dir != markdown_dir:
        print(
            f"Используется Markdown, сверенный с PDF: {processing_markdown_dir}",
            flush=True,
        )

    variants = detect_document_variants(processing_markdown_dir)
    bounded_markdown_dir = normalize_task_boundaries(
        processing_markdown_dir,
        workspace / "markdown_bounded",
        page_groups=(variant.page_numbers for variant in variants),
    )
    if bounded_markdown_dir != processing_markdown_dir:
        print(
            "Используется Markdown с восстановленными границами задач: "
            f"{bounded_markdown_dir}",
            flush=True,
        )
    processing_markdown_dir = bounded_markdown_dir

    if len(variants) == 1:
        records = _process_variant(
            processing_markdown_dir,
            output_dir,
            page_paths=variant_page_paths(processing_markdown_dir, variants[0]),
            include_solutions=include_solutions,
            answer_source=answer_source,
            provider=provider,
            model=args.model,
            expected_tasks=args.expected_tasks or None,
            resume_results=args.resume_results,
        )
        print(
            f"Готово: {len(records)} задач, файл {output_dir / 'tasks.xlsx'}",
            flush=True,
        )
        return

    print(
        "Найдено вариантов: "
        f"{len(variants)} ({', '.join(item.display_name for item in variants)})",
        flush=True,
    )
    total_records = 0
    for variant in variants:
        variant_output_dir = output_dir / variant.output_name
        pages_text = _format_page_numbers(variant.page_numbers)
        print(
            f"Вариант {variant.display_name}: страницы {pages_text}",
            flush=True,
        )
        records = _process_variant(
            processing_markdown_dir,
            variant_output_dir,
            page_paths=variant_page_paths(processing_markdown_dir, variant),
            include_solutions=include_solutions,
            answer_source=answer_source,
            provider=provider,
            model=args.model,
            expected_tasks=args.expected_tasks or None,
            resume_results=args.resume_results,
        )
        total_records += len(records)
        print(
            f"Готово: вариант {variant.display_name}, {len(records)} задач, "
            f"файл {variant_output_dir / 'tasks.xlsx'}",
            flush=True,
        )

    _remove_legacy_single_result(output_dir)
    print(
        f"Готово: {len(variants)} вариантов, {total_records} задач, "
        f"каталог {output_dir}",
        flush=True,
    )


def _process_variant(
    markdown_dir: Path,
    output_dir: Path,
    *,
    page_paths: list[Path],
    include_solutions: bool,
    answer_source: AnswerSource,
    provider: LLMProvider,
    model: str | None,
    expected_tasks: int | None,
    resume_results: bool,
) -> list[TaskRecord]:
    # Клиенту решения нужен точный каталог текущего результата, чтобы найти
    # уже скопированное изображение конкретной задачи. Переменная действует только
    # внутри текущего процесса и не является пользовательской настройкой.
    os.environ["EXAM_PARSER_CURRENT_OUTPUT_DIR"] = str(output_dir.resolve())
    return process_markdown(
        markdown_dir,
        output_dir,
        page_paths=page_paths,
        include_solutions=include_solutions,
        answer_source=answer_source,
        provider=provider,
        model=model,
        expected_tasks=expected_tasks,
        resume_results=resume_results,
    )


def _format_page_numbers(page_numbers: tuple[int, ...]) -> str:
    if len(page_numbers) == 1:
        return str(page_numbers[0])
    consecutive = page_numbers == tuple(
        range(page_numbers[0], page_numbers[-1] + 1)
    )
    if consecutive:
        return f"{page_numbers[0]}-{page_numbers[-1]}"
    return ", ".join(map(str, page_numbers))


def _remove_legacy_single_result(output_dir: Path) -> None:
    legacy_xlsx = output_dir / "tasks.xlsx"
    if legacy_xlsx.is_file():
        legacy_xlsx.unlink()

    legacy_images = output_dir / "images"
    if not legacy_images.is_dir():
        return
    for path in legacy_images.glob("task_*.png"):
        if path.is_file():
            path.unlink()
    try:
        legacy_images.rmdir()
    except OSError:
        pass
