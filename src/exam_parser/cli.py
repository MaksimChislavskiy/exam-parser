from __future__ import annotations

import argparse
from pathlib import Path

from .documents import prepare_pages
from .markdown_pipeline import process_markdown
from .paddle import recognize_pages


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_DIR / "output" / "input" / "trvar540.pdf"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Парсер математических задач")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="PDF или изображение страницы",
    )
    parser.add_argument(
        "--answer-source",
        choices=("generated", "document"),
        default="generated",
        help=(
            "generated — Mistral генерирует подробное решение и ответ; "
            "document — ответ извлекается из самого документа без решения"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--markdown-dir", type=Path, default=None)
    parser.add_argument("--pages-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--device",
        default="gpu:0",
        help="Устройство Paddle: gpu:0 (по умолчанию), cpu или auto",
    )
    parser.add_argument(
        "--reuse-markdown",
        action="store_true",
        help="Принудительно использовать готовый Markdown",
    )
    parser.add_argument(
        "--run-ocr",
        action="store_true",
        help="Принудительно запустить PaddleOCR заново",
    )
    parser.add_argument(
        "--expected-tasks",
        type=int,
        default=19,
        help="Ожидаемое число задач; 0 отключает проверку",
    )
    parser.add_argument("--model", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_was_provided = args.input is not None
    input_path = (args.input or DEFAULT_INPUT).resolve()
    workspace = PROJECT_DIR / "output" / "work" / input_path.stem
    output_dir = args.output_dir or PROJECT_DIR / "output" / "result" / input_path.stem
    markdown_dir = args.markdown_dir or workspace / "markdown"
    pages_dir = args.pages_dir or workspace / "pages"

    markdown_is_ready = any(markdown_dir.glob("page_*/page_*.md"))
    run_ocr = args.run_ocr or input_was_provided or not markdown_is_ready
    if args.reuse_markdown:
        run_ocr = False

    print(f"Входной документ: {input_path}", flush=True)
    print(f"Источник ответов: {args.answer_source}", flush=True)
    if run_ocr:
        pages = prepare_pages(input_path, pages_dir, dpi=args.dpi)
        recognize_pages(pages, markdown_dir, device=args.device)
    else:
        print(f"Используется готовый Markdown: {markdown_dir}", flush=True)

    records = process_markdown(
        markdown_dir,
        output_dir,
        answer_source=args.answer_source,
        model=args.model,
        expected_tasks=args.expected_tasks or None,
    )
    print(
        f"Готово: {len(records)} задач, файл {output_dir / 'tasks.xlsx'}",
        flush=True,
    )
