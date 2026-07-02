from __future__ import annotations

import argparse
from pathlib import Path

from .documents import prepare_pages
from .markdown_pipeline import process_markdown
from .paddle import recognize_pages


PROJECT_DIR = Path(__file__).resolve().parents[2]


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
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "output" / "result",
    )
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        default=PROJECT_DIR / "output" / "markdown",
    )
    parser.add_argument(
        "--pages-dir",
        type=Path,
        default=PROJECT_DIR / "output" / "pages",
    )
    parser.add_argument("--dpi", type=int, default=300)
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
        "--skip-solutions",
        action="store_true",
        help="Не генерировать решения и ответы",
    )
    parser.add_argument("--model", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_was_provided = args.input is not None
    input_path = args.input or PROJECT_DIR / "output" / "input" / "trvar540.pdf"
    markdown_is_ready = any(args.markdown_dir.glob("page_*/page_*.md"))
    run_ocr = args.run_ocr or input_was_provided or not markdown_is_ready
    if args.reuse_markdown:
        run_ocr = False

    if run_ocr:
        pages = prepare_pages(input_path, args.pages_dir, dpi=args.dpi)
        recognize_pages(pages, args.markdown_dir)
    else:
        print(f"Используется готовый Markdown: {args.markdown_dir}", flush=True)

    records = process_markdown(
        args.markdown_dir,
        args.output_dir,
        include_solutions=not args.skip_solutions,
        model=args.model,
    )
    print(
        f"Готово: {len(records)} задач, файл {args.output_dir / 'tasks.xlsx'}",
        flush=True,
    )
