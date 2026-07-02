from __future__ import annotations

import shutil
from pathlib import Path

from paddleocr import PaddleOCRVL


def recognize_pages(
    pages: list[Path],
    markdown_dir: str | Path,
) -> list[Path]:
    markdown_dir = Path(markdown_dir)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    print("Загрузка PaddleOCR-VL...", flush=True)
    pipeline = PaddleOCRVL(pipeline_version="v1")
    markdown_files: list[Path] = []

    for page_num, page_path in enumerate(pages, start=1):
        print(f"PaddleOCR: страница {page_num}/{len(pages)}", flush=True)
        page_dir = markdown_dir / f"page_{page_num}"
        if page_dir.exists():
            shutil.rmtree(page_dir)
        page_dir.mkdir(parents=True)

        results = pipeline.predict(str(page_path))
        for result in results:
            result.save_to_markdown(save_path=str(page_dir))

        markdown_path = page_dir / f"page_{page_num}.md"
        if not markdown_path.is_file():
            raise RuntimeError(f"PaddleOCR не создал {markdown_path}")
        markdown_files.append(markdown_path)
    return markdown_files
