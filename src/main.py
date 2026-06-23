from paddle_parser import parse_page
from pdf_to_images import pdf_to_images
from pathlib import Path
from paddleocr import PaddleOCRVL


INPUT_PDF = "output/input/trvar540.pdf"
PAGES_DIR = "output/pages"
OUTPUT_DIR = Path("output/markdown")


def main():

    pages = pdf_to_images(
        pdf_path=INPUT_PDF,
        output_dir=PAGES_DIR
    )
    print(
        f"Сгенерировано страниц: {len(pages)}"
    )
    pipeline = PaddleOCRVL(pipeline_version="v1")
    for page in pages:
        parse_page(page, pipeline)


if __name__ == "__main__":
    main()
