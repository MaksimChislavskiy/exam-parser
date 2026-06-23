from paddle_parser import parse_page
from pdf_to_images import pdf_to_images
from pathlib import Path
from paddleocr import PaddleOCRVL

INPUT_PDF = "output/input/trvar540.pdf"
PAGES_DIR = "output/pages"
OUTPUT_DIR = Path("output/markdown")

def main():
    print("=" * 50)
    print("📄 Конвертация PDF в изображения...")
    pages = pdf_to_images(
        pdf_path=INPUT_PDF,
        output_dir=PAGES_DIR
    )
    print(f"✅ Сгенерировано страниц: {len(pages)}")
    
    print("\n🚀 Загрузка модели PaddleOCR-VL...")
    pipeline = PaddleOCRVL(pipeline_version="v1")
    print("✅ Модель загружена")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n📝 Начинаю распознавание...")
    for i, page in enumerate(pages, 1):
        print(f"\n--- Страница {i}/{len(pages)} ---")
        parse_page(page, pipeline, page_num=i)
    
    print("\n" + "=" * 50)
    print(f"✅ Готово! Все результаты в папке: {OUTPUT_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    main()
