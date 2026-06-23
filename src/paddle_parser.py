from pathlib import Path

def parse_page(image_path: str, pipeline, page_num: int = 1):
    """Парсинг одной страницы"""
    output = pipeline.predict(image_path)

    for res in output:
        res.print()
        
        # Сохраняем с номером страницы
        save_path = Path("output/markdown") / f"page_{page_num}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        res.save_to_json(save_path=str(save_path))
        res.save_to_markdown(save_path=str(save_path))
        
        print(f"✅ Страница {page_num} сохранена")

    return output


if __name__ == "__main__":
    # Код для тестирования
    from paddleocr import PaddleOCRVL
    
    pipeline = PaddleOCRVL(pipeline_version="v1")
    image_path = "output/pages/page_1.png"
    output = parse_page(image_path, pipeline, page_num=1)
