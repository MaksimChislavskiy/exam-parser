from paddleocr import PaddleOCRVL


def parse_page(image_path: str, pipeline):
    """Парсинг одной страницы"""
    output = pipeline.predict(image_path)

    for res in output:
        res.print()
        res.save_to_json(save_path="output")
        res.save_to_markdown(save_path="output/markdown")

    return


if __name__ == "__main__":
    # Код для тестирования
    pipeline = PaddleOCRVL(pipeline_version="v1")
    image_path = "output/pages/page_2.png"
    output = parse_page(image_path, pipeline)

    for res in output:
        res.print()
        res.save_to_json(save_path="output")
        res.save_to_markdown(save_path="output/markdown")
