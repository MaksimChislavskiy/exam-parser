from paddleocr import PaddleOCRVL

# Создаем пайплайн
pipeline = PaddleOCRVL(pipeline_version="v1")

# Ваше изображение
image_path = "output/pages/page_2.png"

# Обработка
output = pipeline.predict(image_path)

# Сохранение результатов
for res in output:
    res.print()  # Покажет в терминале
    res.save_to_json(save_path="output")  # Сохранит JSON
    res.save_to_markdown(save_path="data/markdown")  # Сохранит MARKDOWN! ✨
