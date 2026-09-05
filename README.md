# Exam Parser

> **Активная разработка:** основная работа сейчас идёт в ветке [`feature/classification-data-center`](https://github.com/MaksimChislavskiy/exam-parser/tree/feature/classification-data-center). В ней уже реализованы DeepSeek/GigaChat, обработка нескольких вариантов в одном PDF, extraction cache, resume после частичных ошибок, OCR review dataset и дополнительные проверки качества. После стабилизации изменения будут перенесены в `main`. Ниже описана стабильная версия ветки `main`.

Пайплайн извлекает математические задачи из PDF или изображения страницы.

```text
PDF/изображение
      ↓
PaddleOCR-VL → Markdown + готовые изображения
      ↓
Mistral Structured Output → задачи и решения
      ↓
tasks.xlsx + images/*.png
```

JSON-файлы приложение не создаёт и не использует.

## Настройка

В корневом `.env` должен быть ключ:

```env
MISTRAL_API_KEY=ваш_ключ
```

Установка зависимостей:

```powershell
uv sync
```

Для Windows в проекте закреплён режим обычного копирования пакетов: это защищает
крупные бинарники Paddle и Torch от повреждения при создании hardlink.

## Запуск

Запуск по готовому Markdown в `output/markdown`:

```powershell
uv run python main.py
```

Если Markdown существует, повторный тяжёлый запуск PaddleOCR не выполняется.

Другой входной файл:

```powershell
uv run python main.py "C:\path\page.png"
```

При явном входном файле PaddleOCR запускается автоматически.

Если Markdown и картинки PaddleOCR уже готовы:

```powershell
uv run python main.py --reuse-markdown
```

Принудительно пересоздать Markdown для стандартного PDF:

```powershell
uv run python main.py --run-ocr
```

На Windows CPU PaddleOCR-VL может обрабатывать одну страницу десятки минут.
Для повторных запусков используйте готовый Markdown. Для быстрого OCR официальная
документация PaddleOCR рекомендует Linux/WSL или Docker с поддерживаемым GPU.

Только извлечение задач, без решений:

```powershell
uv run python main.py --reuse-markdown --skip-solutions
```

По умолчанию результаты сохраняются в:

```text
output/result/
├── tasks.xlsx
└── images/
    ├── task_1.png
    └── ...
```

Дополнительные параметры:

```text
--output-dir PATH     итоговый Excel и PNG
--markdown-dir PATH   Markdown и imgs PaddleOCR
--pages-dir PATH      изображения страниц PDF
--dpi 300             DPI преобразования PDF
--model MODEL         модель Mistral
--run-ocr             принудительно запустить PaddleOCR
```

## Тесты

```powershell
uv run python -m unittest discover -s tests -v
```
