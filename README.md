# Exam Parser

Пайплайн извлекает математические задачи из PDF или изображения страницы.

```text
PDF/изображение
      ↓
PaddleOCR-VL → Markdown + готовые изображения
      ↓
Mistral Structured Output → задачи
      ↓
один из двух режимов ответа
      ├── generated: подробное решение + ответ через Mistral
      └── document: готовый ответ из самого PDF, без решения
      ↓
tasks.xlsx + images/*.png
```

## Один запуск — один конкретный документ

Путь к входному PDF или изображению является обязательным аргументом. Программа
не выбирает документ по умолчанию и не перебирает все файлы из `output/input`.

Корректный запуск всегда содержит путь к одному конкретному исходнику:

```powershell
uv run python main.py "output/input/конкретный_документ.pdf" `
  --answer-source generated
```

Запуск без пути завершится подсказкой командной строки и не начнёт обработку.

## Два исходных документа

В репозитории находится старый исходник:

```text
output/input/trvar540.pdf
```

Новый исходник сохраните как:

```text
output/input/variant_951.pdf
```

Для варианта 540 сохраняется прежнее поведение: каждая задача решается Mistral,
в Excel записываются подробное решение и короткий ответ.

Для варианта 951 Mistral не решает задачи. Он извлекает ответы только из раздела
ответов самого документа, а столбец `solution` остаётся пустым.

## Настройка

В корневом `.env` должен быть ключ:

```env
MISTRAL_API_KEY=ваш_ключ
```

Установка зависимостей:

```powershell
uv sync
```

### Выбор GPU или CPU

Программа по умолчанию запрашивает `--device gpu:0`.

Если Paddle видит CUDA, OCR запускается на GPU. В консоли выводятся реальные
параметры устройства:

```text
compiled=True/False, gpu_count=N, actual=gpu:0/cpu
```

Если GPU недоступен, программа предупреждает, что PaddleOCR-VL может обрабатывать
одну страницу на CPU десятки минут, и предлагает выбор:

```text
GPU недоступен: установленная сборка Paddle не видит CUDA.
PaddleOCR-VL может обрабатывать одну страницу на CPU десятки минут.
Продолжить на CPU? [y/N]:
```

- `y`, `yes`, `д` или `да` — продолжить на CPU;
- Enter, `n` или любой другой ответ — отменить запуск;
- `--device cpu` — сразу запустить CPU без вопроса;
- `--allow-cpu-fallback` — автоматически разрешить переход на CPU без вопроса.

В неинтерактивном запуске программа не зависает в ожидании ввода. Если GPU
недоступен и `--allow-cpu-fallback` не указан, она завершается с понятным
сообщением.

В текущем окружении для GPU должна быть установлена официальная GPU-сборка
Paddle, совместимая с драйвером и поддерживаемой версией CUDA. Пакет
`paddlepaddle` обычно является CPU-сборкой; для GPU требуется подходящая сборка
`paddlepaddle-gpu`.

## Запуск варианта 540: генерация подробных решений

```powershell
uv run python main.py `
  "output/input/trvar540.pdf" `
  --answer-source generated `
  --device gpu:0
```

Использование уже готового Markdown без повторного OCR:

```powershell
uv run python main.py `
  "output/input/trvar540.pdf" `
  --answer-source generated `
  --reuse-markdown
```

## Запуск варианта 951: ответы только из документа

```powershell
uv run python main.py `
  "output/input/variant_951.pdf" `
  --answer-source document `
  --device gpu:0
```

Повторный запуск по готовому Markdown:

```powershell
uv run python main.py `
  "output/input/variant_951.pdf" `
  --answer-source document `
  --reuse-markdown
```

Автоматический переход на CPU, например для скрипта:

```powershell
uv run python main.py `
  "output/input/variant_951.pdf" `
  --answer-source document `
  --device gpu:0 `
  --allow-cpu-fallback
```

Для каждого входного файла создаются отдельные рабочие каталоги, поэтому
Markdown и результаты двух документов не смешиваются:

```text
output/work/<имя_pdf>/pages/
output/work/<имя_pdf>/markdown/
output/result/<имя_pdf>/tasks.xlsx
output/result/<имя_pdf>/images/
```

## Формат Excel

```text
task_num | condition | image_name | solution | answer
```

В режиме `generated` заполняются `solution` и `answer`. В режиме `document`
`solution` пустой, а `answer` переносится из самого PDF.

Геометрические обозначения дополнительно нормализуются после ответа модели.
Например, неразмеченное `A2BB2` преобразуется в `$A_2BB_2$`, а уже существующий
LaTeX повторно не оборачивается.

## Дополнительные параметры

```text
input (обязательный путь к одному PDF или изображению)
--answer-source generated|document
--output-dir PATH
--markdown-dir PATH
--pages-dir PATH
--dpi 300
--device gpu:0|cpu|auto
--allow-cpu-fallback
--expected-tasks 19
--model MODEL
--run-ocr
--reuse-markdown
```

`--expected-tasks 0` отключает проверку количества задач.

## Тесты

```powershell
uv run python -m unittest discover -s tests -v
```
