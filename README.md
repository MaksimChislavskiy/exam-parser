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

### Важная проверка GPU

Программа по умолчанию запускает Paddle с `--device gpu:0` и завершится ошибкой,
если Paddle не видит CUDA. Скрытого перехода на CPU нет. В консоли выводятся:

```text
compiled=True/False, gpu_count=N, actual=gpu:0/cpu
```

В текущем окружении должен быть установлен именно GPU-вариант Paddle,
совместимый с драйвером и поддерживаемой версией CUDA. Пакет `paddlepaddle`
обычно является CPU-сборкой; для GPU требуется соответствующая официальная
сборка `paddlepaddle-gpu`. После её установки проверьте запуск командой ниже.
Осознанный медленный запуск разрешается только с `--device cpu`.

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
--answer-source generated|document
--output-dir PATH
--markdown-dir PATH
--pages-dir PATH
--dpi 300
--device gpu:0|cpu|auto
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
