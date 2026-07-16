# Exam Parser

Пайплайн извлекает математические задачи из одного PDF или изображения страницы.

```text
PDF/изображение
      ↓
PaddleOCR-VL → Markdown + изображения
      ↓
Mistral Structured Output → условия задач
      ↓
независимо настраиваемые результаты
      ├── подробное решение: Mistral или пусто
      └── короткий ответ: Mistral, документ или пусто
      ↓
tasks.xlsx + images/*.png
```

## Входные документы

Все исходники размещаются в стандартной папке:

```text
output/input/
```

В команде указывается только имя одного конкретного файла:

```powershell
uv run python main.py trvar540.pdf
```

Программа обработает только `output/input/trvar540.pdf`. Она не перебирает остальные
файлы в папке. Полный путь передавать не нужно и нельзя.

Примеры документов:

```text
output/input/trvar540.pdf
output/input/variant_951.pdf
```

## Поведение по умолчанию

Запуск без дополнительных флагов выполняет полный цикл:

```powershell
uv run python main.py trvar540.pdf
```

Для каждой задачи Mistral одним запросом создаёт подробное решение и короткий
финальный ответ.

## Независимые флаги результата

Подробное решение и короткий ответ настраиваются отдельно.

| Команда | Подробное решение | Короткий ответ |
|---|---|---|
| `main.py FILE` | Mistral | Mistral |
| `main.py FILE --no-solutions` | пусто | Mistral |
| `main.py FILE --document-answers` | Mistral | из документа |
| `main.py FILE --no-solutions --document-answers` | пусто | из документа |
| `main.py FILE --no-answers` | Mistral | пусто |
| `main.py FILE --no-solutions --no-answers` | пусто | пусто |

Программа не создаёт ненужный результат:

- полный цикл использует один запрос на решение и ответ;
- `--no-solutions` вызывает только короткий запрос за ответом;
- `--document-answers` извлекает короткие ответы из раздела ответов PDF;
- `--no-answers` не создаёт короткие ответы;
- `--no-solutions --no-answers` оставляет только задачи и изображения.

Флаги `--document-answers` и `--no-answers` взаимоисключающие.

## Примеры для двух документов

Вариант 540 с подробными решениями и ответами Mistral:

```powershell
uv run python main.py trvar540.pdf
```

Только короткие ответы:

```powershell
uv run python main.py trvar540.pdf --no-solutions
```

Вариант 951 с ответами из документа и без решений:

```powershell
uv run python main.py variant_951.pdf --no-solutions --document-answers
```

Для того же документа можно позднее добавить подробные решения:

```powershell
uv run python main.py variant_951.pdf --document-answers
```

## Настройка

В корневом `.env` должен быть ключ:

```env
MISTRAL_API_KEY=ваш_ключ
```

Необязательно можно задать модель:

```env
MISTRAL_MODEL=mistral-large-2512
```

Установка зависимостей:

```powershell
uv sync
```

## Выбор GPU или CPU

По умолчанию Paddle запускается с `--device gpu:0`.

Если Paddle видит CUDA, OCR работает на GPU. В консоли выводятся реальные параметры:

```text
compiled=True/False, gpu_count=N, actual=gpu:0/cpu
```

Если GPU недоступен, программа предупреждает о медленной работе CPU и спрашивает:

```text
GPU недоступен: установленная сборка Paddle не видит CUDA.
PaddleOCR-VL может обрабатывать одну страницу на CPU десятки минут.
Продолжить на CPU? [y/N]:
```

- `y`, `yes`, `д` или `да` — продолжить на CPU;
- Enter, `n` или любой другой ответ — отменить запуск;
- `--device cpu` — сразу запустить CPU без вопроса;
- `--allow-cpu-fallback` — автоматически разрешить переход на CPU.

В неинтерактивном запуске программа не зависает в ожидании ввода. Без явного
разрешения перехода на CPU она завершится с понятным сообщением.

Для GPU нужна совместимая официальная сборка `paddlepaddle-gpu`. Обычный пакет
`paddlepaddle` обычно является CPU-сборкой.

## Повторный запуск без OCR

Для каждого входного файла создаются отдельные рабочие каталоги:

```text
output/work/<имя_файла>/pages/
output/work/<имя_файла>/markdown/
output/result/<имя_файла>/tasks.xlsx
output/result/<имя_файла>/images/
```

Готовый Markdown можно использовать повторно:

```powershell
uv run python main.py variant_951.pdf `
  --no-solutions `
  --document-answers `
  --reuse-markdown
```

Если для выбранного документа готового Markdown нет, программа завершится с
сообщением и не создаст пустой результат.

## Формат Excel

```text
task_num | condition | image_name | solution | answer
```

Столбцы `solution` и `answer` остаются в Excel всегда. При отключении результата
соответствующий столбец остаётся пустым.

Геометрические обозначения дополнительно нормализуются после ответа модели.
Например, неразмеченное `A2BB2` преобразуется в `$A_2BB_2$`, а существующий LaTeX
повторно не оборачивается.

## Справка командной строки

```powershell
uv run python main.py --help
```

Справка содержит описание флагов и примеры основных сочетаний.

Основные параметры:

```text
FILE                     имя файла из output/input
--no-solutions           не создавать подробные решения
--document-answers       брать короткие ответы из документа
--no-answers             не создавать короткие ответы
--reuse-markdown         не запускать OCR повторно
--run-ocr                явно запустить OCR заново
--device gpu:0|cpu|auto
--allow-cpu-fallback
--expected-tasks 19
--model MODEL
--output-dir PATH
--markdown-dir PATH
--pages-dir PATH
--dpi 300
```

`--expected-tasks 0` отключает проверку количества задач.

## Тесты

```powershell
uv run python -m unittest discover -s tests -v
```
