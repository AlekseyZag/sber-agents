## RAG-ассистент Сбербанка

Коротко: Telegram‑бот с RAG (Retrieval‑Augmented Generation), отвечающий на вопросы пользователей на основе банковских документов (PDF + JSON). Проект ориентирован на простоту запуска и прозрачность конфигурации.

См. детали в `live/README.md` и архитектурные заметки в `live/docs/vision.md`.


### Вариант задания
- Базовый


### Реализованные возможности
- [x] Индексация PDF документов (разбиение на чанки, эмбеддинги, векторное хранилище)
- [x] Поддержка дополнительного JSON‑датасета вопросов‑ответов (JSONLoader с jq_schema)
- [x] RAG‑цепочка с трансформацией запроса (query transformation) и контекстным диалогом
- [x] Telegram‑бот на aiogram 3.x с асинхронной обработкой сообщений
- [x] Конфигурация через `.env` (модели, ключи, пути, промпты)
- [x] Логирование в файл и консоль
- [x] Команды `/start`, `/help`, `/index`, `/index_status`


### Технологический стек
- Python 3.11+, aiogram 3.x
- LangChain (+ langchain‑openai, langchain‑community, langchain‑text‑splitters)
- PyPDF (загрузка PDF)
- In‑Memory VectorStore (память процесса)
- OpenAI‑совместимый API (OpenRouter / Fireworks)


### Используемые модели
- Генерация ответов (LLM): 
  - Fireworks: `accounts/fireworks/models/gpt-oss-120b` (первоначально), рекомендованные альтернативы: `qwen-2.5-72b-instruct`, `llama-v3p1-70b-instruct`, `mixtral-8x22b-instruct`
  - OpenRouter: `openai/gpt-4o`, `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`
- Query transformation: настраивается отдельно через `MODEL_QUERY_TRANSFORM` (по умолчанию «лёгкая» та же или `gpt-4o-mini`/эквивалент)
- Эмбеддинги:
  - Fireworks: `accounts/fireworks/models/qwen3-embedding-8b`
  - OpenAI/OpenRouter: `openai/text-embedding-3-large` (aka `text-embedding-3-large`)


## ВАЖНО — Эксперименты с индексацией

Исходные настройки в проекте:
- `live/src/indexer.py`: базовый сценарий с `chunk_size=500` (описан в README)
- `live/src/indexer_with_json.py`: расширенный сценарий для PDF + JSON, использует `RecursiveCharacterTextSplitter` с:
  - `chunk_size=800`
  - `chunk_overlap=100`
  - `separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]`

Что пробовали:
- Параметры A: chunk_size=500, overlap=50 (классический базовый)
- Параметры B: chunk_size=800, overlap=100 (актуально в `indexer_with_json.py`)
- Параметры C: chunk_size=1200, overlap=150 (для длинных разделов, меньшее дробление)

Наблюдения (банковские документы, регламенты, тарифы):
- При 500/50 ответы часто короче и точнее на простые вопросы, но встречались случаи «недобора» контекста для длинных формулировок (приходилось поднимать k).
- При 800/100 повысилась полнота извлекаемого контекста по сложным вопросам (тарифные таблицы, условия исключений), при этом рост «шума» не критичен, если `k=3`.
- При 1200/150 иногда попадали слишком «тяжёлые» чанки, ухудшалась релевантность первых результатов и возрастало время ответа.

Выводы (лучше для банковских документов):
- Оптимальный компромисс: `chunk_size≈800`, `overlap≈100`, k=3.
- Важно сохранять разделители (keep_separator=True) и использовать иерархию сепараторов — это помогает сохранять «форму» документов (разделы/подразделы).


## ВАЖНО — Работа с JSON датасетом

Как реализовано:
- Используется `JSONLoader` из `langchain_community.document_loaders` с `jq_schema='.[].full_text'` для извлечения текстовых блоков из массива JSON.
- См. `live/src/indexer_with_json.py` (функция `load_json_documents`).
- Загрузка объединяется с PDF‑чанками перед построением векторного хранилища.

Скриншоты (вопросы про карты):
- [screenshots/json_cards_qna_1.png](screenshots/json_cards_qna_1.png)
- [screenshots/json_cards_qna_2.png](screenshots/json_cards_qna_2.png)


## ВАЖНО — Сравнение моделей эмбеддингов

Тестировались варианты:
- Fireworks: `qwen3-embedding-8b`
- OpenAI/OpenRouter: `text-embedding-3-large`

Методика:
- Одинаковый набор вопросов (про кредиты/вклады/карты), одинаковые чанки и `k=3`.
- Оценка по 3 критериям: точность (фактологическая), полнота (сколько релевантного подтягивается), стабильность формулировок.

Итоги (обобщённо, по нашему датасету на русском):

| Модель эмбеддингов | Точность | Полнота | Стабильность | Примечания |
| --- | --- | --- | --- | --- |
| qwen3-embedding-8b | высокая | высокая | высокая | Чуть лучше извлекает русскоязычные фрагменты, особенно длинные |
| text-embedding-3-large | высокая | высокая | высокая | Отличная база, очень стабильные результаты |

Вывод:
- Оба варианта показывают высокий уровень. Незначимое, но повторяемое преимущество для русского текста наблюдалось у `qwen3-embedding-8b` (Fireworks), особенно на длинных разделах с юридическими формулировками.
- Если уже используете OpenAI‑совместимый стек и важна предсказуемость и документированная поддержка, `text-embedding-3-large` — отличный выбор.
- Если фокус на русском/многоязычном и используется Fireworks, `qwen3-embedding-8b` — предпочтителен.


## Скриншоты

- Индексация и статус: [screenshots/index_status.png](screenshots/index_status.png)
- Диалог с уточнениями: [screenshots/dialog_followups.png](screenshots/dialog_followups.png)
- Ошибки/устранение неполадок (пример): [screenshots/troubleshooting.png](screenshots/troubleshooting.png)


## Примечания по запуску
- Конфигурация: `.env` (см. примеры в `live/README.md` — OpenRouter и Fireworks)
- Запуск: из `live/` — `make run` или `uv run python src/bot.py`
- При смене провайдера проверьте корректность `OPENAI_BASE_URL`, `MODEL`, `EMBEDDING_MODEL`


