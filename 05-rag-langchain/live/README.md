# RAG-ассистент Сбербанка

Telegram-бот с RAG (Retrieval-Augmented Generation) для ответов на вопросы по документам Сбербанка о кредитах и вкладах.

## ✨ Возможности

- 🤖 **RAG на базе LangChain** - ответы на основе реальных документов
- 📚 **Индексация PDF** - автоматическая обработка документов при старте
- 💬 **Контекстный диалог** - понимание уточняющих вопросов
- 🔍 **Query Transformation** - улучшение поисковых запросов с учетом истории
- ⚡ **Асинхронная обработка** - поддержка множества пользователей одновременно
- 📝 **Логирование** - запись всех событий в файл для отладки

## 🚀 Быстрый старт

### Требования

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - менеджер зависимостей

### Установка

1. Клонируйте репозиторий:
   ```bash
   git clone <repository-url>
   cd telegram-llm-bot
   ```

2. Установите зависимости:
   ```bash
   make install
   ```

3. Настройте переменные окружения:
   ```bash
   cp env.example .env
   ```

4. Отредактируйте `.env` (см. раздел "Конфигурация")

5. Запустите бота:
   ```bash
   make run
   ```

## ⚙️ Конфигурация

### Получение токенов

**Telegram Bot Token:**
1. Найдите @BotFather в Telegram
2. Отправьте `/newbot` и следуйте инструкциям
3. Скопируйте токен

**API ключи провайдеров:**

Бот поддерживает работу с разными провайдерами LLM через OpenAI-совместимый API.

#### OpenRouter

1. Зарегистрируйтесь на [OpenRouter.ai](https://openrouter.ai/)
2. Перейдите в раздел API Keys
3. Создайте новый ключ

#### Fireworks

1. Зарегистрируйтесь на [Fireworks.ai](https://fireworks.ai/)
2. Перейдите в раздел API Keys
3. Создайте новый ключ

### Примеры конфигурации

**Вариант 1: OpenRouter**

```bash
# Telegram
TELEGRAM_TOKEN=your_telegram_bot_token

# OpenRouter
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL=openai/gpt-oss-20b:free
MODEL_QUERY_TRANSFORM=openai/gpt-oss-20b:free
EMBEDDING_MODEL=openai/text-embedding-3-large

# Пути
DATA_DIR=data
PROMPTS_DIR=prompts
CONVERSATION_SYSTEM_PROMPT_FILE=conversation_system.txt
QUERY_TRANSFORM_PROMPT_FILE=query_transform.txt

# Системный промпт
SYSTEM_PROMPT=Ты ассистент Сбербанка, отвечающий на вопросы по документам.
```

**Вариант 2: Fireworks**

```bash
# Telegram
TELEGRAM_TOKEN=your_telegram_bot_token

# Fireworks
OPENAI_API_KEY=fw_...
OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1

# ВАЖНО: Если получаете ошибку 404, попробуйте формат без префикса
# Проверьте доступные модели на https://fireworks.ai/models
MODEL=qwen-2.5-72b-instruct
MODEL_QUERY_TRANSFORM=qwen-2.5-72b-instruct
EMBEDDING_MODEL=qwen3-embedding-8b

# Альтернативные варианты (если выше не работает):
# MODEL=accounts/fireworks/models/qwen-2.5-72b-instruct
# MODEL_QUERY_TRANSFORM=accounts/fireworks/models/qwen-2.5-72b-instruct
# EMBEDDING_MODEL=accounts/fireworks/models/qwen3-embedding-8b

# Пути
DATA_DIR=data
PROMPTS_DIR=prompts
CONVERSATION_SYSTEM_PROMPT_FILE=conversation_system.txt
QUERY_TRANSFORM_PROMPT_FILE=query_transform.txt

# Системный промпт
SYSTEM_PROMPT=Ты ассистент Сбербанка, отвечающий на вопросы по документам.
```

**Вариант 3: Fireworks (более мощные модели)**

Если `gpt-oss-120b` не дает нужного качества, попробуйте более мощные модели:

```bash
# Telegram
TELEGRAM_TOKEN=your_telegram_bot_token

# Fireworks - более мощные модели
OPENAI_API_KEY=fw_...
OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1

# Рекомендуемые более мощные модели для Fireworks:
# Попробуйте эти варианты (формат может отличаться в зависимости от версии API):

# Вариант 1: Без префикса (часто используется)
MODEL=qwen-2.5-72b-instruct
MODEL_QUERY_TRANSFORM=qwen-2.5-72b-instruct

# Вариант 2: С префиксом accounts/fireworks/models/
# MODEL=accounts/fireworks/models/qwen-2.5-72b-instruct
# MODEL_QUERY_TRANSFORM=accounts/fireworks/models/qwen-2.5-72b-instruct

# Вариант 3: Другие мощные модели (попробуйте без префикса сначала)
# MODEL=llama-v3p1-70b-instruct
# MODEL_QUERY_TRANSFORM=llama-v3p1-70b-instruct

# MODEL=mixtral-8x22b-instruct
# MODEL_QUERY_TRANSFORM=mixtral-8x22b-instruct

# MODEL=firefunction-v2
# MODEL_QUERY_TRANSFORM=firefunction-v2

# Для эмбеддингов попробуйте:
EMBEDDING_MODEL=qwen3-embedding-8b
# или
# EMBEDDING_MODEL=accounts/fireworks/models/qwen3-embedding-8b

# Пути
DATA_DIR=data
PROMPTS_DIR=prompts
CONVERSATION_SYSTEM_PROMPT_FILE=conversation_system.txt
QUERY_TRANSFORM_PROMPT_FILE=query_transform.txt

# Системный промпт
SYSTEM_PROMPT=Ты ассистент Сбербанка, отвечающий на вопросы по документам.
```

**Вариант 4: OpenRouter (премиум модели)**

Для максимального качества используйте OpenRouter с премиум моделями:

```bash
# Telegram
TELEGRAM_TOKEN=your_telegram_bot_token

# OpenRouter - премиум модели
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1

# Рекомендуемые премиум модели через OpenRouter:
# - GPT-4o (лучшее качество от OpenAI)
MODEL=openai/gpt-4o
MODEL_QUERY_TRANSFORM=openai/gpt-4o

# - Claude 3.5 Sonnet (отличное качество, хорошая работа с русским)
# MODEL=anthropic/claude-3.5-sonnet
# MODEL_QUERY_TRANSFORM=anthropic/claude-3.5-sonnet

# - Claude 3 Opus (максимальное качество, но дороже)
# MODEL=anthropic/claude-3-opus
# MODEL_QUERY_TRANSFORM=anthropic/claude-3-opus

# - Qwen 2.5 72B (хороший баланс цена/качество)
# MODEL=qwen/qwen-2.5-72b-instruct
# MODEL_QUERY_TRANSFORM=qwen/qwen-2.5-72b-instruct

# - DeepSeek-V3 (мощная модель с хорошей поддержкой русского)
# MODEL=deepseek/deepseek-chat
# MODEL_QUERY_TRANSFORM=deepseek/deepseek-chat

EMBEDDING_MODEL=openai/text-embedding-3-large

# Пути
DATA_DIR=data
PROMPTS_DIR=prompts
CONVERSATION_SYSTEM_PROMPT_FILE=conversation_system.txt
QUERY_TRANSFORM_PROMPT_FILE=query_transform.txt

# Системный промпт
SYSTEM_PROMPT=Ты ассистент Сбербанка, отвечающий на вопросы по документам.
```

### Рекомендации по выбору модели

**Если используете Fireworks и хотите более мощную модель:**

**ВАЖНО:** Формат названий моделей может отличаться. Если получаете ошибку 404, попробуйте варианты без префикса `accounts/fireworks/models/`.

1. **Qwen 2.5 72B** - **РЕКОМЕНДУЕТСЯ**
   - Попробуйте: `qwen-2.5-72b-instruct` или `accounts/fireworks/models/qwen-2.5-72b-instruct`
   - Отличная поддержка русского языка
   - Хорошее качество ответов
   - Баланс цена/качество

2. **Firefunction-v2** (новая модель от Fireworks)
   - Попробуйте: `firefunction-v2` или `accounts/fireworks/models/firefunction-v2`
   - Быстрее GPT-4o в 2.5 раза
   - Дешевле GPT-4o на 90%
   - Отличное качество

3. **Llama 3.1 70B**
   - Попробуйте: `llama-v3p1-70b-instruct` или `accounts/fireworks/models/llama-v3p1-70b-instruct`
   - Хорошее качество
   - Быстрая генерация

4. **Mixtral 8x22B**
   - Попробуйте: `mixtral-8x22b-instruct` или `accounts/fireworks/models/mixtral-8x22b-instruct`
   - Очень быстрая
   - Хорошее качество для большинства задач

**Как проверить доступные модели:**
- Зайдите на https://fireworks.ai/models
- Или проверьте документацию API вашего аккаунта

**Если готовы перейти на OpenRouter для максимального качества:**

1. **GPT-4o** (`openai/gpt-4o`) - **ЛУЧШИЙ ВЫБОР**
   - Максимальное качество ответов
   - Отличная работа с русским языком
   - Быстрая генерация

2. **Claude 3.5 Sonnet** (`anthropic/claude-3.5-sonnet`)
   - Очень высокое качество
   - Отличное понимание контекста

3. **Claude 3 Opus** (`anthropic/claude-3-opus`)
   - Максимальное качество (но дороже)

### Описание параметров

**Обязательные:**
- `TELEGRAM_TOKEN` - токен бота от @BotFather
- `OPENAI_API_KEY` - API ключ от выбранного провайдера
- `OPENAI_BASE_URL` - URL API провайдера

**Модели:**
- `MODEL` - модель для генерации ответов (основная LLM)
- `MODEL_QUERY_TRANSFORM` - модель для трансформации запросов
- `EMBEDDING_MODEL` - модель для создания эмбеддингов документов

**Пути:**
- `DATA_DIR` - директория с PDF документами (по умолчанию: `data`)
- `PROMPTS_DIR` - директория с файлами промптов (по умолчанию: `prompts`)
- `CONVERSATION_SYSTEM_PROMPT_FILE` - файл промпта для диалога
- `QUERY_TRANSFORM_PROMPT_FILE` - файл промпта для трансформации запросов

**Промпты:**
- `SYSTEM_PROMPT` - системная инструкция для бота

## 📚 Добавление документов

1. Поместите PDF файлы в директорию `data/`
2. Перезапустите бота (документы проиндексируются автоматически)
   
   ИЛИ
   
3. Используйте команду `/index` в Telegram для переиндексации

**Примечание:** Бот автоматически:
- Загружает все PDF из `data/`
- Разбивает на чанки по 500 символов
- Создает векторные эмбеддинги
- Сохраняет в памяти для быстрого поиска

## 💬 Использование

### Команды бота

- `/start` - Начать новый диалог (сбросить историю)
- `/help` - Показать справку
- `/index` - Переиндексировать документы
- `/index_status` - Проверить статус индексации

### Примеры диалогов

**Простой вопрос:**
```
👤 Какие условия потребительского кредита?
🤖 По документу, потребительский кредит предоставляется на сумму от 30 000 до 5 000 000 рублей, 
   на срок от 3 месяцев до 5 лет. Процентная ставка зависит от категории заемщика и составляет 
   от 12.9% до 19.9% годовых.
```

**Уточняющий вопрос:**
```
👤 Какие вклады есть в Сбербанке?
🤖 В документах указаны следующие виды вкладов: "Пополняй", "Управляй", "Сохраняй"...

👤 А какие проценты по вкладу "Сохраняй"?
🤖 По вкладу "Сохраняй" процентная ставка составляет от 4% до 6% годовых в зависимости 
   от суммы и срока вклада...
```

**Вопрос вне контекста:**
```
👤 Какая погода сегодня?
🤖 Я не нашел ответа на ваш вопрос в доступных документах.
```

## 🏗️ Архитектура

### Структура проекта

```
├── src/
│   ├── bot.py          # Точка входа, инициализация, логирование
│   ├── config.py       # Загрузка конфигурации из .env
│   ├── handlers.py     # Обработчики команд и сообщений
│   ├── indexer.py      # Загрузка и индексация PDF
│   └── rag.py          # RAG-логика: retriever, цепочки, промпты
├── prompts/
│   ├── conversation_system.txt    # Промпт для диалога
│   └── query_transform.txt        # Промпт для трансформации запросов
├── data/               # PDF документы для индексации
├── logs/               # Логи работы бота
├── .env                # Конфигурация (не в git)
├── env.example         # Пример конфигурации
├── Makefile            # Команды для работы
├── pyproject.toml      # Зависимости
└── README.md           # Документация
```

### Как работает RAG

1. **Индексация** (при старте):
   ```
   PDF документы → Разбиение на чанки → Создание эмбеддингов → Векторное хранилище (в памяти)
   ```

2. **Обработка вопроса**:
   ```
   Вопрос пользователя → Query Transformation (с учетом истории) →
   → Поиск релевантных чанков (k=3) → Генерация ответа с контекстом
   ```

3. **Контекстный диалог**:
   - История сохраняется в формате LangChain Messages
   - Уточняющие вопросы понимаются через query transformation
   - LLM получает и историю, и найденный контекст из документов

### Технологический стек

- **aiogram 3.x** - Telegram Bot API
- **LangChain** - фреймворк для RAG
- **LangChain OpenAI** - интеграция с OpenAI-совместимыми API
- **PyPDF** - парсинг PDF документов
- **InMemoryVectorStore** - векторное хранилище в памяти

## 🔧 Разработка

### Команды Makefile

```bash
make install    # Установить зависимости
make run        # Запустить бота
```

### Редактирование промптов

Промпты находятся в `prompts/` и могут редактироваться без изменения кода:

**`prompts/conversation_system.txt`** - как бот отвечает на вопросы:
```
Ты ассистент Сбербанка для ответов на вопросы. Отвечай на вопросы пользователей 
на основе истории диалога и контекста, полученного для последнего вопроса. 

Если в контексте нет информации для ответа, строго отвечай: 
"Я не нашел ответа на ваш вопрос в доступных документах."

Используй максимум 3-4 предложения и давай конкретные ответы.
```

**`prompts/query_transform.txt`** - как трансформируются уточняющие вопросы:
```
Преобразуй последнее сообщение пользователя в поисковый запрос на русском языке, 
учитывая всю историю диалога выше. Тщательно проанализируй все сообщения для 
создания максимально релевантного запроса.
```

### Логи

Логи записываются в `logs/bot.log` и дублируются в консоль.

**Логируются:**
- Старт/остановка бота
- Процесс индексации документов
- Входящие сообщения от пользователей
- Ошибки и исключения

**Пример лога:**
```
2025-11-07 18:32:37,399 - __main__ - INFO - Starting indexing...
2025-11-07 18:32:38,384 - indexer - INFO - Split into 377 chunks
2025-11-07 18:32:41,314 - indexer - INFO - Created vector store with 377 chunks
2025-11-07 18:32:41,314 - __main__ - INFO - Indexing completed successfully
```

### Настройка параметров RAG

В `src/indexer.py` и `src/bot.py` можно настроить:

- **Размер чанков**: `chunk_size=500` (в RecursiveCharacterTextSplitter)
- **Перекрытие чанков**: `chunk_overlap=50`
- **Количество чанков для поиска**: `k=3` (в retriever)
- **Temperature** для LLM: `temperature=0.9` (в rag.py)

## ⚠️ Ограничения

- История хранится в памяти (теряется при перезапуске)
- Векторное хранилище в памяти (требует переиндексации после перезапуска)
- Только текстовые сообщения (нет поддержки фото, файлов, голосовых)
- Ответы основаны только на проиндексированных документах
- При большом количестве документов может требоваться больше памяти

## 🐛 Устранение неполадок

**Проблема: Бот не отвечает на вопросы**
- Проверьте `/index_status` - должны быть проиндексированы документы
- Убедитесь, что PDF файлы находятся в `data/`
- Проверьте логи в `logs/bot.log`

**Проблема: Ошибка при индексации**
- Проверьте корректность `EMBEDDING_MODEL` для вашего провайдера
- Убедитесь, что API ключ валиден и имеет доступ к embeddings

**Проблема: Бот отвечает "Я не нашел ответа" на все вопросы**
- Возможно, вопросы не связаны с содержимым документов
- Попробуйте задать более конкретные вопросы по тематике документов
- Проверьте, что индексация прошла успешно (`/index_status`)

**Проблема: Ошибка 404 "Model not found"**
- Модель с указанным именем не найдена или недоступна
- **Решение для Fireworks:**
  1. Попробуйте формат без префикса: `qwen-2.5-72b-instruct` вместо `accounts/fireworks/models/qwen-2.5-72b-instruct`
  2. Проверьте доступные модели на https://fireworks.ai/models
  3. Убедитесь, что модель развернута в вашем аккаунте Fireworks
  4. Попробуйте альтернативные модели: `firefunction-v2`, `llama-v3p1-70b-instruct`
- **Решение для OpenRouter:**
  1. Проверьте правильность названия модели на https://openrouter.ai/models
  2. Убедитесь, что модель доступна для вашего аккаунта
  3. Попробуйте альтернативные модели: `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`

## 📝 Лицензия

MIT
