# Настройка Ollama для RAGAS Evaluation

## Проблема

Если вы используете Ollama для основного LLM, но не настроили RAGAS метрики, может возникнуть ошибка:
```
The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable
```

Это происходит потому, что RAGAS метрики по умолчанию используют OpenAI.

## Решение

### Вариант 1: Использовать Ollama для RAGAS (рекомендуется)

Добавьте в `.env`:

```bash
# RAGAS Evaluation с Ollama
RAGAS_LLM_PROVIDER=ollama
RAGAS_OLLAMA_MODEL=llama3.1:8b
```

Если не указать `RAGAS_LLM_PROVIDER`, будет использован основной `LLM_PROVIDER`.

### Вариант 2: Использовать OpenAI для RAGAS (если нужна единообразная оценка)

Добавьте в `.env`:

```bash
# RAGAS Evaluation с OpenAI (для единообразной оценки)
RAGAS_LLM_PROVIDER=openai
RAGAS_LLM_MODEL=gpt-4o
OPENAI_API_KEY=your_openai_api_key
```

## Полная конфигурация для Ollama

```bash
# Основной LLM
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_MODEL_QUERY_TRANSFORM=llama3.1:8b

# RAGAS Evaluation (использует тот же провайдер по умолчанию)
RAGAS_LLM_PROVIDER=ollama
RAGAS_OLLAMA_MODEL=llama3.1:8b

# RAGAS Timeouts (опционально, для медленных моделей)
# RAGAS_TIMEOUT=600        # 10 минут (по умолчанию)
# RAGAS_MAX_WAIT=300       # 5 минут между попытками (по умолчанию)
# RAGAS_MAX_RETRIES=5      # Количество попыток (по умолчанию)
# RAGAS_MAX_WORKERS=2      # Меньше воркеров для Ollama (по умолчанию)

# Embeddings (HuggingFace)
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_EMBEDDING_MODEL=intfloat/multilingual-e5-base
HUGGINGFACE_DEVICE=cpu
```

## Настройка таймаутов для Ollama

Ollama работает медленнее, чем OpenAI, поэтому для RAGAS evaluation используются увеличенные таймауты:

- **RAGAS_TIMEOUT** (по умолчанию: 600 секунд = 10 минут) - таймаут для одного запроса
- **RAGAS_MAX_WAIT** (по умолчанию: 300 секунд = 5 минут) - время ожидания между попытками
- **RAGAS_MAX_RETRIES** (по умолчанию: 5) - количество попыток при ошибке
- **RAGAS_MAX_WORKERS** (по умолчанию: 2) - меньше параллельных воркеров для Ollama

Если evaluation все еще падает с TimeoutError, увеличьте эти значения:

```bash
RAGAS_TIMEOUT=900        # 15 минут
RAGAS_MAX_WAIT=600       # 10 минут
RAGAS_MAX_RETRIES=10     # Больше попыток
RAGAS_MAX_WORKERS=1      # Один воркер (самый надежный)
```

## Примечания

- Если `RAGAS_LLM_PROVIDER` не указан, используется `LLM_PROVIDER`
- Если `RAGAS_OLLAMA_MODEL` не указан, используется `OLLAMA_MODEL`
- Если `RAGAS_LLM_MODEL` не указан, выбирается автоматически в зависимости от провайдера
- Для Ollama автоматически применяются увеличенные таймауты

