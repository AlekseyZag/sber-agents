# Настройка Ollama для LLM

## Быстрая настройка

### 1. Установите Ollama

Скачайте и установите с [ollama.com](https://ollama.com)

### 2. Установите модель (рекомендуется для 16GB RAM)

```bash
# Оптимальный вариант - баланс качества и скорости
ollama pull llama3.1:8b

# Или для лучшей поддержки русского языка
ollama pull qwen2.5:7b
```

**Примечание:** Ollama автоматически выбирает оптимальный формат квантования. Не нужно указывать `-q4_K_M` в имени модели.

### 3. Настройте .env

```bash
# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_MODEL_QUERY_TRANSFORM=llama3.1:8b

# Embeddings (оставляем HuggingFace)
EMBEDDING_PROVIDER=huggingface
HUGGINGFACE_EMBEDDING_MODEL=intfloat/multilingual-e5-base
HUGGINGFACE_DEVICE=cpu
```

### 4. Проверьте работу Ollama

```bash
# Проверить, что Ollama запущен
ollama list

# Протестировать модель
ollama run llama3.1:8b-q4_K_M "Привет, как дела?"
```

### 5. Запустите бота

```bash
uv run python src/bot.py
```

В логах должна быть строка:
```
Main LLM initialized: ollama/llama3.1:8b
```

## Рекомендации по моделям

См. подробный файл: [ollama-models-16gb.md](./ollama-models-16gb.md)

**Кратко:**
- **llama3.1:8b** - лучший баланс (~4.9 GB)
- **qwen2.5:7b** - лучше для русского (~4.7 GB)
- **llama3.2:3b** - максимальная скорость (~2 GB)

## Устранение проблем

### Ollama не запускается

```bash
# Проверить статус
ollama serve

# Или запустить вручную
ollama serve
```

### Модель не найдена

```bash
# Проверить установленные модели
ollama list

# Если модели нет - установить
ollama pull llama3.1:8b
```

### Ошибка подключения

Убедитесь, что:
1. Ollama запущен (`ollama serve`)
2. `OLLAMA_BASE_URL` правильный (по умолчанию `http://localhost:11434`)
3. Нет файрвола, блокирующего подключение

