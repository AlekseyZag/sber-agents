# Исправление ошибок таймаутов Telegram

## Проблема

```
ERROR - Failed to fetch updates - TelegramNetworkError: HTTP Client says - Request timeout error
```

Эта ошибка возникает, когда Telegram бот не может получить обновления от Telegram API из-за таймаутов.

## Причины

1. **Медленное интернет-соединение** - запросы к Telegram API занимают слишком много времени
2. **Проблемы с Telegram API** - временные сбои на стороне Telegram
3. **Долгие операции** - evaluation или индексация могут замедлять обработку
4. **Недостаточные таймауты** - стандартные таймауты aiogram слишком короткие

## Решение

### 1. Увеличенные таймауты (уже применено)

В `bot.py` добавлены увеличенные таймауты:

```python
session = AiohttpSession(
    timeout=ClientTimeout(
        total=60,      # Общий таймаут 60 секунд
        connect=30,   # Таймаут подключения 30 секунд
    )
)
```

### 2. Настройки polling

```python
await dp.start_polling(
    bot,
    allowed_updates=["message", "callback_query"],  # Только нужные типы
    drop_pending_updates=True,  # Пропускать старые обновления
)
```

## Что делать если ошибка все еще появляется

### Вариант 1: Увеличить таймауты еще больше

Отредактируйте `live/src/bot.py`:

```python
session = AiohttpSession(
    timeout=ClientTimeout(
        total=120,     # 2 минуты
        connect=60,    # 1 минута
    )
)
```

### Вариант 2: Проверить интернет-соединение

```bash
# Проверка доступности Telegram API
ping api.telegram.org
```

### Вариант 3: Использовать webhook вместо polling

Для production окружения лучше использовать webhook:

```python
# Вместо start_polling
await bot.set_webhook(url="https://your-domain.com/webhook")
```

### Вариант 4: Обработка ошибок

Бот автоматически переподключается при ошибках (aiogram делает это сам). Ошибки в логах - это нормально, если они не критичные.

## Важные замечания

1. **Ошибки таймаутов не критичны** - бот автоматически переподключается
2. **Evaluation не блокирует бота** - оно асинхронное и выполняется в фоне
3. **Логи показывают попытки переподключения** - это нормальное поведение

## Мониторинг

Если ошибки появляются часто:

1. Проверьте интернет-соединение
2. Проверьте доступность Telegram API
3. Увеличьте таймауты в `bot.py`
4. Рассмотрите использование webhook для production

## Логи

Нормальные логи при переподключении:
```
ERROR - Failed to fetch updates - TelegramNetworkError: ...
WARNING - Sleep for 1.000000 seconds and try again...
INFO - Connection established (tryings = 1, ...)
```

Это означает, что бот успешно переподключился.



