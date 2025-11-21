import logging
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import config
import indexer
import rag
import evaluation

logger = logging.getLogger(__name__)
router = Router()

# Глобальный словарь для хранения историй диалогов в формате LangChain Messages
chat_conversations: dict[int, list] = {}

@router.message(Command("start"))
async def cmd_start(message: Message):
    logger.info(f"User {message.chat.id} started the bot")
    
    # Инициализируем историю с системным промптом в LangChain формате
    chat_conversations[message.chat.id] = [
        SystemMessage(content=config.SYSTEM_PROMPT)
    ]
    
    await message.answer(
        "Привет! Я RAG-ассистент Сбербанка.\n\n"
        "Я могу:\n"
        "• Отвечать на вопросы по документам\n"
        "• Помогать с информацией о кредитах и вкладах\n"
        "• Поддерживать диалог с учетом контекста\n\n"
        "Используйте /help для просмотра всех команд."
    )

@router.message(Command("help"))
async def cmd_help(message: Message):
    logger.info(f"User {message.chat.id} requested help")
    help_text = (
        "🤖 *RAG-ассистент Сбербанка*\n\n"
        "Я помогаю отвечать на вопросы по документам о кредитах и вкладах.\n\n"
        "*Доступные команды:*\n"
        "/start - Начать новый диалог (сбросить историю)\n"
        "/help - Показать эту справку\n"
        "/index - Переиндексировать документы\n"
        "/index\\_status - Проверить статус индексации\n"
        "/evaluate\\_dataset - Оценить качество RAG системы\n\n"
        "*Возможности:*\n"
        "• Ответы на вопросы по документам\n"
        "• Понимание уточняющих вопросов\n"
        "• Сохранение контекста диалога\n"
        "• Оценка качества через RAGAS метрики\n\n"
        "*Примеры вопросов:*\n"
        "• Какие условия потребительского кредита?\n"
        "• Какие проценты по вкладам?\n"
        "• Можно ли досрочно погасить кредит?\n\n"
        "_Если вопрос выходит за рамки документов, я сообщу об этом\\._"
    )
    await message.answer(help_text, parse_mode="Markdown")

@router.message(Command("index"))
async def cmd_index(message: Message):
    logger.info(f"User {message.chat.id} requested reindexing")
    await message.answer("Начинаю переиндексацию документов...")
    
    try:
        rag.vector_store = await indexer.reindex_all()
        if rag.vector_store:
            rag.initialize_retriever()
            stats = rag.get_vector_store_stats()
            await message.answer(
                f"✅ Переиндексация завершена!\n"
                f"Проиндексировано документов: {stats['count']}"
            )
        else:
            await message.answer("⚠️ Не найдено документов для индексации")
    except Exception as e:
        logger.error(f"Error during reindexing: {e}")
        await message.answer(f"❌ Ошибка при переиндексации: {str(e)}")

@router.message(Command("index_status"))
async def cmd_index_status(message: Message):
    logger.info(f"User {message.chat.id} requested index status")
    stats = rag.get_vector_store_stats()
    
    if stats["status"] == "not initialized":
        await message.answer("⚠️ Векторное хранилище не инициализировано")
    else:
        await message.answer(
            f"📊 Статус индексации:\n"
            f"Статус: {stats['status']}\n"
            f"Количество документов: {stats['count']}"
        )

@router.message(Command("evaluate_dataset"))
async def cmd_evaluate_dataset(message: Message):
    logger.info(f"User {message.chat.id} requested dataset evaluation")
    
    # Проверка API ключа
    if not config.LANGSMITH_API_KEY:
        await message.answer(
            "⚠️ LangSmith API key не настроен.\n"
            "Установите LANGSMITH_API_KEY в .env файле для использования evaluation."
        )
        return
    
    # Проверка векторного хранилища
    if rag.vector_store is None or rag.retriever is None:
        await message.answer(
            "⚠️ Векторное хранилище не инициализировано.\n"
            "Используйте /index для индексации документов."
        )
        return
    
    # Извлекаем название датасета из команды (опционально)
    command_parts = message.text.split(maxsplit=1)
    dataset_name = command_parts[1] if len(command_parts) > 1 else None
    
    if dataset_name is None:
        dataset_name = config.LANGSMITH_DATASET
        await message.answer(
            f"🔍 Начинаю evaluation датасета: {dataset_name}\n\n"
            f"Это может занять несколько минут...\n"
            f"Шаг 1/3: Запуск эксперимента в LangSmith..."
        )
    else:
        await message.answer(
            f"🔍 Начинаю evaluation датасета: {dataset_name}\n\n"
            f"Это может занять несколько минут..."
        )
    
    try:
        # Запускаем evaluation
        result = evaluation.evaluate_dataset(dataset_name)
        
        # Формируем отчет
        metrics = result["metrics"]
        num_examples = result["num_examples"]
        
        report = (
            f"✅ Evaluation завершен!\n\n"
            f"📊 Датасет: {dataset_name}\n"
            f"📝 Примеров обработано: {num_examples}\n\n"
            f"🎯 RAGAS Метрики:\n"
        )
        
        # Добавляем метрики с описанием
        metric_descriptions = {
            "faithfulness": "Обоснованность (нет галлюцинаций)",
            "answer_relevancy": "Релевантность ответа",
            "answer_correctness": "Правильность ответа",
            "answer_similarity": "Похожесть на эталон",
            "context_recall": "Полнота контекста",
            "context_precision": "Точность поиска"
        }
        
        import math
        for metric_name, score in metrics.items():
            desc = metric_descriptions.get(metric_name, metric_name)
            
            # Обработка NaN значений
            if score is None or (isinstance(score, float) and math.isnan(score)):
                emoji = "⚪"
                report += f"{emoji} {desc}: не вычислено (NaN)\n"
                continue
            
            # Эмодзи в зависимости от оценки
            if score >= 0.8:
                emoji = "🟢"
            elif score >= 0.6:
                emoji = "🟡"
            else:
                emoji = "🔴"
            report += f"{emoji} {desc}: {score:.3f}\n"
        
        report += f"\n💡 Результаты загружены в LangSmith как feedback"
        
        await message.answer(report)
        logger.info(f"Evaluation completed for user {message.chat.id}")
        
    except ValueError as e:
        logger.error(f"ValueError in evaluation: {e}")
        await message.answer(f"❌ Ошибка: {str(e)}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        await message.answer(
            f"❌ Произошла ошибка при evaluation:\n{str(e)}\n\n"
            f"Проверьте логи для подробностей."
        )

@router.message()
async def handle_message(message: Message):
    # Игнорируем сообщения без текста (стикеры, фото и т.д.)
    if not message.text:
        await message.answer("Извините, я работаю только с текстовыми сообщениями.")
        return
    
    logger.info(f"Message from {message.chat.id}: {message.text[:100]}...")
    
    # Инициализируем историю если её нет
    if message.chat.id not in chat_conversations:
        chat_conversations[message.chat.id] = [
            SystemMessage(content=config.SYSTEM_PROMPT)
        ]
    
    # Добавляем сообщение пользователя в историю
    chat_conversations[message.chat.id].append(
        HumanMessage(content=message.text)
    )
    
    try:
        # Проверка инициализации векторного хранилища
        if rag.vector_store is None or rag.retriever is None:
            logger.warning(f"Vector store not initialized for chat {message.chat.id}")
            await message.answer(
                "⚠️ Векторное хранилище не инициализировано. "
                "Пожалуйста, подождите или используйте /index для индексации."
            )
            # Удаляем последнее сообщение из истории
            chat_conversations[message.chat.id].pop()
            return
        
        # Получаем ответ через RAG (передаем историю без system message)
        # Теперь возвращает dict с answer и documents
        result = await rag.rag_answer(chat_conversations[message.chat.id][1:])
        answer = result["answer"]
        documents = result["documents"]
        
        # Добавляем ответ в историю
        chat_conversations[message.chat.id].append(
            AIMessage(content=answer)
        )
        
        # Формируем итоговый ответ с источниками если включено
        final_response = answer
        if config.SHOW_SOURCES and documents:
            sources = rag.format_sources(documents)
            if sources:
                final_response = f"{answer}\n\n{sources}"
        
        await message.answer(final_response)
        
    except ValueError as e:
        logger.error(f"ValueError in handle_message for chat {message.chat.id}: {e}")
        # Удаляем последнее сообщение из истории
        chat_conversations[message.chat.id].pop()
        await message.answer(
            "⚠️ Векторное хранилище не готово. "
            "Используйте /index для индексации документов."
        )
    except Exception as e:
        logger.error(f"Error in handle_message for chat {message.chat.id}: {e}", exc_info=True)
        # Удаляем последнее сообщение из истории
        chat_conversations[message.chat.id].pop()
        await message.answer(
            "Произошла ошибка при обработке вашего сообщения. "
            "Попробуйте еще раз или используйте /start для начала нового диалога."
        )

