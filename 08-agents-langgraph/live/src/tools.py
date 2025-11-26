"""
Инструменты для ReAct агента

Инструменты - это функции, которые агент может вызывать для получения информации.
Декоратор @tool из LangChain автоматически создает описание для LLM.
"""
import json
import logging
import requests
from langchain_core.tools import tool
import rag
from config import config

logger = logging.getLogger(__name__)

@tool
def rag_search(query: str) -> str:
    """
    Ищет информацию в документах Сбербанка (условия кредитов, вкладов и других банковских продуктов).
    
    Возвращает JSON со списком источников, где каждый источник содержит:
    - source: имя файла
    - page: номер страницы (только для PDF)
    - page_content: текст документа
    """
    try:
        # Получаем релевантные документы через RAG (retrieval + reranking)
        documents = rag.retrieve_documents(query)
        
        if not documents:
            return json.dumps({"sources": []}, ensure_ascii=False)
        
        # Формируем структурированный ответ для агента
        sources = []
        for doc in documents:
            source_data = {
                "source": doc.metadata.get("source", "Unknown"),
                "page_content": doc.page_content  # Полный текст документа
            }
            # page только для PDF (у JSON документов его нет)
            if "page" in doc.metadata:
                source_data["page"] = doc.metadata["page"]
            sources.append(source_data)
        
        # ensure_ascii=False для корректной кириллицы
        return json.dumps({"sources": sources}, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error in rag_search: {e}", exc_info=True)
        return json.dumps({"sources": []}, ensure_ascii=False)


@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Конвертирует сумму из одной валюты в другую используя актуальные курсы валют.
    
    Args:
        amount: Сумма для конвертации (например, 100.0)
        from_currency: Исходная валюта в формате ISO (USD, EUR, RUB, CNY и т.д.)
        to_currency: Целевая валюта в формате ISO (USD, EUR, RUB, CNY и т.д.)
    
    Returns:
        Строка с результатом конвертации и актуальным курсом
    """
    try:
        # Проверяем наличие API ключа
        if not config.EXCHANGE_RATE_API_KEY:
            return (
                "Ошибка: не настроен API ключ для конвертации валют.\n"
                "Получите бесплатный ключ на https://exchangerate-api.com\n"
                "и добавьте EXCHANGE_RATE_API_KEY в .env файл"
            )
        
        # Приводим валюты к верхнему регистру
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        # Проверяем корректность суммы
        if amount <= 0:
            return "Ошибка: сумма должна быть положительным числом"
        
        # Запрос к API конвертации валют с конфигурируемым URL
        url = f"{config.EXCHANGE_RATE_API_URL}/{config.EXCHANGE_RATE_API_KEY}/latest/{from_currency}"
        
        logger.info(f"Currency conversion request: {amount} {from_currency} -> {to_currency}")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Проверяем успешность запроса
        if data.get('result') != 'success':
            error_type = data.get('error-type', 'unknown')
            if error_type == 'unsupported-code':
                return f"Ошибка: валюта {from_currency} не поддерживается"
            elif error_type == 'quota-reached':
                return "Ошибка: превышен лимит запросов к API конвертации валют"
            else:
                return f"Ошибка API: {error_type}"
        
        # Проверяем, есть ли целевая валюта в ответе
        conversion_rates = data.get('conversion_rates', {})
        if to_currency not in conversion_rates:
            return f"Ошибка: валюта {to_currency} не поддерживается или неверно указана"
        
        # Получаем курс и рассчитываем конвертацию
        exchange_rate = conversion_rates[to_currency]
        converted_amount = amount * exchange_rate
        
        # Форматируем результат
        result = (
            f"Конвертация валют:\n"
            f"{amount:,.2f} {from_currency} = {converted_amount:,.2f} {to_currency}\n"
            f"Курс: 1 {from_currency} = {exchange_rate:.4f} {to_currency}\n"
            f"Данные актуальны на {data.get('time_last_update_utc', 'сегодня')}"
        )
        
        logger.info(f"Currency conversion successful: {amount} {from_currency} = {converted_amount:.2f} {to_currency}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error in currency_converter: {e}")
        return "Ошибка: не удалось получить актуальные курсы валют. Проверьте подключение к интернету."
    
    except KeyError as e:
        logger.error(f"Currency not found in currency_converter: {e}")
        return f"Ошибка: валюта {from_currency} не поддерживается или неверно указана"
    
    except Exception as e:
        logger.error(f"Error in currency_converter: {e}", exc_info=True)
        return "Ошибка при конвертации валют. Попробуйте позже или проверьте правильность указанных валют."

