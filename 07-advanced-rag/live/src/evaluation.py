import logging
import json
import re
from typing import Optional, Dict, Any
import httpx
from langsmith import Client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    AnswerCorrectness,
    AnswerSimilarity,
    ContextRecall,
    ContextPrecision,
)
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from config import config
import rag

logger = logging.getLogger(__name__)

# Глобальные инициализированные метрики
_ragas_metrics = None
_ragas_run_config = None

def extract_json_from_text(text: str) -> str:
    """Извлекает JSON из текста, удаляя все лишнее"""
    text = text.strip()
    
    # Пытаемся найти JSON в markdown code block
    if "```json" in text:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
    elif "```" in text:
        # Пробуем любой code block
        parts = text.split("```")
        if len(parts) >= 2:
            json_part = parts[1].strip()
            if json_part.startswith("json"):
                json_part = json_part[4:].strip()
            # Проверяем, что это похоже на JSON
            if json_part.startswith("{") or json_part.startswith("["):
                return json_part
    
    # Ищем JSON объект напрямую
    # Ищем первую открывающую скобку
    start_idx = text.find("{")
    if start_idx == -1:
        start_idx = text.find("[")
    
    if start_idx >= 0:
        # Находим соответствующую закрывающую скобку
        bracket = text[start_idx]
        close_bracket = "}" if bracket == "{" else "]"
        depth = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(text)):
            if text[i] == bracket:
                depth += 1
            elif text[i] == close_bracket:
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        
        json_text = text[start_idx:end_idx]
        # Проверяем валидность JSON
        try:
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            pass
    
    # Если ничего не нашли, возвращаем как есть
    return text

class CleanedChatOllama:
    """Обертка для ChatOllama, которая очищает JSON из ответов"""
    
    def __init__(self, ollama_llm):
        self.ollama_llm = ollama_llm
    
    def __getattr__(self, name):
        """Проксируем все атрибуты к оригинальному LLM"""
        return getattr(self.ollama_llm, name)
    
    def invoke(self, input, config=None, **kwargs):
        """Перехватываем invoke и очищаем ответ"""
        response = self.ollama_llm.invoke(input, config=config, **kwargs)
        
        # Очищаем контент от лишнего текста
        if hasattr(response, 'content'):
            cleaned_content = extract_json_from_text(response.content)
            # Создаем новый объект с очищенным контентом
            from langchain_core.messages import AIMessage
            return AIMessage(content=cleaned_content)
        
        return response
    
    async def ainvoke(self, input, config=None, **kwargs):
        """Асинхронная версия"""
        response = await self.ollama_llm.ainvoke(input, config=config, **kwargs)
        
        if hasattr(response, 'content'):
            cleaned_content = extract_json_from_text(response.content)
            from langchain_core.messages import AIMessage
            return AIMessage(content=cleaned_content)
        
        return response

def _get_ollama_chat():
    """Ленивый импорт ChatOllama"""
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama not installed. Install it with: uv add langchain-ollama"
        )

def create_ragas_llm():
    """
    Фабрика для создания LLM для RAGAS по провайдеру из конфига
    Поддерживает: openai, ollama
    """
    provider = config.RAGAS_LLM_PROVIDER.lower()
    
    if provider == "openai":
        logger.info(f"Creating RAGAS OpenAI LLM: {config.RAGAS_LLM_MODEL}")
        return ChatOpenAI(model=config.RAGAS_LLM_MODEL, temperature=0)
    
    elif provider == "ollama":
        ChatOllama = _get_ollama_chat()
        logger.info(f"Creating RAGAS Ollama LLM: {config.RAGAS_LLM_MODEL} at {config.OLLAMA_BASE_URL}")
        logger.info("Using JSON cleaning wrapper for Ollama responses")
        # Увеличиваем таймаут для Ollama (работает медленнее)
        ollama_llm = ChatOllama(
            model=config.RAGAS_LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            timeout=config.RAGAS_TIMEOUT  # Таймаут для HTTP запросов
        )
        # Обертываем для очистки JSON из ответов
        return CleanedChatOllama(ollama_llm)
    
    else:
        raise ValueError(f"Unknown RAGAS LLM provider: {provider}. Use 'openai' or 'ollama'")

def create_ragas_embeddings():
    """
    Фабрика для создания RAGAS embeddings по провайдеру из конфига
    Поддерживает: openai, huggingface
    """
    provider = config.RAGAS_EMBEDDING_PROVIDER.lower()
    
    if provider == "openai":
        logger.info(f"Creating RAGAS OpenAI embeddings: {config.RAGAS_EMBEDDING_MODEL}")
        return OpenAIEmbeddings(model=config.RAGAS_EMBEDDING_MODEL)
    
    elif provider == "huggingface":
        logger.info(f"Creating RAGAS HuggingFace embeddings: {config.RAGAS_HUGGINGFACE_EMBEDDING_MODEL} on {config.RAGAS_HUGGINGFACE_DEVICE}")
        return HuggingFaceEmbeddings(
            model_name=config.RAGAS_HUGGINGFACE_EMBEDDING_MODEL,
            model_kwargs={'device': config.RAGAS_HUGGINGFACE_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    else:
        raise ValueError(f"Unknown RAGAS embedding provider: {provider}. Use 'openai' or 'huggingface'")

def init_ragas_metrics():
    """
    Инициализация RAGAS метрик (один раз)
    
    По образцу референсного ноутбука (раздел 5.1)
    """
    global _ragas_metrics, _ragas_run_config
    
    if _ragas_metrics is not None:
        return _ragas_metrics, _ragas_run_config
    
    logger.info("Initializing RAGAS metrics...")
    
    # Настройка LLM и embeddings для RAGAS (фиксированные модели для единообразной оценки)
    langchain_llm = create_ragas_llm()
    langchain_embeddings = create_ragas_embeddings()
    
    # Создаем метрики
    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),
        AnswerCorrectness(),
        AnswerSimilarity(),
        ContextRecall(),
        ContextPrecision(),
    ]
    
    # Инициализируем метрики
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = ragas_llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = ragas_embeddings
        run_config = RunConfig()
        metric.init(run_config)
    
    # Настройки для выполнения
    # Увеличиваем таймауты для Ollama (работает медленнее чем OpenAI)
    if config.RAGAS_LLM_PROVIDER.lower() == "ollama":
        logger.info(f"Using extended timeouts for Ollama: timeout={config.RAGAS_TIMEOUT}s, max_wait={config.RAGAS_MAX_WAIT}s")
        logger.info(f"Using {config.RAGAS_MAX_WORKERS} worker(s) for Ollama (sequential processing to avoid timeouts)")
        run_config = RunConfig(
            max_workers=config.RAGAS_MAX_WORKERS,  # 1 воркер по умолчанию для Ollama (избегает таймаутов)
            timeout=config.RAGAS_TIMEOUT,  # Увеличенный таймаут
            max_wait=config.RAGAS_MAX_WAIT,  # Увеличенное время ожидания
            max_retries=config.RAGAS_MAX_RETRIES  # Больше попыток
        )
    else:
        run_config = RunConfig(
            max_workers=4,
            max_wait=180,
            max_retries=3
        )
    
    _ragas_metrics = metrics
    _ragas_run_config = run_config
    
    logger.info(f"✓ RAGAS metrics initialized: {', '.join([m.name for m in metrics])}")
    logger.info(f"✓ RAGAS LLM Provider: {config.RAGAS_LLM_PROVIDER}")
    logger.info(f"✓ RAGAS LLM Model: {config.RAGAS_LLM_MODEL}")
    logger.info(f"✓ RAGAS Embedding Provider: {config.RAGAS_EMBEDDING_PROVIDER}")
    if config.RAGAS_EMBEDDING_PROVIDER == "openai":
        logger.info(f"✓ RAGAS Embedding Model: {config.RAGAS_EMBEDDING_MODEL}")
    else:
        logger.info(f"✓ RAGAS Embedding Model: {config.RAGAS_HUGGINGFACE_EMBEDDING_MODEL} on {config.RAGAS_HUGGINGFACE_DEVICE}")
    
    return _ragas_metrics, _ragas_run_config

def check_dataset_exists(dataset_name: str) -> bool:
    """
    Проверка существования датасета в LangSmith
    
    Args:
        dataset_name: имя датасета
    
    Returns:
        True если датасет существует
    """
    if not config.LANGSMITH_API_KEY:
        logger.error("LANGSMITH_API_KEY not set")
        return False
    
    try:
        client = Client()
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        return len(datasets) > 0
    except Exception as e:
        logger.error(f"Error checking dataset: {e}")
        return False

def check_ollama_connection():
    """Проверка доступности Ollama сервера"""
    if config.RAGAS_LLM_PROVIDER.lower() == "ollama" or config.LLM_PROVIDER.lower() == "ollama":
        try:
            response = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if response.status_code == 200:
                logger.info(f"✓ Ollama server is accessible at {config.OLLAMA_BASE_URL}")
                return True
            else:
                logger.warning(f"⚠️ Ollama server returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama at {config.OLLAMA_BASE_URL}: {e}")
            logger.error("   Make sure Ollama is running: ollama serve")
            return False
    return True  # Не проверяем, если не используем Ollama

def evaluate_dataset(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Главная функция evaluation RAG системы
    
    По образцу референсного ноутбука (раздел 5.2):
    1. Запуск эксперимента в LangSmith с blocking=False и сбор данных
    2. RAGAS batch evaluation
    3. Загрузка метрик как feedback в LangSmith
    
    Args:
        dataset_name: имя датасета (по умолчанию из конфига)
    
    Returns:
        dict с результатами evaluation
    """
    if not config.LANGSMITH_API_KEY:
        raise ValueError("LANGSMITH_API_KEY not set. Cannot run evaluation.")
    
    if dataset_name is None:
        dataset_name = config.LANGSMITH_DATASET
    
    logger.info(f"Starting evaluation for dataset: {dataset_name}")
    
    # Проверяем доступность Ollama если используется
    if not check_ollama_connection():
        raise ConnectionError(
            f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. "
            f"Make sure Ollama is running: ollama serve"
        )
    
    # Проверяем существование датасета
    if not check_dataset_exists(dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' not found in LangSmith")
    
    # Инициализируем метрики
    ragas_metrics, ragas_run_config = init_ragas_metrics()
    
    client = Client()
    
    # ========== Шаг 1: Запуск эксперимента и сбор данных ==========
    logger.info("\n[1/3] Running experiment and collecting data...")
    
    # Проверяем инициализацию для hybrid режима
    if config.RETRIEVAL_MODE.lower() in ["hybrid", "hybrid_reranker"]:
        if rag.chunks is None or len(rag.chunks) == 0:
            logger.warning("⚠️ Chunks not initialized for BM25. Hybrid retrieval may not work correctly.")
            logger.warning("⚠️ Run /index command to reindex documents.")
        else:
            logger.info(f"✓ Chunks initialized for BM25: {len(rag.chunks)} chunks")
    
    # Создаем target функцию для нашего RAG
    def target(inputs: dict) -> dict:
        """Target функция для evaluation"""
        question = inputs["question"]
        
        try:
            # Используем RAG цепочку БЕЗ query transformation для evaluation
            # Это позволяет использовать оригинальный вопрос напрямую, что важно для точности метрик
            from langchain_core.messages import HumanMessage
            result = rag.get_rag_chain(use_query_transformation=False).invoke({
                "messages": [HumanMessage(content=question)]
            })
            
            # Логируем для диагностики
            num_docs = len(result.get("documents", []))
            if num_docs == 0:
                logger.warning(f"⚠️ No documents retrieved for question: {question[:50]}...")
            else:
                logger.debug(f"✓ Retrieved {num_docs} documents for question: {question[:50]}...")
            
            return {
                "answer": result["answer"],
                "documents": result["documents"]
            }
        except Exception as e:
            logger.error(f"❌ Error processing question '{question[:50]}...': {e}")
            # Возвращаем пустой результат при ошибке
            return {
                "answer": f"Error: {str(e)}",
                "documents": []
            }
    
    # Собираем данные во время выполнения evaluate
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    run_ids = []
    
    # evaluate() с blocking=False возвращает итератор
    for result in client.evaluate(
        target,
        data=dataset_name,
        evaluators=[],
        experiment_prefix="rag-evaluation",
        metadata={
            "approach": "RAGAS batch evaluation + LangSmith feedback",
            "model": config.MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
        },
        blocking=False,
    ):
        run = result["run"]
        example = result["example"]
        
        # Получаем данные
        question = run.inputs.get("question", "")
        answer = run.outputs.get("answer", "")
        documents = run.outputs.get("documents", [])
        
        # Извлекаем contexts из documents
        contexts = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                contexts.append(doc.page_content)
            elif isinstance(doc, dict):
                contexts.append(doc.get('page_content', str(doc)))
            else:
                contexts.append(str(doc))
        
        ground_truth = example.outputs.get("answer", "") if example else ""
        
        # Диагностика: логируем если нет документов
        if len(documents) == 0:
            logger.warning(f"⚠️ No documents retrieved for question: {question[:80]}...")
        elif len(contexts) == 0:
            logger.warning(f"⚠️ No contexts extracted from documents for question: {question[:80]}...")
        
        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)
        run_ids.append(str(run.id))
    
    logger.info(f"Experiment completed, collected {len(questions)} examples")
    
    # ========== Шаг 2: RAGAS evaluation ==========
    logger.info("\n[2/3] Running RAGAS evaluation...")
    
    # Создаем Dataset для RAGAS
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    })
    
    # Запускаем evaluation
    ragas_result = evaluate(
        ragas_dataset,
        metrics=ragas_metrics,
        run_config=ragas_run_config,
    )
    
    ragas_df = ragas_result.to_pandas()
    
    logger.info("RAGAS evaluation completed")
    
    # Вычисляем средние значения метрик
    metrics_summary = {}
    for metric in ragas_metrics:
        if metric.name in ragas_df.columns:
            avg_score = ragas_df[metric.name].mean()
            metrics_summary[metric.name] = avg_score
            logger.info(f"  {metric.name}: {avg_score:.3f}")
    
    # ========== Шаг 3: Загрузка feedback в LangSmith ==========
    logger.info("\n[3/3] Uploading feedback to LangSmith...")
    
    for idx, run_id in enumerate(run_ids):
        row = ragas_df.iloc[idx]
        
        for metric in ragas_metrics:
            if metric.name in row:
                score = row[metric.name]
                client.create_feedback(
                    run_id=run_id,
                    key=metric.name,
                    score=float(score),
                    comment=f"RAGAS metric: {metric.name}"
                )
    
    logger.info(f"Feedback uploaded ({len(run_ids)} runs)")
    
    return {
        "dataset_name": dataset_name,
        "num_examples": len(questions),
        "metrics": metrics_summary,
        "ragas_result": ragas_result,
        "run_ids": run_ids
    }

