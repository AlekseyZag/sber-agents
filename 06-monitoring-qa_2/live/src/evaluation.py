import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from langsmith import Client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
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

def init_ragas_metrics():
    """
    Инициализация RAGAS метрик (один раз)
    
    По образцу референсного ноутбука (раздел 5.1)
    """
    global _ragas_metrics, _ragas_run_config
    
    if _ragas_metrics is not None:
        return _ragas_metrics, _ragas_run_config
    
    logger.info("Initializing RAGAS metrics...")
    
    # Проверка наличия API ключа
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set! Required for RAGAS evaluation.")
    
    # Настройка LLM и embeddings для RAGAS (фиксированные модели для единообразной оценки)
    # Важно: используем те же настройки API, что и для основного RAG
    llm_kwargs = {
        "model": config.RAGAS_LLM_MODEL,
        "temperature": 0,
        "api_key": config.OPENAI_API_KEY
    }
    if config.OPENAI_BASE_URL:
        llm_kwargs["base_url"] = config.OPENAI_BASE_URL
    
    try:
        langchain_llm = ChatOpenAI(**llm_kwargs)
        logger.info(f"RAGAS LLM initialized: {config.RAGAS_LLM_MODEL}")
        logger.info(f"  Base URL: {config.OPENAI_BASE_URL or 'default (OpenAI)'}")
    except Exception as e:
        logger.error(f"Failed to initialize RAGAS LLM: {e}")
        logger.error(f"  Model: {config.RAGAS_LLM_MODEL}")
        logger.error(f"  Base URL: {config.OPENAI_BASE_URL or 'default'}")
        raise ValueError(f"Cannot initialize RAGAS LLM. Check RAGAS_LLM_MODEL and OPENAI_BASE_URL settings. Error: {e}")
    
    # Настройка embeddings для RAGAS на основе провайдера
    if config.RAGAS_EMBEDDING_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        langchain_embeddings = HuggingFaceEmbeddings(
            model_name=config.RAGAS_EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Using HuggingFace embeddings for RAGAS: {config.RAGAS_EMBEDDING_MODEL}")
    else:
        embedding_kwargs = {
            "model": config.RAGAS_EMBEDDING_MODEL
        }
        if config.OPENAI_BASE_URL:
            embedding_kwargs["base_url"] = config.OPENAI_BASE_URL
        if config.OPENAI_API_KEY:
            embedding_kwargs["api_key"] = config.OPENAI_API_KEY
        
        from langchain_openai import OpenAIEmbeddings
        langchain_embeddings = OpenAIEmbeddings(**embedding_kwargs)
        logger.info(f"Using OpenAI embeddings for RAGAS: {config.RAGAS_EMBEDDING_MODEL} (base_url: {config.OPENAI_BASE_URL or 'default'})")
    
    # Создаем метрики
    metrics = [
        Faithfulness(),
        AnswerRelevancy(strictness=1),
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
    run_config = RunConfig(
        max_workers=4,
        max_wait=180,
        max_retries=3
    )
    
    _ragas_metrics = metrics
    _ragas_run_config = run_config
    
    logger.info(f"RAGAS metrics initialized: {', '.join([m.name for m in metrics])}")
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
    
    # Проверяем существование датасета
    if not check_dataset_exists(dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' not found in LangSmith")
    
    # Инициализируем метрики
    ragas_metrics, ragas_run_config = init_ragas_metrics()
    
    client = Client()
    
    # ========== Шаг 1: Запуск эксперимента и сбор данных ==========
    logger.info("\n[1/3] Running experiment and collecting data...")
    
    # Создаем target функцию для нашего RAG
    def target(inputs: dict) -> dict:
        """Target функция для evaluation"""
        question = inputs["question"]
        
        # Используем существующую RAG цепочку
        # Передаем только вопрос (без истории для evaluation)
        from langchain_core.messages import HumanMessage
        result = rag.get_rag_chain().invoke({"messages": [HumanMessage(content=question)]})
        
        return {
            "answer": result["answer"],
            "documents": result["documents"]
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
        # Вопрос берем из example.inputs (из датасета), а не из run.inputs
        question = example.inputs.get("question", "") if example else run.inputs.get("question", "")
        answer = run.outputs.get("answer", "") if run.outputs else ""
        documents = run.outputs.get("documents", []) if run.outputs else []
        
        # Обработка documents - могут быть Document объекты или строки
        if documents:
            contexts = []
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    contexts.append(doc.page_content)
                elif isinstance(doc, dict):
                    contexts.append(doc.get('page_content', str(doc)))
                else:
                    contexts.append(str(doc))
        else:
            contexts = []
        
        ground_truth = example.outputs.get("answer", "") if example and example.outputs else ""
        
        # Логируем для диагностики, если данные пустые
        if not question:
            logger.debug(f"Run {run.id}: Empty question. example.inputs={example.inputs if example else None}, run.inputs={run.inputs}")
        if not answer and run.outputs:
            logger.debug(f"Run {run.id}: Empty answer. run.outputs={run.outputs}")
        
        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)
        run_ids.append(str(run.id))
    
    logger.info(f"Experiment completed, collected {len(questions)} examples")
    
    # Проверка данных перед evaluation
    logger.info("\nValidating data before RAGAS evaluation...")
    empty_ground_truth_count = 0
    for i, (q, a, ctx, gt) in enumerate(zip(questions, answers, contexts_list, ground_truths)):
        if not q or not q.strip():
            logger.warning(f"Example {i+1}: Empty question")
        if not a or not a.strip():
            logger.warning(f"Example {i+1}: Empty answer")
        if not ctx or len(ctx) == 0:
            logger.warning(f"Example {i+1}: Empty contexts")
        elif not isinstance(ctx, list):
            logger.warning(f"Example {i+1}: contexts is not a list: {type(ctx)}")
        if not gt or not gt.strip():
            empty_ground_truth_count += 1
            logger.warning(f"Example {i+1}: Empty ground_truth (required for AnswerCorrectness)")
    
    if empty_ground_truth_count > 0:
        logger.warning(f"⚠️ Found {empty_ground_truth_count} examples with empty ground_truth!")
        logger.warning("  AnswerCorrectness requires ground_truth - these examples will have NaN for this metric")
    
    # ========== Шаг 2: RAGAS evaluation ==========
    logger.info("\n[2/3] Running RAGAS evaluation...")
    
    # Создаем Dataset для RAGAS
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    })
    
    # Запускаем evaluation с обработкой ошибок
    try:
        ragas_result = evaluate(
            ragas_dataset,
            metrics=ragas_metrics,
            run_config=ragas_run_config,
        )
        
        ragas_df = ragas_result.to_pandas()
        logger.info("RAGAS evaluation completed")
        
        # Логируем детали по каждой метрике
        logger.info("\nDetailed metric results:")
        for metric in ragas_metrics:
            if metric.name in ragas_df.columns:
                col = ragas_df[metric.name]
                nan_count = col.isna().sum()
                valid_count = col.notna().sum()
                avg_score = col.mean()
                
                logger.info(f"  {metric.name}:")
                logger.info(f"    Valid scores: {valid_count}/{len(col)}")
                logger.info(f"    NaN scores: {nan_count}/{len(col)}")
                if valid_count > 0:
                    logger.info(f"    Average: {avg_score:.3f}")
                    logger.info(f"    Min: {col.min():.3f}, Max: {col.max():.3f}")
                else:
                    logger.warning(f"    ⚠️ All scores are NaN!")
                    
                    # Показываем примеры с NaN для диагностики
                    nan_indices = col[col.isna()].index.tolist()[:3]
                    for idx in nan_indices:
                        q = questions[idx][:50] if questions[idx] else "EMPTY"
                        a = answers[idx][:50] if answers[idx] else "EMPTY"
                        gt = ground_truths[idx][:50] if ground_truths[idx] else "EMPTY"
                        ctx_count = len(contexts_list[idx]) if contexts_list[idx] else 0
                        logger.warning(f"      Example {idx+1}:")
                        logger.warning(f"        question='{q}...'")
                        logger.warning(f"        answer='{a}...'")
                        logger.warning(f"        ground_truth='{gt}...'")
                        logger.warning(f"        contexts count: {ctx_count}")
                        
                        # Специальная диагностика для AnswerCorrectness
                        if metric.name == "answer_correctness":
                            if not ground_truths[idx] or not ground_truths[idx].strip():
                                logger.warning(f"        ⚠️ Empty ground_truth - AnswerCorrectness requires it!")
                            if not answers[idx] or not answers[idx].strip():
                                logger.warning(f"        ⚠️ Empty answer - AnswerCorrectness requires it!")
        
    except Exception as e:
        logger.error(f"Error during RAGAS evaluation: {e}", exc_info=True)
        raise
    
    # Вычисляем средние значения метрик (игнорируя NaN)
    metrics_summary = {}
    for metric in ragas_metrics:
        if metric.name in ragas_df.columns:
            col = ragas_df[metric.name]
            # Используем nanmean для игнорирования NaN значений
            avg_score = np.nanmean(col)
            if np.isnan(avg_score):
                logger.warning(f"  {metric.name}: Cannot compute average (all values are NaN)")
                metrics_summary[metric.name] = float('nan')
            else:
                metrics_summary[metric.name] = avg_score
                logger.info(f"  {metric.name}: {avg_score:.3f} (computed from {col.notna().sum()}/{len(col)} valid values)")
    
    # ========== Шаг 3: Загрузка feedback в LangSmith ==========
    logger.info("\n[3/3] Uploading feedback to LangSmith...")
    
    feedback_count = 0
    for idx, run_id in enumerate(run_ids):
        row = ragas_df.iloc[idx]
        
        for metric in ragas_metrics:
            if metric.name in row:
                score = row[metric.name]
                # Пропускаем NaN значения
                if pd.isna(score) or np.isnan(score):
                    logger.debug(f"Skipping NaN feedback for {metric.name} on run {run_id}")
                    continue
                
                try:
                    client.create_feedback(
                        run_id=run_id,
                        key=metric.name,
                        score=float(score),
                        comment=f"RAGAS metric: {metric.name}"
                    )
                    feedback_count += 1
                except Exception as e:
                    logger.error(f"Error creating feedback for {metric.name} on run {run_id}: {e}")
    
    logger.info(f"Feedback uploaded ({len(run_ids)} runs)")
    
    return {
        "dataset_name": dataset_name,
        "num_examples": len(questions),
        "metrics": metrics_summary,
        "ragas_result": ragas_result,
        "run_ids": run_ids
    }

