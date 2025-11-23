import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai/ollama
    MODEL = os.getenv("MODEL")
    MODEL_QUERY_TRANSFORM = os.getenv("MODEL_QUERY_TRANSFORM", "gpt-4o")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # Модель по умолчанию для 16GB RAM
    OLLAMA_MODEL_QUERY_TRANSFORM = os.getenv("OLLAMA_MODEL_QUERY_TRANSFORM", OLLAMA_MODEL)
    
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    DATA_DIR = os.getenv("DATA_DIR", "data")
    PROMPTS_DIR = os.getenv("PROMPTS_DIR", "prompts")
    CONVERSATION_SYSTEM_PROMPT_FILE = os.getenv("CONVERSATION_SYSTEM_PROMPT_FILE", "conversation_system.txt")
    QUERY_TRANSFORM_PROMPT_FILE = os.getenv("QUERY_TRANSFORM_PROMPT_FILE", "query_transform.txt")
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
    
    # Embeddings Configuration
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # openai/huggingface
    HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    HUGGINGFACE_DEVICE = os.getenv("HUGGINGFACE_DEVICE", "cpu")  # cpu/cuda/mps
    
    # Retrieval Configuration
    RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "semantic")  # semantic/hybrid/hybrid_reranker
    SEMANTIC_RETRIEVER_K = int(os.getenv("SEMANTIC_RETRIEVER_K", "10"))
    BM25_RETRIEVER_K = int(os.getenv("BM25_RETRIEVER_K", "10"))
    ENSEMBLE_SEMANTIC_WEIGHT = float(os.getenv("ENSEMBLE_SEMANTIC_WEIGHT", "0.5"))
    ENSEMBLE_BM25_WEIGHT = float(os.getenv("ENSEMBLE_BM25_WEIGHT", "0.5"))
    
    # Cross-Encoder Reranking Configuration
    CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "3"))
    
    # Отображение источников
    SHOW_SOURCES = os.getenv("SHOW_SOURCES", "false").lower() == "true"
    
    # LangSmith настройки
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2", "false").lower() == "true"
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-assistant")
    LANGSMITH_DATASET = os.getenv("LANGSMITH_DATASET", "06-rag-qa-dataset")
    
    # RAGAS evaluation настройки (фиксированные модели для единообразной оценки)
    RAGAS_LLM_PROVIDER = os.getenv("RAGAS_LLM_PROVIDER", LLM_PROVIDER)  # По умолчанию = основному провайдеру
    RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL")  # По умолчанию будет установлено в зависимости от провайдера
    RAGAS_OLLAMA_MODEL = os.getenv("RAGAS_OLLAMA_MODEL")  # По умолчанию будет установлено
    RAGAS_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-large")
    RAGAS_EMBEDDING_PROVIDER = os.getenv("RAGAS_EMBEDDING_PROVIDER", EMBEDDING_PROVIDER)  # По умолчанию = основному провайдеру
    # Для HuggingFace используем те же настройки что и для основных embeddings
    RAGAS_HUGGINGFACE_EMBEDDING_MODEL = os.getenv("RAGAS_HUGGINGFACE_EMBEDDING_MODEL", HUGGINGFACE_EMBEDDING_MODEL)
    RAGAS_HUGGINGFACE_DEVICE = os.getenv("RAGAS_HUGGINGFACE_DEVICE", HUGGINGFACE_DEVICE)
    
    # RAGAS timeout настройки (увеличиваем для Ollama)
    RAGAS_TIMEOUT = int(os.getenv("RAGAS_TIMEOUT", "600"))  # 10 минут по умолчанию
    RAGAS_MAX_WAIT = int(os.getenv("RAGAS_MAX_WAIT", "300"))  # 5 минут между попытками
    RAGAS_MAX_RETRIES = int(os.getenv("RAGAS_MAX_RETRIES", "5"))  # Количество попыток
    # Для Ollama используем 1 воркер по умолчанию (самый надежный, избегает таймаутов)
    # Определяем значение по умолчанию в зависимости от провайдера
    default_workers = "1" if (RAGAS_LLM_PROVIDER.lower() == "ollama" or LLM_PROVIDER.lower() == "ollama") else "4"
    RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", default_workers))
    
    # Устанавливаем RAGAS_OLLAMA_MODEL по умолчанию
    if RAGAS_OLLAMA_MODEL is None:
        RAGAS_OLLAMA_MODEL = OLLAMA_MODEL if OLLAMA_MODEL else "llama3.1:8b"
    
    # Устанавливаем RAGAS_LLM_MODEL по умолчанию в зависимости от провайдера
    if RAGAS_LLM_MODEL is None:
        if RAGAS_LLM_PROVIDER.lower() == "ollama":
            RAGAS_LLM_MODEL = RAGAS_OLLAMA_MODEL
        else:
            RAGAS_LLM_MODEL = "gpt-4o"
    
    @classmethod
    def load_prompt(cls, filename: str) -> str:
        """Загрузка промпта из файла"""
        prompt_path = Path(cls.PROMPTS_DIR) / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding='utf-8')
    
    @classmethod
    def validate(cls):
        """Валидация конфигурации"""
        # Валидация RETRIEVAL_MODE
        valid_retrieval_modes = ["semantic", "hybrid", "hybrid_reranker"]
        if cls.RETRIEVAL_MODE not in valid_retrieval_modes:
            raise ValueError(
                f"Invalid RETRIEVAL_MODE: {cls.RETRIEVAL_MODE}. "
                f"Must be one of: {', '.join(valid_retrieval_modes)}"
            )
        
        # Валидация EMBEDDING_PROVIDER
        valid_embedding_providers = ["openai", "huggingface"]
        if cls.EMBEDDING_PROVIDER not in valid_embedding_providers:
            raise ValueError(
                f"Invalid EMBEDDING_PROVIDER: {cls.EMBEDDING_PROVIDER}. "
                f"Must be one of: {', '.join(valid_embedding_providers)}"
            )
        
        # Валидация RAGAS_EMBEDDING_PROVIDER
        if cls.RAGAS_EMBEDDING_PROVIDER not in valid_embedding_providers:
            raise ValueError(
                f"Invalid RAGAS_EMBEDDING_PROVIDER: {cls.RAGAS_EMBEDDING_PROVIDER}. "
                f"Must be one of: {', '.join(valid_embedding_providers)}"
            )
        
        # Валидация LLM_PROVIDER
        valid_llm_providers = ["openai", "ollama"]
        if cls.LLM_PROVIDER not in valid_llm_providers:
            raise ValueError(
                f"Invalid LLM_PROVIDER: {cls.LLM_PROVIDER}. "
                f"Must be one of: {', '.join(valid_llm_providers)}"
            )
        
        # Валидация RAGAS_LLM_PROVIDER
        if cls.RAGAS_LLM_PROVIDER not in valid_llm_providers:
            raise ValueError(
                f"Invalid RAGAS_LLM_PROVIDER: {cls.RAGAS_LLM_PROVIDER}. "
                f"Must be one of: {', '.join(valid_llm_providers)}"
            )

config = Config()
# Валидация конфигурации при загрузке
config.validate()

