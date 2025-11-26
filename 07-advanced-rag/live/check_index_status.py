"""
Скрипт для проверки статуса индексации из командной строки
Использование: python check_index_status.py
"""
import asyncio
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import config
import indexer
import rag

async def main():
    """Проверка статуса индексации"""
    print("=" * 70)
    print("📊 Проверка статуса индексации")
    print("=" * 70)
    
    # Проверяем текущий статус
    stats = rag.get_vector_store_stats()
    
    print(f"\n📋 Конфигурация:")
    print(f"  Режим retrieval: {stats['retrieval_mode']}")
    print(f"  Провайдер embeddings: {stats['embedding_provider']}")
    
    if stats['embedding_provider'] == 'openai':
        print(f"  Модель embeddings: {stats.get('embedding_model', 'N/A')}")
    elif stats['embedding_provider'] == 'huggingface':
        print(f"  Модель embeddings: {stats.get('embedding_model', 'N/A')}")
        print(f"  Устройство: {stats.get('device', 'N/A')}")
    
    print(f"\n📚 Статус индексации:")
    print(f"  Статус: {stats['status']}")
    print(f"  Документов: {stats['count']}")
    
    # Информация о chunks для hybrid режима
    if stats['retrieval_mode'] in ['hybrid', 'hybrid_reranker']:
        chunks_count = len(rag.chunks) if rag.chunks else 0
        print(f"  Chunks для BM25: {chunks_count}")
        if chunks_count == 0:
            print("  ⚠️  ВНИМАНИЕ: Chunks не инициализированы для BM25!")
            print("  ⚠️  Hybrid retrieval не будет работать корректно!")
    
    # Параметры retrieval
    print(f"\n🔍 Параметры Retrieval:")
    if stats['retrieval_mode'] == 'semantic':
        print(f"  Semantic k: {stats.get('semantic_k', 'N/A')}")
    elif stats['retrieval_mode'] == 'hybrid':
        print(f"  Semantic k: {stats.get('semantic_k', 'N/A')}")
        print(f"  BM25 k: {stats.get('bm25_k', 'N/A')}")
        print(f"  Веса: {stats.get('semantic_weight', 0):.1f}/{stats.get('bm25_weight', 0):.1f}")
    elif stats['retrieval_mode'] == 'hybrid_reranker':
        print(f"  Semantic k: {stats.get('semantic_k', 'N/A')}")
        print(f"  BM25 k: {stats.get('bm25_k', 'N/A')}")
        print(f"  Reranker top k: {stats.get('reranker_top_k', 'N/A')}")
        print(f"  Cross-encoder: {stats.get('cross_encoder_model', 'N/A')}")
    
    # Проверка необходимости индексации
    if stats['status'] == 'not initialized':
        print(f"\n⚠️  Векторное хранилище не инициализировано!")
        print(f"   Запустите индексацию:")
        print(f"   - В Telegram боте: /index")
        print(f"   - Или запустите бота (индексация произойдет автоматически)")
    else:
        print(f"\n✅ Векторное хранилище инициализировано")
        if stats['count'] == 0:
            print(f"   ⚠️  Но документов не найдено!")
            print(f"   Проверьте наличие файлов в директории: {config.DATA_DIR}")
        else:
            print(f"   Готово к использованию!")
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())




