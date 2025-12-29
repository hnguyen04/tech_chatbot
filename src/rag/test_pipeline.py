"""
Smoke test script for RAG pipeline
Tests embedding, retrieval, and LLM generation end-to-end
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.unified_pipeline import UnifiedRAGPipeline


def test_pipeline():
    """Test the RAG pipeline end-to-end"""
    print("Testing Unified RAG Pipeline...")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n1. Initializing Unified RAG Pipeline...")
    try:
        pipeline = UnifiedRAGPipeline()
        print("Pipeline initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check collection stats
    print("\n2. Checking Milvus collection...")
    try:
        stats = pipeline.retriever.get_collection_stats()
        print(f"Collection: {stats.get('collection_name')}")
        print(f"  Documents: {stats.get('num_entities', 0)}")
    except Exception as e:
        print(f"✗ Failed to get collection stats: {e}")
        print("   Please run data ingestion first:")
        print("   python src/chunk_embedding.py")
        return False
    
    # Test query
    test_query = "What are the features of Honor X7d?"
    print(f"\n3. Testing query: '{test_query}'")
    
    try:
        result = pipeline.query(
            query=test_query,
            top_k=5,
            top_n=3,
            use_history=False
        )
        
        print(f"Query processed successfully")
        print(f"\nResults:")
        print(f"   - Retrieved: {result.get('retrieved_count', 0)} documents")
        print(f"   - Reranked: {result.get('reranked_count', 0)} documents")
        print(f"   - Answer length: {len(result.get('answer', ''))} characters")
        print(f"   - Sources: {len(result.get('sources', []))}")
        
        if result.get('answer'):
            print(f"\nAnswer preview:")
            answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
            print(f"   {answer_preview}")
        
        if result.get('sources'):
            print(f"\nSources:")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"   {i}. {source.get('title', 'Unknown')[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"Query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("RAG Pipeline Smoke Test")
    print("=" * 60)
    
    success = test_pipeline()
    
    print("\n" + "=" * 60)
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Tests failed!")
        sys.exit(1)