"""
Simple test script for chatbot using existing Vietnamese embeddings
Uses the tech_embeddings collection in Milvus
"""
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from rag.unified_pipeline import UnifiedRAGPipeline
from rag.config import TOP_K_RETRIEVE, TOP_N_RERANK

def main():
    print("="*70)
    print("TESTING CHATBOT WITH EXISTING VIETNAMESE EMBEDDINGS")
    print("="*70)
    print("\nThis script uses:")
    print("  - Collection: tech_embeddings (existing in Milvus)")
    print("  - Embedding model: dangvantuan/vietnamese-embedding")
    print("  - Reranker: Qwen3-Reranker-0.6B")
    print("  - LLM: Gemini")
    print("\n" + "="*70 + "\n")
    
    try:
        # Initialize pipeline with existing Vietnamese embeddings
        print("Initializing pipeline...")
        pipeline = UnifiedRAGPipeline(collection_name="tech_embeddings")
        
        # Get collection stats
        stats = pipeline.retriever.get_collection_stats()
        print(f"\nCollection Stats:")
        print(f"  Name: {stats['collection_name']}")
        print(f"  Entities: {stats['num_entities']:,}")
        print(f"  Loaded: {stats['loaded']}")
        
        # Interactive query loop
        print("\n" + "="*70)
        print("CHATBOT READY - Type your questions (or 'quit' to exit)")
        print("="*70 + "\n")
        
        while True:
            query = input("\nYou: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            print("\n[Thinking...]")
            
            try:
                result = pipeline.query(
                    query=query,
                    top_k=TOP_K_RETRIEVE,
                    top_n=TOP_N_RERANK,
                    use_history=True,
                    enrich_metadata=True
                )
                
                print(f"\nBot: {result['answer']}")
                print(f"\n[Retrieved: {result['retrieved_count']} | Reranked: {result['reranked_count']}]")
                
                if result['sources']:
                    print(f"\nSources ({len(result['sources'])}):")
                    for i, source in enumerate(result['sources'][:3], 1):  # Show top 3
                        print(f"\n  {i}. {source.get('title', 'Unknown')}")
                        if source.get('brand'):
                            print(f"     Brand: {source['brand']}")
                        if source.get('model'):
                            print(f"     Model: {source['model']}")
                        if source.get('price'):
                            print(f"     Price: {source['price']:,} VNĐ")
                        if source.get('url'):
                            print(f"     URL: {source['url']}")
                        print(f"     Score: {source.get('score', 0):.4f}")
                
            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Check your .env file has correct credentials:")
        print("   - ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_TOKEN")
        print("   - POSTGRES_DB_* (for metadata enrichment)")
        print("   - GEMINI_API_KEY")
        print("2. Verify the 'tech_embeddings' collection exists in Milvus")
        print("3. Check that the collection has data (num_entities > 0)")


if __name__ == "__main__":
    main()

