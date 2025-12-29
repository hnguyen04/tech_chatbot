"""
Test script to verify embedding dimensions and Milvus connection
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunk_embedding.embedding import Embedder
from storage.milvus_client import MilvusRetriever
from rag.config import (
    QWEN_EMBEDDING_MODEL, DEVICE, MILVUS_COLLECTION_NAME,
    MILVUS_EMBEDDING_DIM, ZILLIZ_CLOUD_URI, ZILLIZ_CLOUD_TOKEN
)


def test_embedding_dimensions():
    """Test embedding generation and dimensions"""
    print("\n" + "="*60)
    print("EMBEDDING DIMENSION TEST")
    print("="*60)
    
    print(f"\nModel: {QWEN_EMBEDDING_MODEL}")
    print(f"Device: {DEVICE}")
    print(f"Expected dimension (from config): {MILVUS_EMBEDDING_DIM}")
    
    try:
        print("\nInitializing embeddings service...")
        embeddings = Embedder(model_name=QWEN_EMBEDDING_MODEL, device=DEVICE)
        
        # Test with a sample text
        test_text = "This is a test sentence to check embedding dimensions."
        print(f"\nTest text: '{test_text}'")
        
        print("Generating embedding...")
        embedding = embeddings.embed_query(test_text)
        
        actual_dim = len(embedding)
        print(f"\nSuccessfully generated embedding!")
        print(f"  Actual dimension: {actual_dim}")
        print(f"  Expected dimension: {MILVUS_EMBEDDING_DIM}")
        
        if actual_dim == MILVUS_EMBEDDING_DIM:
            print("  Dimensions match!")
        else:
            print(f"  WARNING: Dimension mismatch!")
            print(f"  Please update MILVUS_EMBEDDING_DIM in config.py or .env to {actual_dim}")
            return False
        
        # Test batch embedding
        test_texts = [
            "First test document",
            "Second test document",
            "Third test document"
        ]
        
        print(f"\nTesting batch embedding with {len(test_texts)} documents...")
        batch_embeddings = embeddings.embed_documents(test_texts)
        
        print(f"Generated {len(batch_embeddings)} embeddings")
        print(f"  All dimensions: {[len(e) for e in batch_embeddings]}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during embedding test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_milvus_connection():
    """Test Milvus/Zilliz Cloud connection"""
    print("\n" + "="*60)
    print("MILVUS CONNECTION TEST")
    print("="*60)
    
    print(f"\nCollection name: {MILVUS_COLLECTION_NAME}")
    print(f"Dimension: {MILVUS_EMBEDDING_DIM}")
    print(f"URI: {ZILLIZ_CLOUD_URI if ZILLIZ_CLOUD_URI else 'Not set'}")
    print(f"Token: {'*' * 20 if ZILLIZ_CLOUD_TOKEN else 'Not set'}")
    
    if not ZILLIZ_CLOUD_URI or not ZILLIZ_CLOUD_TOKEN:
        print("\n✗ Missing Zilliz Cloud credentials!")
        print("Please set the following in your .env file:")
        print("  ZILLIZ_CLOUD_URI=<your-zilliz-cloud-uri>")
        print("  ZILLIZ_CLOUD_TOKEN=<your-api-token>")
        return False
    
    try:
        print("\nConnecting to Zilliz Cloud...")
        retriever = MilvusRetriever(
            collection_name=MILVUS_COLLECTION_NAME,
            dimension=MILVUS_EMBEDDING_DIM
        )
        
        print("\nConnection successful!")
        
        # Get collection stats
        print("\nFetching collection statistics...")
        stats = retriever.get_collection_stats()
        
        if stats:
            print("\nCollection Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during Milvus connection test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MILVUS/ZILLIZ SETUP VERIFICATION")
    print("="*60)
    
    # Test 1: Embedding dimensions
    embedding_test_passed = test_embedding_dimensions()
    
    # Test 2: Milvus connection
    milvus_test_passed = test_milvus_connection()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Embedding dimensions test: {'PASSED' if embedding_test_passed else 'FAILED'}")
    print(f"Milvus connection test: {'PASSED' if milvus_test_passed else 'FAILED'}")
    
    if embedding_test_passed and milvus_test_passed:
        print("\nAll tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Run data ingestion: python -m rag.data_ingestion --vector-store milvus")
        print("2. Test RAG pipeline: python -m rag.test_pipeline")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


