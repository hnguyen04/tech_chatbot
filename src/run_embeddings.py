"""
Script to run the embedding pipeline:
1. Connect to PostgreSQL
2. Connect to Milvus/Zilliz Cloud
3. Load data from PostgreSQL
4. Compute embeddings using Qwen3-Embedding model
5. Store embeddings in Milvus

Usage:
    python -m run_embeddings
    python -m run_embeddings --batch-size 50 --collection tech_embeddings_qwen3
"""
import os
import sys
import argparse
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunk_embedding.pipeline import EmbeddingPipeline
from chunk_embedding.vector_db_writer import MilvusWriter
from chunk_embedding.embedding import Embedder
from storage.postgres_client import PostgresClient
from storage.milvus_client import MilvusRetriever
from rag.config import (
    QWEN_EMBEDDING_MODEL, DEVICE, MILVUS_COLLECTION_NAME, MILVUS_EMBEDDING_DIM,
    QWEN_EMBEDDING_INSTRUCTION, QWEN_EMBEDDING_DIM
)

load_dotenv()


def test_connections():
    """
    Test connections to PostgreSQL and Milvus
    """
    print("\n" + "="*60)
    print("TESTING CONNECTIONS")
    print("="*60)
    
    # Test PostgreSQL connection
    print("\n1. Testing PostgreSQL connection...")
    try:
        pg_client = PostgresClient()
        pg_client.cur.execute("SELECT 1")
        result = pg_client.cur.fetchone()
        print(f"   ✓ PostgreSQL connected successfully")
        print(f"   Host: {pg_client.host}:{pg_client.port}")
        print(f"   Database: {pg_client.dbname}")
        pg_client.close()
        pg_ok = True
    except Exception as e:
        print(f"   ✗ PostgreSQL connection failed: {e}")
        pg_ok = False
    
    # Test Milvus connection
    print("\n2. Testing Milvus/Zilliz Cloud connection...")
    try:
        retriever = MilvusRetriever(
            collection_name=MILVUS_COLLECTION_NAME,
            dimension=MILVUS_EMBEDDING_DIM
        )
        print(f"   ✓ Milvus connected successfully")
        print(f"   URI: {retriever.uri}")
        print(f"   Collection: {retriever.collection_name}")
        print(f"   Dimension: {retriever.dimension}")
        milvus_ok = True
    except Exception as e:
        print(f"   ✗ Milvus connection failed: {e}")
        milvus_ok = False
    
    # Test embedding model
    print("\n3. Testing embedding model...")
    try:
        embedding_dim = int(QWEN_EMBEDDING_DIM) if QWEN_EMBEDDING_DIM else None
        embedder = Embedder(
            model_name=QWEN_EMBEDDING_MODEL,
            device=DEVICE,
            embedding_dim=embedding_dim,
            instruction=QWEN_EMBEDDING_INSTRUCTION
        )
        test_emb = embedder.embed_query("Test query")
        print(f"   ✓ Embedding model loaded successfully")
        print(f"   Model: {QWEN_EMBEDDING_MODEL}")
        print(f"   Dimension: {len(test_emb)}")
        print(f"   Device: {DEVICE}")
        if QWEN_EMBEDDING_INSTRUCTION:
            print(f"   Instruction: {QWEN_EMBEDDING_INSTRUCTION[:50]}...")
        embedding_ok = True
    except Exception as e:
        print(f"   ✗ Embedding model failed: {e}")
        import traceback
        traceback.print_exc()
        embedding_ok = False
    
    print("\n" + "="*60)
    if pg_ok and milvus_ok and embedding_ok:
        print("✓ All connections successful! Ready to run embeddings.")
        return True
    else:
        print("✗ Some connections failed. Please fix the issues above.")
        return False


def run_embeddings(
    batch_size: int = 50,
    collection_name: str = None,
    start_from_id: int = 0,
    create_collection: bool = True
):
    """
    Run the embedding pipeline to compute and store embeddings
    
    Args:
        batch_size: Number of records to process per batch
        collection_name: Name of Milvus collection (uses config default if None)
        start_from_id: Start processing from this record ID (useful for resuming)
        create_collection: If True, create collection if it doesn't exist
    """
    print("\n" + "="*60)
    print("RUNNING EMBEDDING PIPELINE")
    print("="*60)
    
    collection_name = collection_name or MILVUS_COLLECTION_NAME
    
    # Initialize pipeline
    print(f"\nInitializing embedding pipeline...")
    print(f"  Model: {QWEN_EMBEDDING_MODEL}")
    print(f"  Collection: {collection_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Start from ID: {start_from_id}")
    
    pipeline = EmbeddingPipeline()
    
    # Get embedding dimension from the embedder
    embedding_dim = pipeline.embedder.embedding_dim
    print(f"  Embedding dimension: {embedding_dim}")
    
    # Create or verify Milvus collection
    if create_collection:
        print(f"\nEnsuring Milvus collection '{collection_name}' exists...")
        try:
            pipeline.milvus_writer.create_collection(
                collection_name=collection_name,
                dim=embedding_dim,
                metric_type="COSINE",  # or "L2", "IP"
                index_type="HNSW"
            )
            print(f"✓ Collection '{collection_name}' is ready")
        except Exception as e:
            print(f"Warning: Collection setup issue: {e}")
            print("Continuing anyway...")
    
    # Run batch processing
    print(f"\nStarting batch processing...")
    last_id = start_from_id
    total_processed = 0
    batch_num = 0
    
    try:
        while True:
            batch_num += 1
            print(f"\n--- Batch {batch_num} (starting from ID {last_id}) ---")
            
            embedded_docs, new_last_id = pipeline.run_batch(
                batch_size=batch_size,
                last_id=last_id,
                collection_name=collection_name
            )
            
            if not embedded_docs:
                print(f"\n✓ No more records to process. Finished!")
                break
            
            total_processed += len(embedded_docs)
            print(f"✓ Processed {len(embedded_docs)} document chunks")
            print(f"  Last processed ID: {new_last_id}")
            print(f"  Total processed so far: {total_processed}")
            
            last_id = new_last_id
            
            # Safety check to avoid infinite loops
            if last_id == new_last_id and len(embedded_docs) == 0:
                print("Warning: No progress made, stopping to avoid infinite loop")
                break
                
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user")
        print(f"Last processed ID: {last_id}")
        print(f"Total processed: {total_processed}")
        print(f"To resume, run with --start-from-id {last_id}")
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nLast processed ID: {last_id}")
        print(f"Total processed: {total_processed}")
    
    print("\n" + "="*60)
    print(f"EMBEDDING PIPELINE COMPLETE")
    print("="*60)
    print(f"Total document chunks processed: {total_processed}")
    print(f"Collection: {collection_name}")
    print(f"Last ID: {last_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Run embedding pipeline: PostgreSQL → Qwen3 Embeddings → Milvus"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test connections to PostgreSQL, Milvus, and embedding model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of records to process per batch (default: 50)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=f"Milvus collection name (default: {MILVUS_COLLECTION_NAME})"
    )
    parser.add_argument(
        "--start-from-id",
        type=int,
        default=0,
        help="Start processing from this record ID (useful for resuming, default: 0)"
    )
    parser.add_argument(
        "--no-create-collection",
        action="store_true",
        help="Don't create collection if it doesn't exist"
    )
    
    args = parser.parse_args()
    
    if args.test:
        success = test_connections()
        if not success:
            sys.exit(1)
    else:
        # Run the embedding pipeline
        run_embeddings(
            batch_size=args.batch_size,
            collection_name=args.collection,
            start_from_id=args.start_from_id,
            create_collection=not args.no_create_collection
        )


if __name__ == "__main__":
    main()

