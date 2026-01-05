"""
Script to connect to Milvus/Zilliz Cloud and create embeddings using Qwen 3 embedding model

This script helps you:
1. Connect to Milvus database on Zilliz Cloud
2. Generate embeddings using Qwen 3 embedding model
3. Store embeddings in Milvus collection

Usage:
    python -m rag.create_embeddings
"""
import os
import sys
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from chunk_embedding.embedding import Embedder
from rag.config import (
    QWEN_EMBEDDING_MODEL, DEVICE, MILVUS_COLLECTION_NAME,
    ZILLIZ_CLOUD_URI, ZILLIZ_CLOUD_TOKEN, CHUNK_SIZE, CHUNK_OVERLAP,
    QWEN_EMBEDDING_INSTRUCTION, QWEN_EMBEDDING_DIM
)
from storage.milvus_client import MilvusRetriever

load_dotenv()


def connect_to_milvus(
    collection_name: Optional[str] = None,
    dimension: Optional[int] = None
) -> MilvusRetriever:
    """
    Connect to Milvus/Zilliz Cloud database
    
    Args:
        collection_name: Name of the collection (uses config default if None)
        dimension: Embedding dimension (auto-detected if None)
        
    Returns:
        MilvusRetriever instance
    """
    print("\n=== Connecting to Milvus/Zilliz Cloud ===")
    
    # Check credentials
    uri = ZILLIZ_CLOUD_URI
    token = ZILLIZ_CLOUD_TOKEN
    
    if not uri or not token:
        raise ValueError(
            "Missing Zilliz Cloud credentials!\n"
            "Please set the following environment variables:\n"
            "  - ZILLIZ_CLOUD_URI (or ZILLIZIO_HOST)\n"
            "  - ZILLIZ_CLOUD_TOKEN (or ZILLIZIO_API_KEY)\n"
            "\nYou can set these in your .env file."
        )
    
    print(f"URI: {uri}")
    print(f"Collection: {collection_name or MILVUS_COLLECTION_NAME}")
    
    # Initialize embeddings to get dimension if not provided
    if dimension is None:
        print("\nInitializing embedding model to detect dimension...")
        try:
            embeddings = Embedder(model_name=QWEN_EMBEDDING_MODEL, device=DEVICE)
            dimension = embeddings.embedding_dim
            print(f"Detected embedding dimension: {dimension}")
        except Exception as e:
            print(f"Warning: Could not auto-detect dimension: {e}")
            print("Using default dimension from config...")
            from rag.config import MILVUS_EMBEDDING_DIM
            dimension = MILVUS_EMBEDDING_DIM
    
    # Connect to Milvus
    retriever = MilvusRetriever(
        collection_name=collection_name or MILVUS_COLLECTION_NAME,
        dimension=dimension,
        uri=uri,
        token=token
    )
    
    print("✓ Successfully connected to Milvus!")
    return retriever


def create_embeddings_from_texts(
    texts: List[str],
    collection_name: Optional[str] = None,
    force_reindex: bool = False
) -> None:
    """
    Create embeddings from a list of texts and store in Milvus
    
    Args:
        texts: List of text strings to embed
        collection_name: Name of the collection
        force_reindex: If True, delete existing collection and recreate
    """
    print("\n=== Creating Embeddings ===")
    print(f"Number of texts: {len(texts)}")
    
    # Initialize embedding model using existing Embedder class
    # Support Qwen3-Embedding features: instruction-aware and custom dimensions
    print(f"\nLoading embedding model: {QWEN_EMBEDDING_MODEL}")
    embedding_dim = int(QWEN_EMBEDDING_DIM) if QWEN_EMBEDDING_DIM else None
    embeddings = Embedder(
        model_name=QWEN_EMBEDDING_MODEL,
        device=DEVICE,
        embedding_dim=embedding_dim,
        instruction=QWEN_EMBEDDING_INSTRUCTION
    )
    dimension = embeddings.embedding_dim
    
    # Connect to Milvus
    retriever = connect_to_milvus(
        collection_name=collection_name,
        dimension=dimension
    )
    
    # Force reindex if requested
    if force_reindex and retriever.collection_exists():
        print(f"\nForce reindex enabled - deleting existing collection...")
        retriever.delete_collection()
        retriever._ensure_collection()
    
    # Create LangChain documents
    documents = [
        Document(page_content=text, metadata={"source": f"text_{i}", "index": i})
        for i, text in enumerate(texts)
    ]
    
    # Generate embeddings in batches
    print(f"\nGenerating embeddings (batch size: 50)...")
    all_embeddings = []
    batch_size = 50
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
        print(f"  Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
    
    # Add to Milvus
    print(f"\nAdding {len(documents)} documents to Milvus...")
    retriever.add_documents(documents, all_embeddings)
    
    # Show stats
    stats = retriever.get_collection_stats()
    print(f"\n✓ Embeddings created successfully!")
    print(f"Collection: {retriever.collection_name}")
    print(f"Total entities: {stats.get('row_count', 'N/A')}")
    print(f"Embedding dimension: {dimension}")


def create_embeddings_from_documents(
    documents: List[Document],
    collection_name: Optional[str] = None,
    force_reindex: bool = False,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> None:
    """
    Create embeddings from LangChain documents and store in Milvus
    
    Args:
        documents: List of LangChain Document objects
        collection_name: Name of the collection
        force_reindex: If True, delete existing collection and recreate
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    print("\n=== Creating Embeddings from Documents ===")
    print(f"Number of documents: {len(documents)}")
    
    # Chunk documents if needed
    if chunk_size > 0:
        print(f"Chunking documents (size: {chunk_size}, overlap: {chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        documents = text_splitter.split_documents(documents)
        print(f"After chunking: {len(documents)} chunks")
    
    # Extract texts
    texts = [doc.page_content for doc in documents]
    
    # Create embeddings
    create_embeddings_from_texts(
        texts=texts,
        collection_name=collection_name,
        force_reindex=force_reindex
    )


def test_connection():
    """
    Test connection to Milvus and embedding model
    """
    print("=== Testing Connection ===")
    
    try:
        # Test embedding model
        print("\n1. Testing embedding model...")
        embedding_dim = int(QWEN_EMBEDDING_DIM) if QWEN_EMBEDDING_DIM else None
        embeddings = Embedder(
            model_name=QWEN_EMBEDDING_MODEL,
            device=DEVICE,
            embedding_dim=embedding_dim,
            instruction=QWEN_EMBEDDING_INSTRUCTION
        )
        test_emb = embeddings.embed_query("Test query")
        print(f"   ✓ Embedding model works (dimension: {len(test_emb)})")
        
        # Test Milvus connection
        print("\n2. Testing Milvus connection...")
        retriever = connect_to_milvus(dimension=len(test_emb))
        print(f"   ✓ Milvus connection successful")
        
        # Test retrieval
        print("\n3. Testing retrieval...")
        results = retriever.retrieve(test_emb, k=1)
        print(f"   ✓ Retrieval works (found {len(results)} results)")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False


def main():
    """
    Main function - example usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Create embeddings using Qwen 3 and store in Milvus")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test connection to Milvus and embedding model"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name (uses config default if not specified)"
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Delete existing collection and recreate"
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        help="Text strings to embed (for testing)"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_connection()
        return
    
    if args.texts:
        # Create embeddings from provided texts
        create_embeddings_from_texts(
            texts=args.texts,
            collection_name=args.collection,
            force_reindex=args.force_reindex
        )
    else:
        print("No action specified. Use --test to test connection or --texts to create embeddings.")
        print("\nExample usage:")
        print("  python -m rag.create_embeddings --test")
        print("  python -m rag.create_embeddings --texts 'Text 1' 'Text 2' --collection my_collection")


if __name__ == "__main__":
    main()

