"""
Data ingestion script: Load from PostgreSQL, chunk, embed, and store in Milvus
"""
import os
from typing import List
from langchain_core.documents import Document
from storage.postgres_client import PostgresClient
from storage.milvus_client import MilvusRetriever
from chunk_embedding.embedding import Embedder
from rag.utils import chunk_text, format_document_metadata
from rag.config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    MILVUS_COLLECTION_NAME, MILVUS_EMBEDDING_DIM,
    QWEN_EMBEDDING_MODEL, DEVICE, QWEN_EMBEDDING_INSTRUCTION, QWEN_EMBEDDING_DIM
)


def load_products_from_postgres(chunked_only: bool = False) -> List[dict]:
    """
    Load products from PostgreSQL
    
    Args:
        chunked_only: If True, only load products that are already chunked
        
    Returns:
        List of product dictionaries
    """
    pg_client = PostgresClient()
    
    if chunked_only:
        query = """
        SELECT id, source_url, domain, full_title, category, content_type, 
               content_text, images, chunked
        FROM products
        WHERE chunked = TRUE
        ORDER BY id
        """
    else:
        query = """
        SELECT id, source_url, domain, full_title, category, content_type, 
               content_text, images, chunked
        FROM products
        WHERE content_text IS NOT NULL AND content_text != ''
        ORDER BY id
        """
    
    try:
        pg_client.cur.execute(query)
        rows = pg_client.cur.fetchall()
        
        products = []
        for row in rows:
            products.append({
                "id": row[0],
                "source_url": row[1],
                "domain": row[2],
                "title": row[3],
                "category": row[4],
                "content_type": row[5],
                "content_text": row[6],
                "images": row[7] if row[7] else [],
                "chunked": row[8]
            })
        
        print(f"Loaded {len(products)} products from PostgreSQL")
        return products
    
    except Exception as e:
        print(f"Error loading products: {e}")
        return []
    finally:
        pg_client.close()


def create_documents_from_products(products: List[dict], force_chunk: bool = False) -> List[Document]:
    """
    Create LangChain Document objects from products
    
    Args:
        products: List of product dictionaries
        force_chunk: If True, chunk even if already chunked
        
    Returns:
        List of Document objects
    """
    documents = []
    
    for product in products:
        content_text = product.get("content_text", "")
        if not content_text:
            continue
        
        # Check if needs chunking
        needs_chunking = not product.get("chunked", False) or force_chunk
        
        if needs_chunking:
            # Chunk the text
            chunks = chunk_text(content_text, CHUNK_SIZE, CHUNK_OVERLAP)
            
            if not chunks:
                # If chunking produces no chunks, use original text
                chunks = [content_text]
            
            # Create a document for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                metadata = format_document_metadata(
                    doc_id=product["id"],
                    source_url=product["source_url"],
                    title=product["title"],
                    category=product["category"],
                    content_type=product["content_type"],
                    chunk_index=chunk_idx
                )
                metadata["domain"] = product.get("domain", "")
                metadata["images"] = product.get("images", [])
                
                doc = Document(page_content=chunk, metadata=metadata)
                documents.append(doc)
        else:
            # Use text as-is (already chunked)
            metadata = format_document_metadata(
                doc_id=product["id"],
                source_url=product["source_url"],
                title=product["title"],
                category=product["category"],
                content_type=product["content_type"],
                chunk_index=0
            )
            metadata["domain"] = product.get("domain", "")
            metadata["images"] = product.get("images", [])
            
            doc = Document(page_content=content_text, metadata=metadata)
            documents.append(doc)
    
    print(f"Created {len(documents)} documents from {len(products)} products")
    return documents


def update_chunked_flag(product_ids: List[int]):
    """
    Update chunked flag in PostgreSQL
    
    Args:
        product_ids: List of product IDs to mark as chunked
    """
    if not product_ids:
        return
    
    pg_client = PostgresClient()
    
    try:
        placeholders = ",".join(["%s"] * len(product_ids))
        query = f"UPDATE products SET chunked = TRUE WHERE id IN ({placeholders})"
        pg_client.cur.execute(query, product_ids)
        pg_client.conn.commit()
        print(f"Updated chunked flag for {len(product_ids)} products")
    except Exception as e:
        print(f"Error updating chunked flag: {e}")
        pg_client.conn.rollback()
    finally:
        pg_client.close()


def ingest_data(
    force_reindex: bool = False,
    batch_size: int = 100
):
    """
    Main ingestion function: Load from PostgreSQL, chunk, embed, and store in Milvus
    
    Args:
        force_reindex: If True, reindex even if already chunked
        batch_size: Batch size for processing
    """
    print(f"Starting data ingestion from PostgreSQL to Milvus...")
    
    # Load from PostgreSQL
    products = load_products_from_postgres(chunked_only=False)
    
    if not products:
        print("No documents found in PostgreSQL")
        return
    
    # Filter products that need processing
    if not force_reindex:
        products_to_process = [p for p in products if not p.get("chunked", False)]
    else:
        products_to_process = products
    
    if not products_to_process:
        print("All products are already chunked. Use force_reindex=True to reindex.")
        return
    
    print(f"Processing {len(products_to_process)} documents from PostgreSQL...")
    
    # Create documents
    documents = create_documents_from_products(products_to_process, force_chunk=force_reindex)
    
    if not documents:
        print("No documents created")
        return
    
    # Ingest to Milvus
    _ingest_to_milvus(documents, force_reindex)
    
    # Update chunked flag for processed products
    product_ids = list(set([p["id"] for p in products_to_process]))
    update_chunked_flag(product_ids)
    
    print(f"Data ingestion complete! Indexed {len(documents)} documents from {len(products_to_process)} PostgreSQL records")


def _ingest_to_milvus(documents: List[Document], force_reindex: bool = False):
    """
    Ingest documents to Milvus/Zilliz Cloud
    
    Args:
        documents: List of LangChain Document objects
        force_reindex: If True, delete and recreate collection
    """
    print("\n=== Ingesting to Milvus/Zilliz Cloud ===")
    
    # Initialize Milvus retriever
    milvus_retriever = MilvusRetriever(
        collection_name=MILVUS_COLLECTION_NAME,
        dimension=MILVUS_EMBEDDING_DIM
    )
    
    # If force reindex, delete existing collection
    if force_reindex and milvus_retriever.collection_exists():
        print(f"Force reindex enabled - deleting existing collection '{MILVUS_COLLECTION_NAME}'")
        milvus_retriever.delete_collection()
        # Recreate collection
        milvus_retriever._ensure_collection()
    
    # Initialize embeddings service using existing Embedder class
    # Support Qwen3-Embedding features: instruction-aware and custom dimensions
    print("Initializing embeddings service...")
    embedding_dim = int(QWEN_EMBEDDING_DIM) if QWEN_EMBEDDING_DIM else None
    embeddings_service = Embedder(
        model_name=QWEN_EMBEDDING_MODEL,
        device=DEVICE,
        embedding_dim=embedding_dim,
        instruction=QWEN_EMBEDDING_INSTRUCTION
    )
    
    # Generate embeddings for documents
    print(f"Generating embeddings for {len(documents)} documents...")
    embeddings = []
    batch_size = 50  # Process in smaller batches for memory efficiency
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_texts = [doc.page_content for doc in batch_docs]
        batch_embeddings = embeddings_service.embed_documents(batch_texts)
        embeddings.extend(batch_embeddings)
        print(f"Generated embeddings for batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
    
    # Add documents and embeddings to Milvus
    print(f"\nAdding {len(documents)} documents to Milvus collection '{MILVUS_COLLECTION_NAME}'...")
    milvus_retriever.add_documents(documents, embeddings)
    
    # Show collection stats
    stats = milvus_retriever.get_collection_stats()
    print(f"\nCollection Statistics:")
    print(f"  Collection: {MILVUS_COLLECTION_NAME}")
    print(f"  Total entities: {stats.get('row_count', 'N/A')}")
    print("✓ Milvus ingestion complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest data from PostgreSQL to Milvus")
    parser.add_argument("--force-reindex", action="store_true", 
                       help="Force reindexing even if already chunked")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    ingest_data(
        force_reindex=args.force_reindex,
        batch_size=args.batch_size
    )