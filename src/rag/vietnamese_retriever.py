"""
Retriever that uses the existing tech_embeddings collection with Vietnamese embeddings
Compatible with the chunk_embedding pipeline
"""
import os
from typing import List, Optional
from dotenv import load_dotenv
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

load_dotenv()


class VietnameseRetriever:
    """
    Retriever for querying the tech_embeddings collection
    Uses the same Vietnamese embedding model as the ingestion pipeline
    """
    
    def __init__(
        self,
        collection_name: str = "tech_embeddings",
        model_name: str = "dangvantuan/vietnamese-embedding",
        alias: str = "zilliz_query"
    ):
        """
        Initialize Vietnamese retriever
        
        Args:
            collection_name: Name of the Milvus collection
            model_name: SentenceTransformer model name
            alias: Milvus connection alias
        """
        self.collection_name = collection_name
        self.alias = alias
        
        # Load embedding model (same as ingestion)
        print(f"Loading Vietnamese embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        
        # Connect to Milvus
        self._connect_milvus()
        
        # Load collection
        self.collection = Collection(name=collection_name, using=self.alias)
        self.collection.load()
        print(f"Retriever ready with collection '{collection_name}'")
    
    def _connect_milvus(self):
        """Connect to Zilliz Cloud"""
        host = os.getenv("ZILLIZIO_HOST") or os.getenv("ZILLIZ_CLOUD_URI")
        api_key = os.getenv("ZILLIZIO_API_KEY") or os.getenv("ZILLIZ_CLOUD_TOKEN")
        
        if not host or not api_key:
            raise ValueError(
                "Missing Zilliz credentials. Set ZILLIZIO_HOST and ZILLIZIO_API_KEY "
                "(or ZILLIZ_CLOUD_URI and ZILLIZ_CLOUD_TOKEN) in .env"
            )
        
        try:
            connections.connect(
                alias=self.alias,
                uri=host,
                token=api_key
            )
            print(f"Connected to Zilliz Cloud at {host}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Zilliz: {e}")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query using Vietnamese model
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        embedding = self.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
        return embedding[0].tolist()
    
    def retrieve(
        self,
        query: str,
        k: int = 20,
        filter_expr: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve top-k documents for a query
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            filter_expr: Optional filter (e.g., "record_id > 100")
            
        Returns:
            List of LangChain Document objects
        """
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"ef": 64}  # HNSW parameter
        }
        
        # Perform search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["record_id", "chunk_index", "text"],
            expr=filter_expr
        )
        
        # Convert to LangChain Documents
        documents = []
        for hits in results:
            for hit in hits:
                doc = Document(
                    page_content=hit.entity.get("text", ""),
                    metadata={
                        "record_id": hit.entity.get("record_id"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "distance": hit.distance,
                        "score": 1.0 / (1.0 + hit.distance)  # Convert distance to similarity
                    }
                )
                documents.append(doc)
        
        return documents
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 20,
        filter_expr: Optional[str] = None
    ) -> List[tuple[Document, float]]:
        """
        Retrieve top-k documents with scores
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            filter_expr: Optional filter expression
            
        Returns:
            List of (Document, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"ef": 64}
        }
        
        # Perform search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["record_id", "chunk_index", "text"],
            expr=filter_expr
        )
        
        # Convert to LangChain Documents with scores
        documents_with_scores = []
        for hits in results:
            for hit in hits:
                doc = Document(
                    page_content=hit.entity.get("text", ""),
                    metadata={
                        "record_id": hit.entity.get("record_id"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "distance": hit.distance
                    }
                )
                # Convert L2 distance to similarity score
                score = 1.0 / (1.0 + hit.distance)
                documents_with_scores.append((doc, score))
        
        return documents_with_scores
    
    def get_collection_stats(self):
        """Get collection statistics"""
        stats = {
            "collection_name": self.collection_name,
            "num_entities": self.collection.num_entities,
            "loaded": True
        }
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize retriever
    retriever = VietnameseRetriever()
    
    # Get stats
    stats = retriever.get_collection_stats()
    print(f"\nCollection Stats: {stats}")
    
    # Test query
    query = "laptop gaming tốt nhất"
    print(f"\nQuerying: '{query}'")
    
    results = retriever.retrieve(query, k=5)
    print(f"\nFound {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. [Record {doc.metadata['record_id']}, Chunk {doc.metadata['chunk_index']}]")
        print(f"   Score: {doc.metadata.get('score', 'N/A'):.4f}")
        print(f"   Text: {doc.page_content[:200]}...")

