"""
Unified RAG Pipeline using the production Vietnamese embeddings
Works with the tech_embeddings collection from chunk_embedding pipeline
"""
from typing import List, Dict, Optional
from langchain_core.documents import Document
from rag.vietnamese_retriever import VietnameseRetriever
from rag.reranker import QwenReranker
from rag.llm_service import GeminiLLMService
from rag.config import TOP_K_RETRIEVE, TOP_N_RERANK, MILVUS_COLLECTION_NAME
from storage.postgres_client import PostgresClient


class UnifiedRAGPipeline:
    """
    RAG Pipeline using production Vietnamese embeddings for retrieval
    """
    
    def __init__(
        self,
        retriever: Optional[VietnameseRetriever] = None,
        reranker: Optional[QwenReranker] = None,
        llm_service: Optional[GeminiLLMService] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize unified RAG pipeline
        
        Args:
            retriever: VietnameseRetriever instance
            reranker: QwenReranker instance
            llm_service: GeminiLLMService instance
            collection_name: Milvus collection name
        """
        print("Initializing Unified RAG Pipeline...")
        
        target_collection = collection_name or MILVUS_COLLECTION_NAME
        self.retriever = retriever or VietnameseRetriever(collection_name=target_collection)
        self.reranker = reranker or QwenReranker()
        self.llm_service = llm_service or GeminiLLMService()
        self.pg_client = PostgresClient()
        self.conversation_history: List[Dict[str, str]] = []
        
        print("Unified RAG Pipeline initialized")
    
    def _enrich_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Enrich documents with full product metadata from PostgreSQL
        
        Args:
            documents: Documents with record_id in metadata
            
        Returns:
            Documents with enriched metadata
        """
        if not documents:
            return documents
        
        # Get unique record IDs
        record_ids = list(set([doc.metadata.get("record_id") for doc in documents if doc.metadata.get("record_id")]))
        
        if not record_ids:
            return documents
        
        # Check if PostgreSQL is connected
        if not self.pg_client.cur:
            # PostgreSQL not connected, return documents as-is
            return documents
        
        # Fetch full product info
        sql = """
            SELECT id, source_url, full_title, category, category_eng,
                   brand, model, price, content_type
            FROM products
            WHERE id = ANY(%s)
        """
        
        try:
            self.pg_client.cur.execute(sql, (record_ids,))
            rows = self.pg_client.cur.fetchall()
            
            # Create lookup dict
            product_info = {}
            for row in rows:
                product_info[row[0]] = {
                    "id": row[0],
                    "source_url": row[1],
                    "title": row[2],
                    "category": row[3] or row[4],  # Vietnamese or English
                    "brand": row[5],
                    "model": row[6],
                    "price": row[7],
                    "content_type": row[8]
                }
            
            # Enrich documents
            enriched = []
            for doc in documents:
                record_id = doc.metadata.get("record_id")
                if record_id and record_id in product_info:
                    # Merge metadata
                    enriched_doc = Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, **product_info[record_id]}
                    )
                    enriched.append(enriched_doc)
                else:
                    enriched.append(doc)
            
            return enriched
            
        except Exception as e:
            print(f"Warning: Could not enrich metadata: {e}")
            return documents
    
    def query(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        top_n: int = TOP_N_RERANK,
        use_history: bool = True,
        enrich_metadata: bool = True
    ) -> Dict:
        """
        Process a query through the RAG pipeline
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            top_n: Number of documents after reranking
            use_history: Whether to use conversation history
            enrich_metadata: Whether to fetch full product metadata
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Step 1: Retrieve documents using Vietnamese embeddings
        print(f"\nRetrieving top-{top_k} documents...")
        retrieved_docs = self.retriever.retrieve(query, k=top_k)
        
        if not retrieved_docs:
            return {
                "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan để trả lời câu hỏi của bạn.",
                "sources": [],
                "retrieved_count": 0,
                "reranked_count": 0,
                "query": query
            }
        
        print(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Enrich metadata if requested
        if enrich_metadata:
            print("Enriching metadata from PostgreSQL...")
            retrieved_docs = self._enrich_metadata(retrieved_docs)
        
        # Step 3: Rerank documents
        print(f"Reranking to top-{top_n}...")
        reranked_docs = self.reranker.rerank(query, retrieved_docs, top_n=top_n)
        print(f"Reranked to {len(reranked_docs)} documents")
        
        # Step 4: Format context for LLM
        context = self._format_context(reranked_docs)
        
        # Step 5: Generate response
        print("Generating response with Gemini...")
        history = self.conversation_history if use_history else None
        answer = self.llm_service.generate_response(
            prompt=query,
            context=context,
            conversation_history=history
        )
        
        # Step 6: Update conversation history
        if use_history:
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
        
        # Step 7: Prepare sources
        sources = []
        for doc in reranked_docs:
            sources.append({
                "title": doc.metadata.get("title", "Unknown"),
                "url": doc.metadata.get("source_url", ""),
                "category": doc.metadata.get("category", ""),
                "brand": doc.metadata.get("brand", ""),
                "model": doc.metadata.get("model", ""),
                "price": doc.metadata.get("price", ""),
                "record_id": doc.metadata.get("record_id", ""),
                "score": doc.metadata.get("score", 0),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        print("Response generated\n")
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_count": len(retrieved_docs),
            "reranked_count": len(reranked_docs),
            "query": query
        }
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format documents into context string for LLM
        
        Args:
            documents: List of documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get("title", "Unknown")
            brand = doc.metadata.get("brand", "")
            model = doc.metadata.get("model", "")
            price = doc.metadata.get("price", "")
            url = doc.metadata.get("source_url", "")
            content = doc.page_content
            
            context_part = f"[Nguồn {i}]\n"
            if title:
                context_part += f"Sản phẩm: {title}\n"
            if brand:
                context_part += f"Thương hiệu: {brand}\n"
            if model:
                context_part += f"Model: {model}\n"
            if price:
                context_part += f"Giá: {price:,} VNĐ\n"
            if url:
                context_part += f"Link: {url}\n"
            context_part += f"Nội dung: {content}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def set_history(self, history: List[Dict[str, str]]):
        """Set conversation history"""
        self.conversation_history = history.copy()
    
    def __del__(self):
        """Cleanup PostgreSQL connection"""
        if hasattr(self, 'pg_client'):
            self.pg_client.close()


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = UnifiedRAGPipeline()
    
    # Test queries
    test_queries = [
        "Laptop gaming tốt nhất giá dưới 20 triệu",
        "So sánh iPhone 15 và Samsung S24",
        "Điện thoại có camera tốt nhất"
    ]
    
    for query in test_queries:
        print("\n" + "="*70)
        print(f"QUERY: {query}")
        print("="*70)
        
        result = pipeline.query(query, top_k=10, top_n=3)
        
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source['title']}")
            print(f"   Brand: {source['brand']}, Model: {source['model']}")
            if source['price']:
                print(f"   Price: {source['price']:,} VNĐ")
            print(f"   URL: {source['url']}")
            print(f"   Score: {source['score']:.4f}")
        
        print("\n" + "="*70)

