"""
Unified RAG Pipeline using the production Vietnamese embeddings
Works with the tech_embeddings collection from chunk_embedding pipeline
"""
from typing import List, Dict, Optional, Set
from langchain_core.documents import Document
from rag.vietnamese_retriever import VietnameseRetriever
from rag.reranker import QwenReranker
from rag.llm_service import GeminiLLMService
from rag.config import TOP_K_RETRIEVE, TOP_N_RERANK, MILVUS_COLLECTION_NAME
from rag.utils import reciprocal_rank_fusion
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
                   brand, model, price, content_type, content_text, images
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
                    "content_type": row[8],
                    "content_text": row[9],
                    "images": row[10] if row[10] else []
                }
            
            # Enrich documents
            enriched = []
            for doc in documents:
                record_id = doc.metadata.get("record_id")
                if record_id:
                    record_id = int(record_id) if isinstance(record_id, str) and record_id.isdigit() else record_id
                    
                if record_id and record_id in product_info:
                    info = product_info[record_id].copy()
                    # Use full content text if available (Parent Document Retrieval)
                    # This provides better context for Reranker and LLM
                    full_content = info.pop("content_text", None)
                    new_content = full_content if full_content else doc.page_content
                    
                    # Merge metadata
                    enriched_doc = Document(
                        page_content=new_content,
                        metadata={**doc.metadata, **info}
                    )
                    enriched.append(enriched_doc)
                else:
                    enriched.append(doc)
            
            return enriched
            
        except Exception as e:
            print(f"Warning: Could not enrich metadata: {e}")
            return documents
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Document]:
        """
        Perform hybrid search (Vector + Keyword) with RRF fusion
        
        Args:
            query: Search query
            top_k: Number of documents to return
            
        Returns:
            List of fused documents
        """
        # 1. Vector Search (Milvus)
        # print(f"  - Vector search for '{query}'")
        vector_docs = self.retriever.retrieve(query, k=top_k)
        
        # 2. Keyword Search (Postgres)
        keyword_results = []
        if self.pg_client.cur:
            # print(f"  - Keyword search for '{query}'")
            keyword_results = self.pg_client.search_keyword(query, limit=top_k)
        
        # 3. RRF Fusion
        # Use record_id (Milvus) and id (Postgres) as identifier
        vector_ids = [str(doc.metadata.get("record_id")) for doc in vector_docs if doc.metadata.get("record_id")]
        keyword_ids = [str(item["id"]) for item in keyword_results]
        
        if not vector_ids and not keyword_ids:
            return []
            
        fused_scores = reciprocal_rank_fusion([vector_ids, keyword_ids])
        
        # 4. Reconstruct/Merge Documents
        all_docs_map = {}
        
        # Add Milvus docs (prefer these for chunks)
        for doc in vector_docs:
            rid = str(doc.metadata.get("record_id"))
            if rid:
                all_docs_map[rid] = doc
        
        # Add Postgres docs (if not in Milvus or to supplement)
        for item in keyword_results:
            rid = str(item["id"])
            if rid not in all_docs_map:
                # Create new doc from Postgres result
                metadata = {
                    "record_id": item["id"],
                    "title": item["full_title"],
                    "brand": item["brand"],
                    "model": item["model"],
                    "price": item["price"],
                    "source_url": item["source_url"],
                    "category": item["category"],
                    "content_type": "product" # default
                }
                # Use content_text or fallback
                content = item["content_text"] or f"{item['full_title']} {item['brand']} {item['model']}"
                all_docs_map[rid] = Document(page_content=content, metadata=metadata)
        
        # 5. Sort by Fused Score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        top_ids = sorted_ids[:top_k]
        
        final_docs = []
        for rid in top_ids:
            if rid in all_docs_map:
                doc = all_docs_map[rid]
                doc.metadata["score"] = fused_scores[rid]
                final_docs.append(doc)
                
        return final_docs

    def query(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        top_n: int = TOP_N_RERANK,
        use_history: bool = True,
        enrich_metadata: bool = True
    ) -> Dict:
        """
        Process a query through the RAG pipeline with Hybrid Search and Query Decomposition
        
        Args:
            query: User query
            top_k: Number of documents to retrieve per sub-query
            top_n: Number of documents after reranking
            use_history: Whether to use conversation history
            enrich_metadata: Whether to fetch full product metadata
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Step 1: Query Decomposition
        print(f"\nProcessing Query: {query}")
        sub_queries = self.llm_service.generate_search_queries(query)
        print(f"Generated sub-queries: {sub_queries}")
        
        # Step 2: Hybrid Search & Aggregation
        all_docs = []
        seen_ids = set()
        
        for sub_q in sub_queries:
            docs = self._hybrid_search(sub_q, top_k)
            for doc in docs:
                # Use record_id for deduplication
                rid = doc.metadata.get("record_id")
                # If no record_id, use content hash or skip? Milvus/Postgres should have IDs.
                if rid:
                    rid = str(rid)
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        all_docs.append(doc)
                else:
                    # Fallback for docs without ID (unlikely)
                    all_docs.append(doc)
        
        print(f"Aggregated {len(all_docs)} unique documents from hybrid search")
        
        if not all_docs:
            return {
                "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan để trả lời câu hỏi của bạn.",
                "sources": [],
                "retrieved_count": 0,
                "reranked_count": 0,
                "query": query
            }
        
        # Step 3: Enrich metadata if requested
        if enrich_metadata:
            print("Enriching metadata from PostgreSQL...")
            all_docs = self._enrich_metadata(all_docs)
        
        # Step 4: Rerank documents
        # print(f"Reranking to top-{top_n}...")
        # reranked_docs = self.reranker.rerank(query, all_docs, top_n=top_n)
        # print(f"Reranked to {len(reranked_docs)} documents")
        
        # Bypass reranking for faster testing
        reranked_docs = all_docs[:top_n]
        
        # Step 5: Format context for LLM
        context = self._format_context(reranked_docs)
        
        # Step 6: Generate response
        print("Generating response with Gemini (CoT)...")
        history = self.conversation_history if use_history else None
        answer = self.llm_service.generate_response(
            prompt=query,
            context=context,
            conversation_history=history,
            current_date="2026-01-02"
        )
        
        # Step 7: Update conversation history
        if use_history:
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
        
        # Step 8: Prepare sources
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
                "images": doc.metadata.get("images", []),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        print("Response generated\n")
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_count": len(all_docs),
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
    import traceback
    
    try:
        # Initialize pipeline
        pipeline = UnifiedRAGPipeline()
        
        # Test queries
        test_queries = [
            "giới thiệu cho tôi 5 mẫu điện thoại trong tầm giá 20 triệu",
            "Laptop gaming tốt nhất giá dưới 20 triệu",
            "So sánh iPhone 15 và Samsung S24"
        ]
        
        for query in test_queries:
            print("\n" + "="*70)
            print(f"QUERY: {query}")
            print("="*70)
            
            result = pipeline.query(query, top_k=10, top_n=5)
            
            print(f"\nANSWER:\n{result['answer']}")
            print(f"\nSOURCES ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n{i}. {source['title']}")
                print(f"   Brand: {source['brand']}, Model: {source['model']}")
                if source['price']:
                    print(f"   Price: {source['price']:,} VNĐ")
                print(f"   URL: {source['url']}")
                print(f"   Score: {source['score']:.4f}")
                print(f"   Images: {len(source.get('images', []))}")
            
            print("\n" + "="*70)
            
    except Exception as e:
        print("\n!!! CRITICAL ERROR !!!")
        print(traceback.format_exc())

