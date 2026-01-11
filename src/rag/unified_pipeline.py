"""
Unified RAG Pipeline using the production Vietnamese embeddings
Works with the tech_embeddings collection from chunk_embedding pipeline
"""
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Optional, Set
from langchain_core.documents import Document
from rag.vietnamese_retriever import VietnameseRetriever
from rag.reranker import QwenReranker
from rag.llm_service import GeminiLLMService
from rag.config import TOP_K_RETRIEVE, TOP_N_RERANK, MILVUS_COLLECTION_NAME, ENABLE_RERANKING
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
        
        # Only load reranker if enabled (saves memory and startup time)
        if ENABLE_RERANKING:
            self.reranker = reranker or QwenReranker()
        else:
            self.reranker = None
            print("Reranking disabled (ENABLE_RERANKING=false)")
        
        self.llm_service = llm_service or GeminiLLMService()
        self.pg_client = PostgresClient()
        self.conversation_history: List[Dict[str, str]] = []
        
        print("Unified RAG Pipeline initialized")
    
    def _is_simple_query(self, query: str) -> bool:
        """
        Detect if a query is simple (single product lookup) vs complex (needs decomposition).
        
        Simple: "iPhone 15", "Galaxy S24 Ultra specs"
        Complex: "so sánh iPhone vs Samsung", "điện thoại tốt nhất dưới 10 triệu"
        
        Returns:
            True if simple lookup, False if complex (needs decomposition)
        """
        query_lower = query.lower().strip()
        
        # COMPLEX: Comparison indicators (mentions 2+ products)
        comparison_keywords = ["so sánh", "compare", " vs ", " với ", " và ", " hay ", " or "]
        for kw in comparison_keywords:
            if kw in query_lower:
                return False
        
        # COMPLEX: Recommendation/ranking queries
        recommendation_keywords = [
            "nên mua", "nên chọn", "recommend", "gợi ý",
            "top ", "best", "tốt nhất", "rẻ nhất", "đáng mua",
            "tư vấn", "chọn gì", "mua gì"
        ]
        for kw in recommendation_keywords:
            if kw in query_lower:
                return False
        
        # COMPLEX: Price/constraint queries
        constraint_keywords = ["dưới", "under", "trên", "above", "từ", "đến", "triệu", "nghìn", "k "]
        for kw in constraint_keywords:
            if kw in query_lower:
                return False
        
        # COMPLEX: Analysis queries  
        analysis_keywords = ["ưu điểm", "nhược điểm", "pros", "cons", "đánh giá", "review"]
        for kw in analysis_keywords:
            if kw in query_lower:
                return False
        
        # SIMPLE: Everything else is likely a product lookup
        # e.g., "iPhone 15 Pro Max", "Samsung Galaxy S24 Ultra", "thông số Macbook"
        return True
    
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
            # Debug: Check record IDs
            print(f"Enriching {len(record_ids)} records. IDs (sample): {record_ids[:10]}")
            
            self.pg_client.cur.execute(sql, (record_ids,))
            rows = self.pg_client.cur.fetchall()
            
            # Create lookup dict
            product_info = {}
            for row in rows:
                product_info[row['id']] = {
                    "id": row['id'],
                    "source_url": row['source_url'],
                    "title": row['full_title'],
                    "category": row['category'] or row['category_eng'],  # Vietnamese or English
                    "brand": row['brand'],
                    "model": row['model'],
                    "price": row['price'],
                    "content_type": row['content_type'],
                    "content_text": row['content_text'],
                    "images": row['images'] if row['images'] else []
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
            import traceback
            traceback.print_exc()
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

    def _parallel_hybrid_search(self, sub_queries: List[str], top_k: int) -> List[Document]:
        """
        Thread-safe parallel hybrid search.
        
        Safety: Each thread works independently on its own query.
        Aggregation and deduplication happen AFTER all futures complete
        to avoid race conditions on shared state.
        
        Args:
            sub_queries: List of search queries
            top_k: Number of documents to retrieve per query
            
        Returns:
            Deduplicated list of documents
        """
        # Single query - no parallelization needed
        if len(sub_queries) == 1:
            return self._hybrid_search(sub_queries[0], top_k)
        
        all_results = []  # Collect all results first (no shared state during parallel execution)
        
        with ThreadPoolExecutor(max_workers=min(3, len(sub_queries))) as executor:
            # Submit all search tasks
            future_to_query = {
                executor.submit(self._hybrid_search, q, top_k): q 
                for q in sub_queries
            }
            
            # Wait for ALL futures to complete, then collect results
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    docs = future.result()
                    all_results.extend(docs)  # Safe: single-threaded aggregation
                except Exception as e:
                    print(f"Search error for query '{query}': {e}")
                    continue
        
        # Deduplicate in single thread (no race condition)
        seen_ids: Set[str] = set()
        unique_docs = []
        for doc in all_results:
            rid = doc.metadata.get("record_id")
            if rid:
                rid_str = str(rid)
                if rid_str not in seen_ids:
                    seen_ids.add(rid_str)
                    unique_docs.append(doc)
            else:
                # Docs without ID - include them (rare case)
                unique_docs.append(doc)
        
        return unique_docs

    def _extract_relevant_indices(self, text: str) -> List[int]:
        """
        Extract relevant source indices from LLM response
        Expected format: RELEVANT_SOURCE_INDICES: [1, 2, 5]
        """
        indices = []
        try:
            match = re.search(r"RELEVANT_SOURCE_INDICES:\s*\[(.*?)\]", text)
            if match:
                content = match.group(1)
                # Split by comma and convert to int
                parts = [p.strip() for p in content.split(',') if p.strip()]
                for p in parts:
                    if p.isdigit():
                        indices.append(int(p))
        except Exception as e:
            print(f"Error parsing indices: {e}")
        return indices

    def query(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        top_n: int = TOP_N_RERANK,
        use_history: bool = True,
        enrich_metadata: bool = True,
        skip_decomposition: Optional[bool] = None
    ) -> Dict:
        """
        Process a query through the RAG pipeline with Hybrid Search and Query Decomposition
        
        Args:
            query: User query
            top_k: Number of documents to retrieve per sub-query
            top_n: Number of documents after reranking
            use_history: Whether to use conversation history
            enrich_metadata: Whether to fetch full product metadata
            skip_decomposition: If True, skip decomposition. If None, auto-detect based on query complexity.
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Step 1: Query Decomposition (auto-detect if not specified)
        print(f"\nProcessing Query: {query}")
        
        # Auto-detect query complexity if not explicitly set
        if skip_decomposition is None:
            skip_decomposition = self._is_simple_query(query)
        
        if skip_decomposition:
            sub_queries = [query]
            print("Simple query detected - skipping decomposition")
        else:
            sub_queries = self.llm_service.generate_search_queries(query)
            print(f"Generated sub-queries: {sub_queries}")
        
        # Step 2: Parallel Hybrid Search & Aggregation (thread-safe)
        all_docs = self._parallel_hybrid_search(sub_queries, top_k)
        
        print(f"Aggregated {len(all_docs)} unique documents from hybrid search")
        
        if not all_docs:
            return {
                "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan để trả lời câu hỏi của bạn.",
                "sources": [],
                "retrieved_count": 0,
                "reranked_count": 0,
                "query": query
            }
        
        # Step 3: Pre-filter to reasonable size, then enrich
        # Limit to top 20 candidates max to balance speed vs coverage
        max_candidates = min(len(all_docs), 20)
        candidates = all_docs[:max_candidates]
        
        if enrich_metadata:
            print(f"Enriching {len(candidates)} documents...")
            candidates = self._enrich_metadata(candidates)
        
        # Step 4: Rerank or take top-N
        if ENABLE_RERANKING and self.reranker:
            print(f"Reranking to top-{top_n}...")
            reranked_docs = self.reranker.rerank(query, candidates, top_n=top_n)
            print(f"Reranked to {len(reranked_docs)} documents")
        else:
            print(f"Taking top-{top_n} by score...")
            reranked_docs = candidates[:top_n]
        
        # Step 5: Format context for LLM
        context = self._format_context(reranked_docs)
        
        # Step 6: Generate response
        print("Generating response with Gemini (CoT)...")
        history = self.conversation_history if use_history else None
        current_date = datetime.now().strftime("%Y-%m-%d")
        full_answer = self.llm_service.generate_response(
            prompt=query,
            context=context,
            conversation_history=history,
            current_date=current_date
        )

        # Parse relevant indices
        relevant_indices = self._extract_relevant_indices(full_answer)
        print(f"LLM identified relevant sources: {relevant_indices}")
        
        # Clean answer (remove the metadata line)
        answer = re.sub(r"RELEVANT_SOURCE_INDICES:.*", "", full_answer).strip()
        
        # Step 7: Update conversation history
        if use_history:
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
        
        # Step 8: Prepare sources
        # Filter sources based on LLM selection
        if relevant_indices:
            final_docs_for_sources = []
            for idx in relevant_indices:
                zero_idx = idx - 1 # Convert 1-based [Nguồn X] to 0-based list index
                if 0 <= zero_idx < len(reranked_docs):
                    final_docs_for_sources.append(reranked_docs[zero_idx])
            
            # Fallback if filtering failed (e.g. invalid indices)
            if not final_docs_for_sources:
                 print("Warning: LLM returned indices but none matched. Falling back to all docs.")
                 final_docs_for_sources = reranked_docs
        else:
            # Fallback if no indices returned
            final_docs_for_sources = reranked_docs

        sources = []
        for doc in final_docs_for_sources:
            sources.append({
                "title": doc.metadata.get("title", "Unknown"),
                "url": doc.metadata.get("source_url", ""),
                "category": doc.metadata.get("category", ""),
                "content_type": doc.metadata.get("content_type", "unknown"),
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
                try:
                    context_part += f"Giá: {int(price):,} VNĐ\n"
                except:
                    context_part += f"Giá: {price} VNĐ\n"
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
    
    def query_for_eval(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        top_n: int = TOP_N_RERANK,
        skip_decomposition: bool = False
    ) -> Dict:
        """
        Process a query and return additional data needed for evaluation.
        
        This method returns more detailed information than query() for
        evaluation purposes, including retrieved IDs and context.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve per sub-query
            top_n: Number of documents after reranking
            skip_decomposition: If True, skip query decomposition
            
        Returns:
            Dictionary with answer, sources, retrieved_ids, context, etc.
        """
        # Step 1: Query Decomposition
        if skip_decomposition:
            sub_queries = [query]
        else:
            sub_queries = self.llm_service.generate_search_queries(query)
        
        # Step 2: Hybrid Search
        all_docs = self._parallel_hybrid_search(sub_queries, top_k)
        
        # Track retrieved IDs before enrichment
        retrieved_ids = []
        for doc in all_docs:
            rid = doc.metadata.get("record_id")
            if rid:
                retrieved_ids.append(int(rid) if isinstance(rid, str) else rid)
        
        if not all_docs:
            return {
                "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan.",
                "sources": [],
                "retrieved_ids": [],
                "reranked_ids": [],
                "context": "",
                "sub_queries": sub_queries,
                "query": query
            }
        
        # Step 3: Pre-filter and enrich (limit to 20 for speed)
        max_candidates = min(len(all_docs), 20)
        candidates = all_docs[:max_candidates]
        candidates = self._enrich_metadata(candidates)
        
        # Step 4: Rerank or take top-N
        if ENABLE_RERANKING and self.reranker:
            reranked_docs = self.reranker.rerank(query, candidates, top_n=top_n)
        else:
            reranked_docs = candidates[:top_n]
        
        # Track reranked IDs
        reranked_ids = []
        for doc in reranked_docs:
            rid = doc.metadata.get("record_id")
            if rid:
                reranked_ids.append(int(rid) if isinstance(rid, str) else rid)
        
        # Step 5: Format context
        context = self._format_context(reranked_docs)
        
        # Step 6: Generate response
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        full_answer = self.llm_service.generate_response(
            prompt=query,
            context=context,
            conversation_history=None,  # No history for eval
            current_date=current_date
        )
        
        # Parse relevant indices
        relevant_indices = self._extract_relevant_indices(full_answer)
        
        # Clean answer
        import re
        answer = re.sub(r"RELEVANT_SOURCE_INDICES:.*", "", full_answer).strip()
        
        # Prepare sources
        sources = []
        for doc in reranked_docs:
            sources.append({
                "title": doc.metadata.get("title", "Unknown"),
                "url": doc.metadata.get("source_url", ""),
                "record_id": doc.metadata.get("record_id", ""),
                "score": doc.metadata.get("score", 0),
                "brand": doc.metadata.get("brand", ""),
                "model": doc.metadata.get("model", ""),
                "price": doc.metadata.get("price", ""),
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_ids": retrieved_ids,
            "reranked_ids": reranked_ids,
            "context": context,
            "sub_queries": sub_queries,
            "query": query,
            "retrieved_count": len(all_docs),
            "reranked_count": len(reranked_docs)
        }
    
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
            "giới thiệu cho tôi 10 mẫu điện thoại chơi genshin mượt giá rẻ"
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
                    try:
                        print(f"   Price: {int(source['price']):,} VNĐ")
                    except:
                        print(f"   Price: {source['price']} VNĐ")
                print(f"   URL: {source['url']}")
                print(f"   Score: {source['score']:.4f}")
                print(f"   Images: {len(source.get('images', []))}")
            
            print("\n" + "="*70)
            
    except Exception as e:
        print("\n!!! CRITICAL ERROR !!!")
        print(traceback.format_exc())

