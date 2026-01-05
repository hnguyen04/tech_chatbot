"""
RAG Pipeline orchestrating retrieval → reranking → generation
"""
from typing import List, Dict, Optional
from langchain_core.documents import Document
from storage.milvus_client import MilvusRetriever
from chunk_embedding.embedding import Embedder
from rag.reranker import QwenReranker
from rag.llm_service import GeminiLLMService
from rag.config import (
    TOP_K_RETRIEVE, TOP_N_RERANK,
    QWEN_EMBEDDING_MODEL, DEVICE, MILVUS_COLLECTION_NAME, MILVUS_EMBEDDING_DIM,
    QWEN_EMBEDDING_INSTRUCTION, QWEN_EMBEDDING_DIM
)


class RAGPipeline:
    """
    Main RAG pipeline orchestrating the full flow
    """
    
    def __init__(
        self,
        retriever: Optional[MilvusRetriever] = None,
        reranker: Optional[QwenReranker] = None,
        llm_service: Optional[GeminiLLMService] = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            retriever: MilvusRetriever instance (uses default if None)
            reranker: QwenReranker instance
            llm_service: GeminiLLMService instance
        """
        # Initialize retriever
        if retriever is None:
            print(f"Initializing Milvus retriever with collection '{MILVUS_COLLECTION_NAME}'...")
            self.retriever = MilvusRetriever(
                collection_name=MILVUS_COLLECTION_NAME,
                dimension=MILVUS_EMBEDDING_DIM
            )
        else:
            self.retriever = retriever
        
        # Initialize embeddings using existing Embedder class
        # Support Qwen3-Embedding features: instruction-aware and custom dimensions
        embedding_dim = int(QWEN_EMBEDDING_DIM) if QWEN_EMBEDDING_DIM else None
        self.embeddings = Embedder(
            model_name=QWEN_EMBEDDING_MODEL,
            device=DEVICE,
            embedding_dim=embedding_dim,
            instruction=QWEN_EMBEDDING_INSTRUCTION
        )
        
        self.reranker = reranker or QwenReranker()
        self.llm_service = llm_service or GeminiLLMService()
        self.conversation_history: List[Dict[str, str]] = []
        print(f"✓ RAG Pipeline initialized with Milvus vector store")
    
    def query(self, query: str, top_k: int = TOP_K_RETRIEVE, 
              top_n: int = TOP_N_RERANK, use_history: bool = True) -> Dict:
        """
        Process a query through the RAG pipeline
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            top_n: Number of documents after reranking
            use_history: Whether to use conversation history
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Step 1: Retrieve documents
        # Embed the query first
        query_embedding = self.embeddings.embed_query(query)
        retrieved_docs = self.retriever.retrieve(query_embedding, k=top_k)
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "sources": [],
                "retrieved_count": 0,
                "reranked_count": 0
            }
        
        # Step 2: Rerank documents
        reranked_docs = self.reranker.rerank(query, retrieved_docs, top_n=top_n)
        
        # Step 3: Format context for LLM
        context_docs = []
        for doc in reranked_docs:
            context_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Format context string
        context = self._format_context(context_docs)
        
        # Step 4: Generate response
        history = self.conversation_history if use_history else None
        answer = self.llm_service.generate_response(
            prompt=query,
            context=context,
            conversation_history=history
        )
        
        # Step 5: Update conversation history
        if use_history:
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
        
        # Prepare sources
        sources = []
        for doc in reranked_docs:
            sources.append({
                "title": doc.metadata.get("title", "Unknown"),
                "url": doc.metadata.get("source_url", ""),
                "category": doc.metadata.get("category", ""),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_count": len(retrieved_docs),
            "reranked_count": len(reranked_docs),
            "query": query
        }
    
    def _format_context(self, documents: List[Dict]) -> str:
        """
        Format documents into context string
        
        Args:
            documents: List of document dicts
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get("metadata", {}).get("title", "Unknown")
            url = doc.get("metadata", {}).get("source_url", "")
            content = doc.get("content", "")
            
            context_parts.append(f"[Source {i}]\nTitle: {title}\nURL: {url}\nContent: {content}\n")
        
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

