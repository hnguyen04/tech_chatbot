"""
Smart RAG Pipeline with mode selection.
Routes between Fast (simple RAG with Gemini) and Deep (agentic RAG with GPT) modes.
"""
from typing import Dict, Any, Optional, Literal
from rag.unified_pipeline import UnifiedRAGPipeline
from rag.agentic_pipeline import AgenticRAGPipeline
from rag.config import MILVUS_COLLECTION_NAME


SearchMode = Literal["fast", "deep"]


class SmartRAGPipeline:
    """
    Unified entry point with intelligent mode selection.
    
    Modes:
    - "fast": Auto-detects query complexity. Simple lookups are direct, complex queries use decomposition.
    - "deep": Agentic reasoning with GPT and tool use (best for multi-step/comparison queries)
    """
    
    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize smart pipeline with both simple and agentic backends.
        
        Args:
            collection_name: Milvus collection name
        """
        print("Initializing Smart RAG Pipeline...")
        
        target_collection = collection_name or MILVUS_COLLECTION_NAME
        
        # Initialize the simple/fast pipeline (uses Gemini)
        self.simple_pipeline = UnifiedRAGPipeline(collection_name=target_collection)
        
        # Initialize the agentic pipeline with OpenAI GPT for Deep mode
        # Shares retriever and pg_client for efficiency, but uses its own LLM
        self.agentic_pipeline = AgenticRAGPipeline(
            llm_service=None,  # Will create OpenAI service internally
            retriever=self.simple_pipeline.retriever,
            pg_client=self.simple_pipeline.pg_client,
            collection_name=target_collection,
            use_openai=True  # Use GPT for Deep mode
        )
        
        print("Smart RAG Pipeline ready (Fast: Gemini, Deep: GPT)")
    
    def query(
        self,
        query: str,
        mode: SearchMode = "fast",
        top_k: int = 20,
        top_n: int = 5,
        use_history: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query using the selected mode.
        
        Args:
            query: User's question
            mode: "fast" for simple RAG, "deep" for agentic RAG
            top_k: Number of documents to retrieve (fast mode)
            top_n: Number of documents after reranking (fast mode)
            use_history: Whether to use conversation history
            
        Returns:
            Response dict with answer, sources, and metadata
        """
        # Sync conversation history between pipelines before query
        if use_history:
            self._sync_history()
        
        if mode == "deep":
            print(f"\n[SmartRAG] Mode: DEEP (Agentic)")
            result = self.agentic_pipeline.query(query, use_history=use_history)
            result["mode"] = "deep"
        else:
            print(f"\n[SmartRAG] Mode: FAST")
            # Fast mode: auto-detect query complexity
            # Simple lookups skip decomposition, complex queries use it
            result = self.simple_pipeline.query(
                query,
                top_k=top_k,
                top_n=top_n,
                use_history=use_history
                # skip_decomposition=None (default) -> auto-detect
            )
            result["mode"] = "fast"
        
        # Sync history after query so both pipelines stay in sync
        if use_history:
            self._sync_history()
        
        return result
    
    def _sync_history(self):
        """Synchronize conversation history between both pipelines."""
        # Use simple pipeline as source of truth (most commonly used)
        simple_history = self.simple_pipeline.conversation_history
        agentic_history = self.agentic_pipeline.conversation_history
        
        # Merge: take the longer one as it's more up-to-date
        if len(agentic_history) > len(simple_history):
            self.simple_pipeline.conversation_history = agentic_history.copy()
        elif len(simple_history) > len(agentic_history):
            self.agentic_pipeline.conversation_history = simple_history.copy()
    
    def clear_history(self):
        """Clear conversation history for both pipelines."""
        self.simple_pipeline.clear_history()
        self.agentic_pipeline.clear_history()
    
    def get_history(self):
        """Get conversation history from active pipeline."""
        # Both pipelines may have history; return the simple one as primary
        return self.simple_pipeline.get_history()
    
    def __del__(self):
        """Cleanup resources."""
        # Resources are cleaned up by child pipelines
        pass
