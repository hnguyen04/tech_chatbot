"""
Reranker using Qwen 3 model for cross-encoder reranking
"""
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document
from rag.config import QWEN_RERANKER_MODEL, DEVICE, TOP_N_RERANK


class QwenReranker:
    """
    Reranker using Qwen model for cross-encoder reranking
    """
    
    def __init__(self, model_name: str = QWEN_RERANKER_MODEL, device: str = DEVICE):
        """
        Initialize Qwen reranker
        
        Args:
            model_name: Model name/path
            device: Device to use (cpu/cuda)
        """
        self.model_name = model_name
        self.device = device
        
        # Load tokenizer and model
        # Note: Qwen models may need special handling for reranking
        # We'll use the model for sequence classification/scoring
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # For reranking, we'll use the base model and score query-doc pairs
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                trust_remote_code=True,
                num_labels=1  # Single score output
            )
        except Exception:
            # Fallback: use base model if sequence classification not available
            from transformers import AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        self.model.to(device)
        self.model.eval()
    
    def rerank(self, query: str, documents: List[Document], top_n: int = TOP_N_RERANK) -> List[Document]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: Query text
            documents: List of Document objects to rerank
            top_n: Number of top documents to return
            
        Returns:
            Reranked list of Document objects
        """
        if not documents:
            return []
        
        # Score each query-document pair
        scored_docs = []
        with torch.no_grad():
            for doc in documents:
                # Format as query-document pair
                # For Qwen, we can use a prompt format or concatenate
                text_pair = f"Query: {query}\nDocument: {doc.page_content}"
                
                inputs = self.tokenizer(
                    text_pair,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                
                # Get score (logit or pooled output)
                if hasattr(outputs, 'logits'):
                    score = outputs.logits.item()
                elif hasattr(outputs, 'last_hidden_state'):
                    # Use mean pooling if logits not available
                    score = outputs.last_hidden_state.mean().item()
                else:
                    score = 0.0
                
                scored_docs.append((doc, score))
        
        # Sort by score (descending) and return top_n
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in scored_docs[:top_n]]
        
        return reranked_docs
    
    def rerank_with_scores(self, query: str, documents: List[Document], top_n: int = TOP_N_RERANK) -> List[Tuple[Document, float]]:
        """
        Rerank documents with scores
        
        Args:
            query: Query text
            documents: List of Document objects to rerank
            top_n: Number of top documents to return
            
        Returns:
            List of tuples (Document, score) sorted by score descending
        """
        if not documents:
            return []
        
        scored_docs = []
        with torch.no_grad():
            for doc in documents:
                text_pair = f"Query: {query}\nDocument: {doc.page_content}"
                
                inputs = self.tokenizer(
                    text_pair,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                
                if hasattr(outputs, 'logits'):
                    score = outputs.logits.item()
                elif hasattr(outputs, 'last_hidden_state'):
                    score = outputs.last_hidden_state.mean().item()
                else:
                    score = 0.0
                
                scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_n]

