from sentence_transformers import SentenceTransformer
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import os
from dotenv import load_dotenv

load_dotenv()


class Embedder:
    def __init__(self, model_name: str = "dangvantuan/vietnamese-document-embedding"):
        """
        Embedder offline cho tiếng Việt, không dùng LLM API.
        model_name: Hugging Face model repository
        """
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts using the appropriate method
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (list of floats)
        """
        if not texts:
            return []
        
        if self.use_qwen:
            return self._embed_qwen(texts)
        elif self.use_sentence_transformer:
            return self._embed_sentence_transformer(texts)
        else:
            raise RuntimeError("Embedder not properly initialized")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text (LangChain compatibility)
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector as list of floats
        """
        return self.embed([text])[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents (LangChain compatibility)
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        return self.embed(texts)