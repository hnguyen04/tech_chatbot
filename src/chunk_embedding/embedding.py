from sentence_transformers import SentenceTransformer
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import os
from dotenv import load_dotenv

load_dotenv()


class Embedder:
    """
    Unified embedder supporting both SentenceTransformer models and Qwen models.
    Automatically detects model type and uses appropriate method.
    
    Supports Qwen3-Embedding models with:
    - Instruction-aware embeddings (for better performance)
    - Custom dimensions (MRL Support: 32-1024 for 0.6B model)
    - SentenceTransformer compatibility (requires transformers>=4.51.0, sentence-transformers>=2.7.0)
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        instruction: Optional[str] = None
    ):
        """
        Initialize embedder
        
        Args:
            model_name: Hugging Face model repository name
                       If None, uses QWEN_EMBEDDING_MODEL from config, or defaults to Vietnamese model
            device: Device to use (cpu/cuda). If None, uses DEVICE from config or defaults to cpu
            embedding_dim: Custom embedding dimension (for Qwen3 models with MRL support: 32-1024)
                          If None, uses model's default dimension
            instruction: Instruction text for instruction-aware embeddings (recommended for Qwen3)
                        Improves performance by 1-5% when task-specific instructions are provided
        """
        # Get model name from env or use default
        if model_name is None:
            model_name = os.getenv("QWEN_EMBEDDING_MODEL") or "dangvantuan/vietnamese-embedding"
        
        if device is None:
            device = os.getenv("DEVICE", "cpu")
        
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim
        self.instruction = instruction or os.getenv("QWEN_EMBEDDING_INSTRUCTION")
        self.use_sentence_transformer = False
        self.use_qwen = False
        self.is_qwen3 = False
        
        # Check if this is a Qwen3-Embedding model
        if "qwen3-embedding" in model_name.lower() or "qwen3" in model_name.lower():
            self.is_qwen3 = True
        
        # Try SentenceTransformer first (for models like Vietnamese embedding, Qwen3-Embedding, etc.)
        try:
            print(f"Attempting to load as SentenceTransformer: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)
            self.use_sentence_transformer = True
            
            # Get default embedding dimension
            default_dim = self.model.get_sentence_embedding_dimension()
            
            # For Qwen3 models, support custom dimensions (MRL)
            if self.is_qwen3 and embedding_dim is not None:
                if not (32 <= embedding_dim <= 1024):
                    raise ValueError(f"Qwen3-Embedding-0.6B supports dimensions 32-1024, got {embedding_dim}")
                self.embedding_dim = embedding_dim
                print(f"✓ Loaded as SentenceTransformer with custom dimension: {self.embedding_dim}")
            else:
                self.embedding_dim = default_dim
                print(f"✓ Loaded as SentenceTransformer (dimension: {self.embedding_dim})")
            
            if self.is_qwen3 and self.instruction:
                print(f"  Instruction-aware mode enabled with instruction: '{self.instruction[:50]}...'")
                
        except Exception as e:
            # If SentenceTransformer fails, try as Qwen/transformers model
            print(f"SentenceTransformer failed: {e}")
            print(f"Attempting to load as Qwen/transformers model: {model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                self.model.to(device)
                self.model.eval()
                self.use_qwen = True
                
                # Get embedding dimension
                if embedding_dim is not None:
                    if self.is_qwen3 and not (32 <= embedding_dim <= 1024):
                        raise ValueError(f"Qwen3-Embedding-0.6B supports dimensions 32-1024, got {embedding_dim}")
                    self.embedding_dim = embedding_dim
                elif hasattr(self.model.config, 'hidden_size'):
                    self.embedding_dim = self.model.config.hidden_size
                else:
                    # Test with a dummy embedding
                    test_emb = self._embed_qwen(["test"])
                    self.embedding_dim = len(test_emb[0])
                
                print(f"✓ Loaded as Qwen/transformers model (dimension: {self.embedding_dim})")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{model_name}' as both SentenceTransformer and Qwen model: {e2}"
                ) from e
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling of token embeddings for Qwen models
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _embed_qwen(self, texts: List[str]) -> List[List[float]]:
        """
        Embed using Qwen/transformers model with mean pooling
        """
        if not texts:
            return []
        
        # Tokenize
        # Qwen3 models support up to 32K context, but 512 is reasonable for embeddings
        max_length = 512 if not self.is_qwen3 else 8192  # Can use longer for Qwen3 if needed
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Mean pooling
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize (improves cosine similarity)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to list
        return embeddings.cpu().tolist()
    
    def _embed_sentence_transformer(self, texts: List[str]) -> List[List[float]]:
        """
        Embed using SentenceTransformer
        Supports Qwen3-Embedding features:
        - Instruction-aware embeddings (prepend instruction to text)
        - Custom dimensions (MRL) - handled automatically by SentenceTransformer
        """
        if not texts:
            return []
        
        dim = self.embedding_dim
        
        # For Qwen3 models with instruction-aware support
        # According to Qwen3 docs, instructions improve performance by 1-5%
        if self.is_qwen3 and self.instruction:
            # Prepend instruction to each text for better performance
            texts_with_instruction = [f"{self.instruction}\n{text}" for text in texts]
        else:
            texts_with_instruction = texts
        
        # Standard SentenceTransformer encoding
        # Note: For Qwen3 models with custom dimensions (MRL), SentenceTransformer 
        # handles it automatically when model supports it
        try:
            # Batch encoding is more efficient
            encode_kwargs = {
                'show_progress_bar': False,
                'convert_to_numpy': True,
                'device': self.device,
                'normalize_embeddings': True
            }
            
            # For Qwen3 with custom dimension, try to pass output_dim if supported
            if self.is_qwen3 and self.embedding_dim and hasattr(self.model, 'encode'):
                # Some SentenceTransformer versions support output_dim parameter
                try:
                    emb = self.model.encode(
                        texts_with_instruction,
                        output_dim=self.embedding_dim,
                        **encode_kwargs
                    )
                    return emb.tolist()
                except TypeError:
                    # output_dim not supported, use default dimension
                    pass
            
            # Standard encoding
            emb = self.model.encode(texts_with_instruction, **encode_kwargs)
            embeddings = emb.tolist()
            
            # Verify dimensions match expected
            if self.embedding_dim and len(embeddings[0]) != self.embedding_dim:
                print(f"Warning: Expected dimension {self.embedding_dim}, got {len(embeddings[0])}. "
                      f"Using model's default dimension.")
                self.embedding_dim = len(embeddings[0])
            
            return embeddings
            
        except Exception as e:
            print(f"Error in batch encoding, falling back to individual encoding: {e}")
            # Fallback to individual encoding
            embeddings = []
            for t in texts_with_instruction:
                try:
                    emb = self.model.encode(
                        [t],
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        device=self.device,
                        normalize_embeddings=True
                    )
                    embeddings.append(emb[0].tolist())
                except Exception:
                    # If error, add zero vector
                    embeddings.append([0.0] * dim)
            return embeddings
    
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