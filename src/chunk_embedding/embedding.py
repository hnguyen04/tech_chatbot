import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional


class Embedder:
    """
    Embedder with MPS (Apple Silicon) support for faster inference on Mac.
    """
    
    def __init__(
        self,
        model_name: str = "dangvantuan/vietnamese-document-embedding",
        device: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        instruction: Optional[str] = None
    ):
        """
        Initialize embedder with automatic device detection for Mac MPS.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('mps', 'cuda', 'cpu'). Auto-detects if None.
            embedding_dim: Target embedding dimension (for Matryoshka models)
            instruction: Query instruction prefix (for instruction-tuned models)
        """
        # Auto-detect best device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
                print("Using MPS (Apple Silicon GPU)")
            elif torch.cuda.is_available():
                device = "cuda"
                print("Using CUDA GPU")
            else:
                device = "cpu"
                print("Using CPU")
        
        self.device = device
        self.target_dim = embedding_dim
        self.instruction = instruction
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device
        )
        
        # Get native embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedder ready (dim={self.embedding_dim}, device={device})")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            # Batch encode for efficiency
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Handle Matryoshka dimension reduction if specified
            if self.target_dim and embeddings.shape[1] > self.target_dim:
                embeddings = embeddings[:, :self.target_dim]
            
            return embeddings.tolist()
            
        except Exception as e:
            print(f"Embedding error: {e}")
            # Fallback to zero vectors
            dim = self.target_dim or self.embedding_dim
            return [[0.0] * dim for _ in texts]
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query with optional instruction prefix.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        text = f"{self.instruction}: {query}" if self.instruction else query
        return self.embed([text])[0]