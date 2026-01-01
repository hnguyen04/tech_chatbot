from sentence_transformers import SentenceTransformer
from typing import List

class Embedder:
    def __init__(self, model_name: str = "dangvantuan/vietnamese-document-embedding"):
        """
        Embedder offline cho tiếng Việt, không dùng LLM API.
        model_name: Hugging Face model repository
        """
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        dim = self.model.get_sentence_embedding_dimension()
        embeddings = []
        for t in texts:
            try:
                emb = self.model.encode([t], show_progress_bar=False, convert_to_numpy=True)
                embeddings.append(emb[0].tolist())
            except Exception:
                # Nếu lỗi, thêm zero vector
                embeddings.append([0.0]*dim)
        return embeddings