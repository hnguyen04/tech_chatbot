from abc import ABC, abstractmethod
from typing import List, Dict


class LLMClient(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate_json(self, prompt: str) -> List[Dict]:
        pass

class EmbeddingClient(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        texts: list of string
        return: list of embeddings
        """
        pass