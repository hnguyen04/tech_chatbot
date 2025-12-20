# llm/gemini.py
import json
import google.generativeai as genai
from llm.base import LLMClient, EmbeddingClient


class GeminiClient(LLMClient):
    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def generate_json(self, prompt: str):
        resp = self._model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        return json.loads(resp.text or "[]")

class GeminiEmbeddingClient(EmbeddingClient):
    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        genai.configure(api_key=api_key)
        self._model = genai.EmbeddingModel(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        texts: list of string
        returns: list of embedding vectors
        """
        resp = self._model.generate(texts)
        # giả sử resp.data[i].embedding trả về vector float
        embeddings = [item.embedding for item in resp.data]
        return embeddings