# llm/vertex.py
import json
from google.oauth2 import service_account
from vertexai import init
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput
from llm.base import LLMClient, EmbeddingClient
from typing import List


class VertexAIClient(LLMClient):
    def __init__(self, model: str, project: str, location: str, sa_path: str):
        super().__init__(model)

        credentials = service_account.Credentials.from_service_account_file(
            sa_path
        )

        init(
            project=project,
            location=location,
            credentials=credentials,
        )

        self._model = GenerativeModel(model)

    def generate_json(self, prompt: str):
        resp = self._model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
            },
        )
        return json.loads(resp.text or "[]")

class VertexEmbeddingClient(EmbeddingClient):
    def __init__(self, model: str, project: str, location: str, sa_path: str):
        super().__init__(model)

        # Load credentials
        credentials = service_account.Credentials.from_service_account_file(sa_path)

        # Initialize Vertex AI
        init(
            project=project,
            location=location,
            credentials=credentials,
        )

        # Load embedding model
        self._model = TextEmbeddingModel.from_pretrained(model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        texts: list of string
        returns: list of embedding vectors (list of floats)
        """

        # Chuẩn bị input objects
        inputs = [TextEmbeddingInput(text=t) for t in texts]

        # Gọi batch embedding 1 lần
        responses = self._model.get_embeddings(inputs)

        # Trích xuất vector
        return [resp.values for resp in responses]