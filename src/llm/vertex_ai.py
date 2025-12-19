# llm/vertex.py
import json
from google.oauth2 import service_account
from vertexai import init
from vertexai.preview.generative_models import GenerativeModel
from llm.base import LLMClient


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
