# llm/gemini.py
import json
import google.generativeai as genai
from llm.base import LLMClient


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
