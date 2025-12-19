import json
from openai import OpenAI
from preprocess.llm.base import LLMClient


class OpenAIClient(LLMClient):
    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        self.client = OpenAI(api_key=api_key)

    def generate_json(self, prompt: str):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        return json.loads(content)