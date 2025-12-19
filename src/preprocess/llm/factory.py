# llm/factory.py
import os
import time
import random
from dotenv import load_dotenv
from preprocess.llm.gemini import GeminiClient
from preprocess.llm.openai import OpenAIClient
from preprocess.llm.vertex_ai import VertexAIClient


def build_llm():
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "gemini")
    primary = os.getenv("LLM_MODEL_PRIMARY")
    fallback = os.getenv("LLM_MODEL_FALLBACK")

    retries = int(os.getenv("LLM_RETRIES", "2"))
    sleep = float(os.getenv("LLM_SLEEP", "1.0"))

    if provider == "openai":
        primary_client = OpenAIClient(primary, os.getenv("OPENAI_API_KEY"))
        fallback_client = OpenAIClient(fallback, os.getenv("OPENAI_API_KEY"))
    elif provider == "vertex":
        project = os.getenv("VERTEX_PROJECT_ID")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        sa_path = os.getenv("VERTEX_SA_PATH")

        if not project:
            raise RuntimeError("VERTEX_PROJECT_ID is required")
        if not sa_path:
            raise RuntimeError("VERTEX_SA_PATH is required for vertex provider")

        primary_client = VertexAIClient(primary, project, location, sa_path)
        fallback_client = VertexAIClient(fallback, project, location, sa_path)
    else:
        primary_client = GeminiClient(primary, os.getenv("GEMINI_API_KEY"))
        fallback_client = GeminiClient(fallback, os.getenv("GEMINI_API_KEY"))

    def call(prompt: str):
        last_err = None

        for attempt in range(retries + 1):
            try:
                return primary_client.generate_json(prompt)
            except Exception as e:
                last_err = e
                time.sleep(sleep + random.uniform(0, 0.5))

        if fallback:
            try:
                return fallback_client.generate_json(prompt)
            except Exception as e:
                raise RuntimeError(f"LLM fallback failed: {e}") from last_err

        raise last_err

    return call
