"""
Configuration for RAG system
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Qwen Model Configuration
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")  # Qwen3-Embedding-0.6B (default: 1024 dim)
QWEN_RERANKER_MODEL = os.getenv("QWEN_RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")  # Qwen3-Reranker-0.6B
QWEN_EMBEDDING_INSTRUCTION = os.getenv("QWEN_EMBEDDING_INSTRUCTION", None)  # Optional instruction for better performance
QWEN_EMBEDDING_DIM = os.getenv("QWEN_EMBEDDING_DIM", None)  # Custom dimension (32-1024 for 0.6B model), None = use default

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# OpenAI API Configuration (Used for ground truth generation)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Milvus/Zilliz Cloud Configuration
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "tech_embeddings")
MILVUS_EMBEDDING_DIM = int(os.getenv("MILVUS_EMBEDDING_DIM", "1024"))  # Default 1024 for Qwen3-Embedding-0.6B, will be auto-detected
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI") or os.getenv("ZILLIZIO_HOST")
ZILLIZ_CLOUD_TOKEN = os.getenv("ZILLIZ_CLOUD_TOKEN") or os.getenv("ZILLIZIO_API_KEY")

# Vector Store Configuration (Milvus only)

# Retrieval Configuration
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "10"))  # Number of docs to retrieve per sub-query
TOP_N_RERANK = int(os.getenv("TOP_N_RERANK", "5"))  # Number of docs after reranking

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# PostgreSQL Configuration (using existing env vars)
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST")
POSTGRES_DB_PORT = os.getenv("POSTGRES_DB_PORT")
POSTGRES_DB_NAME = os.getenv("POSTGRES_DB_NAME")
POSTGRES_DB_USER = os.getenv("POSTGRES_DB_USER")
POSTGRES_DB_PASSWORD = os.getenv("POSTGRES_DB_PASSWORD")

# Device Configuration
DEVICE = os.getenv("DEVICE", "cpu")  # cpu or cuda or mps

# Reranking Configuration
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"  # Set to false to disable

# Evaluation Configuration
EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "data/eval_dataset.json")