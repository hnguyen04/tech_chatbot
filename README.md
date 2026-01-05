# Tech Chatbot with RAG

A Retrieval-Augmented Generation (RAG) chatbot for technical product information using Milvus/Zilliz Cloud for vector storage.

## Features

- 🔍 **Advanced Retrieval**: Semantic search using Milvus/Zilliz Cloud with Vietnamese embeddings
- 🧠 **Smart Reranking**: Qwen-based reranking for better relevance
- 💬 **Conversational AI**: Powered by Google Gemini
- 📊 **Flexible Data Sources**: Load from PostgreSQL with intelligent chunking
- 🎯 **Metadata Filtering**: Filter by category, brand, product specs
- 🚀 **Production Ready**: Scalable cloud-native vector storage

## Architecture

```
PostgreSQL Products
    ↓
Smart Chunking (Spec-Aware)
    ↓
Vietnamese Embeddings (768-dim)
    ↓
Milvus/Zilliz Cloud (tech_embeddings)
    ↓
User Query → Vietnamese Embedding
    ↓
Retrieval (Top-K) + Metadata Enrichment
    ↓
Qwen Reranking (Top-N)
    ↓
Gemini LLM (Response Generation)
    ↓
User Response with Sources
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd tech_chatbot

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Milvus/Zilliz Cloud

1. Create a free account at [Zilliz Cloud](https://cloud.zilliz.com/)
2. Create a new cluster
3. Copy your cluster URI and API token

### 3. Configure Environment

Create a `.env` file:

```bash
# Zilliz Cloud (both naming styles for compatibility)
ZILLIZIO_HOST=https://your-cluster.serverless.aws-eu-central-1.cloud.zilliz.com
ZILLIZIO_API_KEY=your_api_token
ZILLIZ_CLOUD_URI=https://your-cluster.serverless.aws-eu-central-1.cloud.zilliz.com
ZILLIZ_CLOUD_TOKEN=your_api_token

# Milvus Collection
MILVUS_COLLECTION_NAME=tech_embeddings
MILVUS_EMBEDDING_DIM=768

# Models
QWEN_RERANKER_MODEL=Qwen/Qwen2.5-0.5B-Instruct
GEMINI_API_KEY=your_gemini_api_key

# Device
DEVICE=cpu  # or "cuda"
```

### 4. Verify Setup

```bash
# Run automated setup and verification
python setup_milvus.py
```

### 5. Ingest Data

```bash
cd src

# From JSON files (default)
python -m rag.data_ingestion --vector-store milvus

# From PostgreSQL
python -m rag.data_ingestion --use-postgres --vector-store milvus

# Force reindex (recreate collection)
python -m rag.data_ingestion --vector-store milvus --force-reindex
```

### 6. Run the Chatbot

```bash
cd src

# Streamlit web interface
streamlit run rag/streamlit_app.py
```

## Project Structure

```
tech_chatbot/
├── src/
│   ├── chunk_embedding/      # Embedding pipeline modules
│   │   ├── builder.py         # Document and spec block builders
│   │   ├── chunker.py         # Text chunking logic
│   │   ├── db_loader.py       # PostgreSQL data loader
│   │   ├── embedding.py       # Embedding generation
│   │   ├── pipeline.py        # End-to-end embedding pipeline
│   │   └── vector_db_writer.py # Milvus writer
│   ├── crawlers/              # Web scraping modules
│   │   ├── cell_phone_s/      # Cellphones.com crawler
│   │   ├── thegioididong/     # Thegioididong.com crawler
│   │   └── core/              # Core crawling infrastructure
│   ├── eval/                  # Evaluation and testing tools
│   │   ├── evaluation.py      # Automated batch evaluation
│   │   ├── generate_ground_truth.py  # Golden dataset generation
│   │   ├── run_retrieval_benchmark.py # Retrieval benchmarking
│   │   └── test_chatbot_vietnamese.py # Interactive CLI testing
│   ├── llm/                   # LLM service abstractions
│   │   ├── base.py            # Base LLM interface
│   │   ├── factory.py         # LLM factory
│   │   ├── gemini.py          # Google Gemini implementation
│   │   ├── openai.py          # OpenAI implementation
│   │   └── vertex_ai.py       # Vertex AI implementation
│   ├── preprocess/            # Data preprocessing
│   │   ├── cleaners.py        # Text cleaning utilities
│   │   ├── db_writer.py       # Database writing
│   │   ├── models.py          # Data models
│   │   └── pipeline.py        # Preprocessing pipeline
│   ├── rag/                   # RAG pipeline components
│   │   ├── config.py          # Configuration
│   │   ├── vietnamese_retriever.py  # Vietnamese embedding retriever
│   │   ├── unified_pipeline.py      # Integrated RAG pipeline
│   │   ├── reranker.py              # Qwen reranker
│   │   ├── llm_service.py           # LLM service wrapper
│   │   ├── streamlit_app.py         # Web interface
│   │   └── data_ingestion.py        # Data ingestion scripts
│   ├── storage/               # Storage clients
│   │   ├── milvus_client.py   # Milvus/Zilliz client
│   │   ├── postgres_client.py # PostgreSQL client
│   │   └── s3_storage.py      # S3 storage client
│   ├── run_embeddings.py      # Run embedding pipeline
│   ├── run_preprocess.py      # Run preprocessing
│   └── run_rag_chatbot.py     # Run RAG chatbot
├── docs/                      # Documentation
│   ├── PLAN.md                # Project plan
│   ├── PLAN_DATASETS.md       # Dataset planning
│   └── PLAN_EVAL.md           # Evaluation planning
├── output/                    # Generated data files
│   ├── tgdd_articles.json     # Scraped articles
│   ├── tgdd_products.json     # Scraped products
│   └── manual_evaluation_prompts/  # Evaluation prompts
├── entities.json              # Entity definitions
├── EVALUATION_DATASET.json    # Evaluation dataset
├── GOLDEN_DATASET.json        # Golden dataset for evaluation
├── requirements.txt           # Python dependencies
├── setup_milvus.py            # Setup automation script
└── README.md                  # This file
```

## Components

### Vector Store

**Milvus/Zilliz Cloud**
- ✅ Cloud-native, scalable
- ✅ Metadata filtering
- ✅ Production-ready
- ✅ Real-time updates
- ✅ Supports billions of vectors
- ✅ Vietnamese-optimized embeddings

### Embeddings

- **Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Dimension**: 896
- **Method**: Mean pooling of last hidden states
- **Max Length**: 512 tokens

### Retrieval Pipeline

1. **Initial Retrieval**: Top-20 documents via vector similarity (COSINE)
2. **Reranking**: Top-5 documents using Qwen cross-encoder
3. **Generation**: Context-aware response via Gemini

### Data Schema

Milvus collection fields:
- `id`: Auto-generated primary key (INT64)
- `vector`: Embedding vector (FLOAT_VECTOR, dim=896)
- `text`: Document content (VARCHAR, max=65535)
- `metadata`: Document metadata (JSON)
  - `doc_id`: Original document ID
  - `source_url`: Source URL
  - `title`: Document title
  - `category`: Category
  - `content_type`: article/product
  - `chunk_index`: Chunk number
  - `domain`: Source domain
  - `images`: Image URLs array

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VECTOR_STORE` | Vector store type | `milvus` |
| `ZILLIZ_CLOUD_URI` | Zilliz cluster endpoint | Required |
| `ZILLIZ_CLOUD_TOKEN` | Zilliz API token | Required |
| `MILVUS_COLLECTION_NAME` | Collection name | `tech_chatbot_collection` |
| `MILVUS_EMBEDDING_DIM` | Embedding dimension | `896` |
| `QWEN_EMBEDDING_MODEL` | Embedding model | `Qwen/Qwen2.5-0.5B-Instruct` |
| `GEMINI_API_KEY` | Gemini API key | Required |
| `TOP_K_RETRIEVE` | Initial retrieval count | `20` |
| `TOP_N_RERANK` | Reranked document count | `5` |
| `CHUNK_SIZE` | Text chunk size | `500` |
| `CHUNK_OVERLAP` | Chunk overlap | `50` |
| `DEVICE` | Computation device | `cpu` |

## Usage Examples

### Python API

```python
from rag.rag_pipeline import RAGPipeline

# Initialize pipeline (auto-detects vector store from config)
pipeline = RAGPipeline()

# Query the chatbot
response = pipeline.query("What are the best laptops for gaming?")

print(response["answer"])
print(f"Sources: {len(response['sources'])}")
for source in response["sources"]:
    print(f"  - {source['title']}: {source['url']}")
```

### Custom Retriever

```python
from storage.milvus_client import MilvusRetriever
from rag.embedding_service import QwenEmbeddings

# Initialize components
retriever = MilvusRetriever(
    collection_name="my_collection",
    dimension=896
)
embeddings = QwenEmbeddings()

# Search with filter
query_embedding = embeddings.embed_query("laptop gaming")
results = retriever.retrieve(
    query_embedding,
    k=10,
    filter_expr='category == "laptop"'
)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
```

### Using the Pipeline

```python
from rag.unified_pipeline import UnifiedRAGPipeline

# Initialize pipeline (uses Milvus with Vietnamese embeddings)
pipeline = UnifiedRAGPipeline()

# Query
result = pipeline.query("Laptop gaming tốt nhất")
print(result["answer"])
```

## Advanced Features

### Metadata Filtering

```python
# Filter by category
retriever.retrieve(
    query_embedding,
    k=20,
    filter_expr='category == "smartphone"'
)

# Complex filters
retriever.retrieve(
    query_embedding,
    k=20,
    filter_expr='category == "laptop" and domain == "thegioididong.com"'
)
```

### Collection Management

```python
from storage.milvus_client import MilvusRetriever

retriever = MilvusRetriever()

# Get statistics
stats = retriever.get_collection_stats()
print(f"Total documents: {stats['row_count']}")

# Check if collection exists
exists = retriever.collection_exists()

# Delete collection (caution!)
retriever.delete_collection()
```

## Troubleshooting

### Common Issues

**Problem**: Dimension mismatch error

**Solution**: 
```bash
# Run setup script to check dimensions
python setup_milvus.py
# Update MILVUS_EMBEDDING_DIM in .env
```

**Problem**: Connection timeout to Zilliz

**Solution**:
- Verify cluster is active in Zilliz dashboard
- Check URI format includes `https://`
- Verify API token is correct

**Problem**: Out of memory during ingestion

**Solution**:
- Reduce batch size in `data_ingestion.py`
- Use CPU instead of CUDA if GPU memory is limited
- Process data in smaller batches

### Getting Help

1. Check `MILVUS_SETUP.md` for detailed setup guide
2. Run `python setup_milvus.py` for automated diagnostics
3. See [Milvus Documentation](https://milvus.io/docs)
4. See [Zilliz Documentation](https://docs.zilliz.com/)

## Performance

### Benchmarks


### Optimization Tips

1. **Batch Processing**: Use batch embedding generation
2. **Top-K Selection**: Reduce `TOP_K_RETRIEVE` if speed is critical
3. **GPU Acceleration**: Set `DEVICE=cuda` for faster embeddings
4. **Collection Tuning**: Adjust index parameters in Milvus for your use case
5. **Caching**: Cache frequently used embeddings

## Development

### Adding New Data Sources

1. Implement data loader in `data_ingestion.py`
2. Format as LangChain `Document` objects
3. Run ingestion pipeline
4. Verify in Zilliz dashboard

## License

[Your License Here]

## Acknowledgments

- [Milvus](https://milvus.io/) - Vector database
- [Zilliz Cloud](https://cloud.zilliz.com/) - Managed Milvus service
- [LangChain](https://langchain.com/) - RAG framework
- [Qwen](https://huggingface.co/Qwen) - Embeddings and reranking
- [Google Gemini](https://ai.google.dev/) - LLM

## Support

For issues and questions:
- Setup help: Run `python setup_milvus.py`