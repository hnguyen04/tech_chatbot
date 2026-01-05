import os
from dotenv import load_dotenv
from typing import List
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)

class MilvusWriter:
    def __init__(self, alias: str = "zilliz"):
        """
        Kết nối Milvus ORM thông qua alias
        """
        load_dotenv()
        host = os.getenv("ZILLIZIO_HOST")
        api_key = os.getenv("ZILLIZIO_API_KEY")

        if not host or not api_key:
            raise ValueError("Missing environment variables: ZILLIZIO_HOST or ZILLIZIO_API_KEY")

        self.alias = alias
        try:
            connections.connect(
                alias=self.alias,
                uri=host,
                token=api_key
            )
            print(f"Connected to Zilliz Cloud Milvus (ORM) at {host} with alias '{self.alias}'")
        except Exception as e:
            print(f"Failed to connect to Milvus ORM: {e}")
            raise e

    def create_collection(
        self,
        collection_name: str,
        dim: int = 768,
        metric_type: str = "L2",
        index_type: str = "HNSW",
        index_params: dict | None = None,
    ) -> Collection:
        """
        Tạo collection ORM với schema chuẩn và index
        """
        # Kiểm tra collection đã tồn tại chưa
        from pymilvus.orm import utility
        if utility.has_collection(collection_name, using=self.alias):
            print(f"Collection '{collection_name}' already exists")
            collection = Collection(name=collection_name, using=self.alias)
            collection.load()
            return collection

        # Schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="record_id", dtype=DataType.INT64, is_primary=False, auto_id=False),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        ]
        schema = CollectionSchema(fields, description="Product chunks embeddings")
        collection = Collection(name=collection_name, schema=schema, using=self.alias)
        print(f"Created collection '{collection_name}' with dim={dim} and fields: record_id, chunk_index, embedding, text")

        # Index
        if not index_params:
            if index_type.upper() == "HNSW":
                index_params = {"M": 16, "efConstruction": 200}
            elif index_type.upper() in ("IVF_FLAT", "IVF_PQ"):
                index_params = {"nlist": 128}

        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": index_type,
                "metric_type": metric_type,
                **index_params
            }
        )
        print(f"Created {index_type} index on '{collection_name}.embedding' with metric={metric_type}")

        # Load collection
        collection.load()
        print(f"Collection '{collection_name}' is loaded and ready for insert/query")

        return collection
    
    def save(self, collection_name: str, embedded_docs: List[dict]):
        """
        Lưu batch embedded_docs vào collection Milvus
        embedded_docs = [
            {"record_id": int, "chunk_index": int, "text": str, "embedding": List[float]}
        ]
        """
        if not embedded_docs:
            print("No documents to insert into Milvus.")
            return

        collection = Collection(name=collection_name, using=self.alias)
        collection.load()

        try:
            # Chuyển dữ liệu sang dạng columnar
            data = [
                [doc["record_id"] for doc in embedded_docs],
                [doc["chunk_index"] for doc in embedded_docs],
                [doc["embedding"] for doc in embedded_docs],
                [doc["text"] for doc in embedded_docs],
            ]
            mr = collection.insert(data)
            print(f"Inserted {len(embedded_docs)} vectors into '{collection_name}'")
        except Exception as e:
            print(f"Failed to insert into Milvus: {e}")
            raise e
    
    def drop_collection(self, collection_name: str):
        """
        Xoá collection khỏi Milvus
        """
        from pymilvus.orm import utility
        if utility.has_collection(collection_name, using=self.alias):
            utility.drop_collection(collection_name, using=self.alias)
            print(f"Dropped collection '{collection_name}'")
        else:
            print(f"Collection '{collection_name}' does not exist")
