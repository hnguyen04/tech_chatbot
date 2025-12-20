from chunk_embedding.db_loader import PostgresLoader
from chunk_embedding.builder import SpecBlockBuilder, DocumentBuilder
from chunk_embedding.chunker import Chunker
from chunk_embedding.embedding import Embedder
from chunk_embedding.vector_db_writer import MilvusWriter
from typing import List


class EmbeddingPipeline:
    def __init__(self):
        self.loader = PostgresLoader()
        self.spec_builder = SpecBlockBuilder()
        self.chunker = Chunker()
        self.doc_builder = DocumentBuilder()
        self.embedder = Embedder()
        self.milvus_writer = MilvusWriter()

    def _load_records(self, batch_size: int, last_id: int):
        records = self.loader.load_records_batch(batch_size=batch_size, last_id=last_id)
        if not records:
            return [], {}, last_id

        record_ids = [r["id"] for r in records]
        specs_map = self.loader.load_specs_by_product_ids(record_ids)
        return records, specs_map, records[-1]["id"]

    def _build_docs(self, records, specs_map):
        docs = []
        texts_to_embed = []

        for record in records:
            specs = specs_map.get(record["id"], [])
            spec_block = self.spec_builder.build(specs)

            # chỉ chunk phần content_text
            content_chunks = self.chunker.chunk(record.get("content_text"))

            for idx, chunk_text in content_chunks:
                # kết hợp spec_block + chunk_text
                full_text = ""
                if record["content_type"] == "product" and spec_block:
                    full_text += spec_block + "\n"
                full_text += f"Title: {record['full_title']}\n"
                full_text += f"Category: {record['category_eng'] or record['category']}\n"
                if record.get("brand"):
                    full_text += f"Brand: {record['brand']}\n"
                full_text += "Content:\n" + chunk_text

                # nếu full_text vẫn quá dài → cắt thêm
                safe_chunks = self.chunker.chunk(full_text)
                for safe_idx, safe_text in safe_chunks:
                    docs.append({
                        "record_id": record["id"],
                        "chunk_index": idx*100 + safe_idx,  # giữ thứ tự
                        "text": safe_text,
                    })
                    texts_to_embed.append(safe_text)

        return docs, texts_to_embed

    def _compute_embeddings(self, texts):
        return self.embedder.embed(texts)

    #main run method
    def run_batch(self, batch_size: int = 50, last_id: int = 0,
                collection_name: str | None = None):
        """
        Chạy pipeline 1 batch:
        - Load dữ liệu
        - Build document chunks
        - Tính embedding
        - Nếu collection_name được truyền:
            + Lưu vào Milvus
            + Đánh dấu chunked trong Postgres
        """
        try:
            records, specs_map, new_last_id = self._load_records(batch_size, last_id)
            if not records:
                return [], last_id

            docs, texts_to_embed = self._build_docs(records, specs_map)

            try:
                embeddings = self._compute_embeddings(texts_to_embed)
            except Exception as e:
                print(f"❌ Failed to compute embeddings: {e}")
                embeddings = [None] * len(texts_to_embed)  # giữ số lượng, bỏ chunk fail

            # zip docs với embeddings, bỏ None
            embedded_docs = [
                {
                    "record_id": doc["record_id"],
                    "chunk_index": doc["chunk_index"],
                    "text": doc["text"],
                    "embedding": emb,
                }
                for doc, emb in zip(docs, embeddings)
                if emb is not None
            ]

            # Lưu Milvus + mark chunked
            if collection_name and embedded_docs:
                try:
                    self.save_to_milvus(embedded_docs, collection_name)
                except Exception as e:
                    print(f"❌ Failed to save to Milvus: {e}")

            return embedded_docs, new_last_id

        except Exception as e:
            print(f"❌ Failed to process batch: {e}")
            return [], last_id

    # -------------------------
    # 5️⃣ Save vào Milvus
    # -------------------------
    def save_to_milvus(self, embedded_docs: List[dict], collection_name: str):
        """
        Lưu vào Milvus và đánh dấu chunked trong Postgres
        """
        if not embedded_docs:
            return

        # Lưu vào Milvus
        self.milvus_writer.save(collection_name, embedded_docs)

        # Lấy record_id duy nhất để đánh dấu chunked
        record_ids = list({doc["record_id"] for doc in embedded_docs})
        self.loader.mark_chunked(record_ids)
        print(f"✅ Marked {len(record_ids)} records as chunked in Postgres")