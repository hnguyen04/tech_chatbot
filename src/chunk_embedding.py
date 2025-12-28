from chunk_embedding.pipeline import EmbeddingPipeline
from chunk_embedding.vector_db_writer import MilvusWriter

if __name__ == "__main__":
    # milvusWriter = MilvusWriter()
    # milvusWriter.drop_collection("tech_embeddings_v2")
    # milvusWriter.create_collection("tech_embeddings_v2")
    pipeline = EmbeddingPipeline()
    batch_size = 10
    last_id = 0

    while True:
        embedded_docs, last_id = pipeline.run_batch(batch_size, last_id, collection_name="tech_embeddings_v2")
        if not embedded_docs:
            print("✅ No more docs to process, finished.")
            break  # hết dữ liệu, thoát loop
        else:
            print(f"Processed {len(embedded_docs)} docs, last_id={last_id}")