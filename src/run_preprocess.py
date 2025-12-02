"""
Runner script:
    python src/run_preprocess.py path/to/data.json
    python src/run_preprocess.py s3://bucket/key.json

Pipeline steps: load -> clean -> dedup -> LLM enrich (stub) -> insert Postgres.
"""
import argparse
import json
from dotenv import load_dotenv
from preprocess.pipeline import PreprocessPipeline
from storage.s3_storage import S3Storage


def parse_s3_uri(uri: str) -> tuple[str | None, str]:
    """Return (bucket, key). If bucket missing, (None, key)."""
    path = uri.replace("s3://", "", 1)
    if "/" not in path:
        return None, path
    bucket, key = path.split("/", 1)
    return bucket, key


def load_docs(source: str):
    if source.startswith("s3://"):
        bucket_from_uri, key = parse_s3_uri(source)
        s3 = S3Storage()
        if bucket_from_uri:
            s3.bucket = bucket_from_uri
        return s3.download_json(key)

    with open(source, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Preprocess and load into Postgres")
    parser.add_argument("source", help="Local JSON path or s3://bucket/key.json")
    args = parser.parse_args()

    docs = load_docs(args.source)
    pipeline = PreprocessPipeline()
    pipeline.run(docs)


if __name__ == "__main__":
    main()
