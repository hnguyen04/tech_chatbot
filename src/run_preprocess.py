"""
Runner script (fixed source, no CLI args):

Pipeline steps:
    load (stream) -> clean -> dedup -> LLM enrich -> insert Postgres
"""

import json
import os
import traceback
from typing import Iterable
from dotenv import load_dotenv
import google.generativeai as genai
from dotenv import load_dotenv
from preprocess.pipeline import PreprocessPipeline
from storage.s3_storage import S3Storage
from preprocess.models import CleanRecord
from preprocess.db_writer import PostgresWriter
from datetime import datetime


# =========================
# FIXED INPUT SOURCE HERE
# =========================
SOURCE = "s3://20251-data-science/output/"
# SOURCE = "data/sample.json"   # local debug

def list_s3_files(source: str) -> list[str]:
    bucket_from_uri, prefix = parse_s3_uri(source)

    s3 = S3Storage()
    if bucket_from_uri:
        s3.bucket = bucket_from_uri

    return s3.list_keys(prefix)


def parse_s3_uri(uri: str) -> tuple[str | None, str]:
    """Return (bucket, key). If bucket missing, (None, key)."""
    path = uri.replace("s3://", "", 1)
    if "/" not in path:
        return None, path
    bucket, key = path.split("/", 1)
    return bucket, key


def load_docs(source: str) -> Iterable[dict]:
    """
    - Trả về Iterable
    - S3: stream JSON array
    - Local: json.load (debug/dev)
    """

    # ---------- S3 SOURCE ----------
    if source.startswith("s3://"):
        bucket_from_uri, key = parse_s3_uri(source)
        s3 = S3Storage()

        if bucket_from_uri:
            s3.bucket = bucket_from_uri

        return s3.stream_json(key)

    # ---------- LOCAL FILE ----------
    with open(source, "r", encoding="utf-8") as f:
        data = json.load(f)

        if isinstance(data, list):
            return iter(data)

        return iter([])


def main():
    load_dotenv()

    # 🔹 DB writer dùng chung
    db_writer = PostgresWriter()

    pipeline = PreprocessPipeline(
        db_writer=db_writer
    )

    # =========================
    # S3 MODE
    # =========================
    if SOURCE.startswith("s3://"):
        bucket, prefix = parse_s3_uri(SOURCE)

        s3 = S3Storage()
        if bucket:
            s3.bucket = bucket

        keys = s3.list_keys(prefix)
        json_keys = [k for k in keys if k.endswith(".json")]

        print(f"Found {len(json_keys)} JSON files")

        for key in json_keys:
            full_uri = f"s3://{bucket}/{key}"
            print(f"\n▶ Processing {full_uri}")

            # ==================================
            # CHECK processed TRƯỚC KHI RUN
            # ==================================
            if db_writer.is_json_processed(key):
                print(f"⏭  Skipped (already processed): {key}")
                continue

            docs = load_docs(full_uri)

            try:
                pipeline.run(docs)

                # ==================================
                # MARK processed SAU KHI THÀNH CÔNG
                # ==================================
                db_writer.mark_json_processed(key)
                print(f"Marked processed: {key}")

            except Exception as e:
                # KHÔNG mark processed nếu lỗi
                print(f"Failed processing {key}: {e}")
                # traceback.print_exc()
                # print(f"Failed processing {key}: {e}")
                # return

        return

    # =========================
    # LOCAL MODE
    # =========================
    docs = load_docs(SOURCE)
    pipeline.run(docs)

if __name__ == "__main__":
    main()