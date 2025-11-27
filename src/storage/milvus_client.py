import os
from dotenv import load_dotenv
from pymilvus import MilvusClient


class ZillizMilvusClient:

    def __init__(self):
        load_dotenv()
        # Lấy thông tin từ biến môi trường
        host = os.getenv("ZILLIZIO_HOST")
        api_key = os.getenv("ZILLIZIO_API_KEY")

        if not host or not api_key:
            raise ValueError("Missing environment variables: ZILLIZIO_HOST or ZILLIZIO_API_KEY")

        # Connect tới Zilliz Cloud
        try:
            self.client = MilvusClient(uri=host, token=api_key)
            print(f"✅ Connected successfully to Zilliz Cloud Milvus at {host}")
        except Exception as e:
            print(f"❌ Failed to connect to Zilliz Cloud Milvus: {e}")
            self.client = None



# client = ZillizMilvusClient()
