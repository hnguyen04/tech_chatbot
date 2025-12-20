import json
import boto3
import os
from dotenv import load_dotenv
import re
from datetime import datetime
import ijson

load_dotenv()

class S3Storage:

    def __init__(self):
        self.bucket = os.getenv("AWS_S3_BUCKET_NAME")
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )

    def save_json(self, docs: list[dict], filename: str) -> str:
        json_content = json.dumps(docs, ensure_ascii=False, indent=2)

        self.s3.put_object(
            Bucket=self.bucket,
            Key=filename,
            Body=json_content.encode("utf-8"),
            ContentType="application/json"
        )

        return f"s3://{self.bucket}/{filename}"
    
    def delete_file(self, filename: str):
        self.s3.delete_object(Bucket=self.bucket, Key=filename)

    def download_json(self, filename: str) -> list[dict]:
        response = self.s3.get_object(Bucket=self.bucket, Key=filename)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)

    def upload_file(self, local_path: str, s3_key: str) -> str:
        self.s3.upload_file(
            Filename=local_path,
            Bucket=self.bucket,
            Key=s3_key,
            ExtraArgs={"ContentType": "application/json"}
        )
        return f"s3://{self.bucket}/{s3_key}"
    
    def save_json_content(self, json_content: str, s3_key: str):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=json_content.encode("utf-8"),
            ContentType="application/json"
        )

    def stream_json(self, key: str):
        """
        ✅ CHANGED:
        - Stream JSON array từ S3
        - Không load toàn bộ object
        """
        if not self.bucket:
            raise ValueError("S3 bucket not set")

        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        body = obj["Body"]

        # item = từng phần tử trong JSON array
        for item in ijson.items(body, "item"):
            yield item

    def list_keys(self, prefix: str) -> list[str]:
        """
        List toàn bộ object key dưới prefix
        """
        paginator = self.s3.get_paginator("list_objects_v2")
        keys = []

        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=prefix
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/"):
                    keys.append(key)

        return keys
