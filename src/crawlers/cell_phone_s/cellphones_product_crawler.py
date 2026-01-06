from crawlers.cell_phone_s.cellphones_product_parser import CellphonesProductSParser
from crawlers.cell_phone_s.cellphones_config import CellphoneSConfig
from storage.s3_storage import S3Storage
from urllib.parse import urlparse


import os
import math
import datetime
import json


class CellphonesProductCrawler:
    """
    Điều phối crawl CellphonesS
    - Parser lo Selenium
    - Crawler lo batch
    - Ưu tiên upload S3, lỗi thì fallback local
    """

    def __init__(
        self,
        batch_size: int = 10,
        output_dir: str = "output/cellphones_products",
        max_clicks: int = 5,
        wait_time: float = 1.0,
        s3_storage: S3Storage | None = None,
    ):
        self.parser = CellphonesProductSParser(
            max_clicks=max_clicks,
            wait_time=wait_time,
            config=CellphoneSConfig(),
        )
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.max_clicks = max_clicks

        self.s3_storage = s3_storage or S3Storage()
        self.s3_folder = datetime.datetime.now().strftime(
            "cellphones_products_%Y%m%d_%H%M%S"
        )

        os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Dump helpers
    # -----------------------------

    def _dump_local(self, batch: list, filename: str):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [d.to_dict() for d in batch],
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"📁 Fallback local saved -> {path}")

    def _dump_s3(self, batch: list, filename: str) -> bool:
        if not self.s3_storage:
            return False

        try:
            json_content = json.dumps(
                [d.to_dict() for d in batch],
                ensure_ascii=False,
                indent=2,
            )
            s3_key = f"output/{self.s3_folder}/{filename}"
            self.s3_storage.save_json_content(json_content, s3_key)
            print(f"☁️ Uploaded to S3 -> {s3_key}")
            return True
        except Exception as e:
            print(f"⚠️ S3 upload failed: {e}")
            return False

    def _dump_batches(self, docs: list, prefix: str):
        total_batches = math.ceil(len(docs) / self.batch_size)

        for i in range(total_batches):
            batch = docs[i * self.batch_size:(i + 1) * self.batch_size]
            filename = f"{prefix}_batch_{i+1}.json"

            # ưu tiên S3
            uploaded = self._dump_s3(batch, filename)

            # fallback local
            if not uploaded:
                self._dump_local(batch, filename)

    # -----------------------------
    # Utils
    # -----------------------------

    def extract_category_from_url(self, url: str) -> str:
        """
        Rule:
        - /mobile/apple.html        -> mobile
        - /do-choi-cong-nghe.html   -> do-choi-cong-nghe
        - /linh-kien.html           -> linh-kien
        """

        try:
            path = urlparse(url).path.strip("/")

            # case: mobile/apple.html
            if "/" in path:
                return path.split("/")[0]

            # case: linh-kien.html
            return path.replace(".html", "")

        except Exception:
            return "unknown"

    # -----------------------------
    # Main
    # -----------------------------

    def run(self):
        for url in self.parser.config.PRODUCT_URL_LIST:
            slug = url.rstrip("/").split("/")[-1].replace(".html", "")
            category = self.extract_category_from_url(url)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            print(f"🔍 Crawling {slug} (category={category})")

            docs = self.parser.fetch_all_products(url, category)

            self._dump_batches(docs, f"{slug}_{ts}")

            print(f"✅ Done {len(docs)} products from {slug}")
