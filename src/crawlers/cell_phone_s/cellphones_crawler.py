import json
import datetime
from crawlers.cell_phone_s.cellphones_parser import CellphonesSParser
from crawlers.core.article_document import ArticleDocument
from storage.s3_storage import S3Storage


class CellphonesSCrawler:
    """Điều phối crawl CellphonesS theo từng lần click và upload từng batch lên S3"""

    def __init__(
        self,
        parser: CellphonesSParser = None,
        s3_storage=None,
        max_clicks: int = 20,
        wait_time: float = 1.0,
    ):
        self.parser = parser or CellphonesSParser(
            max_clicks=max_clicks,
            wait_time=wait_time
        )
        self.s3_storage = s3_storage or S3Storage()

        # ✅ Tạo folder logical trên S3 theo timestamp
        self.s3_folder = datetime.datetime.now().strftime("cellphones_%Y%m%d_%H%M%S")

    # -------------------------------
    # Main run
    # -------------------------------

    def run(self, skip_images: bool = True):
        print("🟢 Starting CellphonesS crawler (NO LOCAL SAVE)")
        print(f"📦 S3 folder: output/{self.s3_folder}/")

        batch_index = 1

        for batch_articles in self.parser.fetch_articles_by_click():
            print(f"🆕 Batch {batch_index}: {len(batch_articles)} articles")

            detailed_articles = self.parser.fetch_all_details(
                batch_articles, skip_images=skip_images
            )

            docs = self.build_documents(detailed_articles)

            if not docs:
                print("⚠️ Batch rỗng, bỏ qua")
                continue

            filename = f"cellphones_batch_{str(batch_index).zfill(3)}.json"

            # ✅ Convert JSON trong memory
            json_content = json.dumps(docs, ensure_ascii=False, indent=2)

            # ✅ Upload trực tiếp lên S3, KHÔNG FILE LOCAL
            if self.s3_storage:
                s3_key = f"output/{self.s3_folder}/{filename}"
                try:
                    self.s3_storage.save_json_content(json_content, s3_key)
                    print(f"☁️ Uploaded to S3 → {s3_key} at {datetime.datetime.now().isoformat()}")
                except Exception as e:
                    print(f"❌ Upload S3 lỗi: {e}")
            else:
                print("⚠️ Chưa cấu hình s3_storage")

            batch_index += 1

    # -------------------------------
    # Helper
    # -------------------------------

    def build_documents(self, articles: list[dict]) -> list[dict]:
        docs = []
        for art in articles:
            doc = ArticleDocument.from_raw({
                "id": None,
                "url": art.get("url"),
                "title": art.get("title"),
                "author": art.get("author", "Không rõ"),
                "thumbnail": None,
                "domain": art.get("domain", "cellphones.com.vn"),
                "published_time": art.get("published_time"),
                "raw_html": art.get("raw_html"),
                "text": art.get("text", ""),
                "images": art.get("images", []),
            }).to_dict()
            docs.append(doc)
        return docs
