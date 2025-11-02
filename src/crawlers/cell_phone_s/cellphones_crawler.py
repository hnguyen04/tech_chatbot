import os
import json
from crawlers.cell_phone_s.cellphones_parser import CellphonesSParser
from crawlers.core.article_document import ArticleDocument

class CellphonesSCrawler:
    """Điều phối crawl CellphonesS và tạo ArticleDocument."""

    def __init__(
        self,
        parser: CellphonesSParser = None,
        max_clicks: int = 3,
        wait_time: float = 1.0,
        output_dir: str = "output"
    ):
        # Nếu chưa có parser, khởi tạo mặc định
        self.parser = parser or CellphonesSParser(max_clicks=max_clicks, wait_time=wait_time)
        self.max_clicks = max_clicks
        self.wait_time = wait_time
        self.output_dir = output_dir

    def run(self, output_file="cellphones_s_articles.json", max_workers: int = 8, skip_images: bool = True):
        """Thực thi crawl, fetch chi tiết, tạo ArticleDocument và lưu JSON."""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"🟢 Starting crawl CellphonesS with max_clicks={self.max_clicks} ...")

        # 1. Lấy tất cả URL + title
        raw_articles = self.parser.fetch_articles()
        print(f"✅ Fetched {len(raw_articles)} articles")

        # 2. Lấy chi tiết từng bài (title, author, published_time, text, raw_html)
        detailed_articles = self.parser.fetch_all_details(
            raw_articles, max_workers=max_workers, skip_images=skip_images
        )
        print(f"✅ Fetched details for {len(detailed_articles)} articles")

        # 3. Tạo ArticleDocument
        all_docs = []
        for art in detailed_articles:
            doc = ArticleDocument({
                "id": None,
                "url": art.get("url"),
                "title": art.get("title"),
                "author": art.get("author", "Không rõ"),
                "thumbnail": None,
                "domain": art.get("domain", self.parser.config.DOMAIN),
                "published_time": art.get("published_time"),
                "raw_html": art.get("raw_html"),
                "text": art.get("text", ""),
                "images": art.get("images", []),
            }).to_dict()
            all_docs.append(doc)

        # 4. Lưu ra file JSON
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved {len(all_docs)} articles → {output_path}")
