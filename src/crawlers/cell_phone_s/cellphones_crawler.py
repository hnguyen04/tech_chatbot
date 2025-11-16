import os
import json
from crawlers.cell_phone_s.cellphones_parser import CellphonesSParser
from crawlers.core.article_document import ArticleDocument


class CellphonesSCrawler:
    """Điều phối crawl CellphonesS & tạo ArticleDocument."""

    def __init__(
        self,
        parser: CellphonesSParser = None,
        max_clicks: int = 3,
        wait_time: float = 1.0,
        output_dir: str = "output",
    ):
        self.parser = parser or CellphonesSParser(max_clicks=max_clicks, wait_time=wait_time)
        self.max_clicks = max_clicks
        self.wait_time = wait_time
        self.output_dir = output_dir

    # -------------------------------
    # Orchestrate
    # -------------------------------

    def run(self, output_file="cellphones_s_articles.json", skip_images: bool = True):
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"🟢 Starting crawl CellphonesS with max_clicks={self.max_clicks} ...")

        raw_articles = self.parser.fetch_articles()
        print(f"✅ Fetched {len(raw_articles)} articles")

        detailed_articles = self.parser.fetch_all_details(raw_articles, skip_images=skip_images)
        print(f"✅ Fetched details for {len(detailed_articles)} articles")

        all_docs = self.build_documents(detailed_articles)

        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(all_docs)} articles → {output_path}")

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
    