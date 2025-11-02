import os
import json
from crawlers.thegioididong.tggd_article_parser import TGGDArticleParser
from crawlers.thegioididong.tggd_api_client import TGGDApiClient 
from crawlers.core.article_document import ArticleDocument  


class TGGDCrawler:
    """Điều phối toàn bộ quá trình crawl dữ liệu từ thegioididong."""

    def __init__(
        self,
        api_client: TGGDApiClient = None,
        parser: TGGDArticleParser = None,
        start_index: int = 1,
        end_index: int = 3,
        output_dir: str = "output"
    ):
        self.api_client = api_client or TGGDApiClient()
        self.parser = parser or TGGDArticleParser()
        self.start_index = start_index
        self.end_index = end_index
        self.output_dir = output_dir

    def run(self):
        """Thực thi quá trình crawl."""
        os.makedirs(self.output_dir, exist_ok=True)
        all_docs = []

        for i in range(self.start_index, self.end_index + 1):
            print(f"📡 Fetching list page {i} ...")
            html = self.api_client.fetch_articles_html(i)
            articles = self.parser.parse_list(html)
            print(f"✅ Found {len(articles)} articles on page {i}")

            print("🔍 Fetching article details (multi-thread)...")
            detailed_articles = self.parser.fetch_all_details(
                articles, max_workers=8, skip_images=True
            )

            for art in detailed_articles:
                doc = ArticleDocument(art).to_dict()
                all_docs.append(doc)

        output_path = os.path.join(self.output_dir, "tgdd_articles.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved {len(all_docs)} docs → {output_path}")