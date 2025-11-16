import os
import json
from crawlers.thegioididong.tggd_article_parser import TGGDArticleParser
from crawlers.thegioididong.tggd_api_client import TGGDApiClient
from crawlers.core.article_document import ArticleDocument


class TGGDCrawler:

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

    # -------------------------------
    # Orchestrate
    # -------------------------------

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        all_docs = []

        for page in range(self.start_index, self.end_index + 1):
            print(f"📡 Crawling page {page} ...")
            page_docs = self.parse_and_download(page)
            all_docs.extend(page_docs)

        output_file = self.save_output(all_docs)
        print(f"💾 Saved {len(all_docs)} documents → {output_file}")

    # -------------------------------
    # Steps
    # -------------------------------

    def parse_and_download(self, page: int) -> list[dict]:
        html = self.api_client.fetch_articles_html(page)
        articles = self.parser.parse_list(html)
        print(f"  ✔ Found {len(articles)} items")

        detailed = self.parser.fetch_all_details(articles, max_workers=8, skip_images=True)
        return self.build_documents(detailed)

    def build_documents(self, articles: list[dict]) -> list[dict]:
        docs = []
        for art in articles:
            doc = ArticleDocument.from_raw(art).to_dict()
            docs.append(doc)
        return docs

    def save_output(self, docs: list[dict]) -> str:
        path = os.path.join(self.output_dir, "tgdd_articles.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        return path
