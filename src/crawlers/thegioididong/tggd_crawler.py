import os
import json
from crawlers.thegioididong.tggd_article_parser import TGGDArticleParser
from crawlers.thegioididong.tggd_api_client import TGGDApiClient
from crawlers.thegioididong.tggd_product_crawler import TGDDProductCrawler
from crawlers.core.article_document import ArticleDocument
from crawlers.thegioididong.tggd_config import TGGDConfig
from storage.s3_storage import S3Storage
import datetime


class TGGDCrawler:

    def __init__(
        self,
        config: TGGDConfig = None,
        api_client: TGGDApiClient = None,
        parser: TGGDArticleParser = None,
        should_crawl_products: bool = False,
        should_crawl_articles: bool = False,
        start_index: int = 1,
        end_index: int = 3,
        output_dir: str = datetime.datetime.now().strftime("output/tggd_%Y%m%d_%H%M%S"),
        products_crawl_limit: int = 100000,
        s3_storage: S3Storage = None,
    ):
        self.api_client = api_client or TGGDApiClient()
        self.parser = parser or TGGDArticleParser()
        self.start_index = start_index
        self.end_index = end_index
        self.output_dir = output_dir
        self.should_crawl_products = should_crawl_products
        self.config = config or TGGDConfig()
        self.should_crawl_articles = should_crawl_articles
        self.products_crawl_limit = products_crawl_limit
        self.s3_storage = s3_storage or S3Storage()

    # -------------------------------
    # Orchestrate
    # -------------------------------

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        all_products = []

        if self.should_crawl_articles == True:
            # Crawl articles
            for page in range(self.start_index, self.end_index + 1):
                print(f"📡 Crawling page {page} ...")
                try: 
                    page_docs = self.parse_and_download(page)
                    self.save_json(page_docs, f"tgdd_articles_page_{page}.json")
                    print(f"✅ Saved {len(page_docs)} articles from page {page}")
                except Exception as e:
                    print(f"❌ Error occurred while crawling page {page}: {e}")
                    pass



        # Crawl products if needed
        if self.should_crawl_products:
            print("🛒 Starting product crawl ...")
            for category in self.config.PRODUCT_CATEGORY_LIST:
                print(f"🛒 Crawling category: {category} ...")
                product_crawler = TGDDProductCrawler(category=category, crawl_limit=self.products_crawl_limit)
                products = product_crawler.run()  # danh sách dict hoặc ArticleDocument
                all_products.extend([doc.to_dict() for doc in products])

            products_file = self.save_json(all_products, "tgdd_products.json")
            print(f"✅ Saved {len(all_products)} products to {products_file}")
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
        return [ArticleDocument.from_raw(art).to_dict() for art in articles]

    def save_json(self, docs: list[dict], filename: str) -> str:
        if self.s3_storage:
            key = f"{self.output_dir}/{filename}"
            print(f"💾 Saving to S3: {key} ...")
            return self.s3_storage.save_json(docs, key)

        path = os.path.join(self.output_dir, filename)
        print(f"💾 Saving to local file: {path} ...")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        return path