import sys
import os
import time
import pandas as pd
import json


# Thêm src vào sys.path trước tất cả import
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from crawlers.thegioididong.tggd_crawler import TGGDCrawler
from crawlers.thegioididong.tggd_api_client import TGGDApiClient
from crawlers.thegioididong.tggd_article_parser import TGGDArticleParser
from crawlers.thegioididong.tggd_product_crawler import TGDDProductCrawler
from crawlers.cell_phone_s.cellphones_crawler import CellphonesSCrawler
from crawlers.cell_phone_s.cellphones_config import CellphoneSConfig

class CrawlRunner:
    """Điều phối chạy nhiều crawler khác nhau."""
    def __init__(self):
        self.crawlers = []

    def register(self, crawler):
        self.crawlers.append(crawler)

    def run_all(self):
        total = len(self.crawlers)

        for i, crawler in enumerate(self.crawlers, start=1):
            crawler_name = crawler.__class__.__name__
            print(f"🚀 Running {crawler_name} ({i}/{total}) ...")

            crawler_start = time.time()
            crawler.run()
            crawler_end = time.time()

            elapsed = crawler_end - crawler_start

            print(f"⏱ Finished {crawler_name} in {elapsed:.1f}s")
            print("-" * 40)

