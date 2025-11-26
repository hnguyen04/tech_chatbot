

from crawlers.cell_phone_s.cellphones_crawler import CellphonesSCrawler
from crawlers.core.crawl_runners import CrawlRunner


if __name__ == "__main__":
    crawl_runner = CrawlRunner()
    cellphones_crawler = CellphonesSCrawler(
        max_clicks=20, 
        wait_time=1.0
    )
    crawl_runner.register(cellphones_crawler)
    crawl_runner.run_all()