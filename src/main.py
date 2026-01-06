from crawlers.cell_phone_s.cellphones_product_crawler import CellphonesProductCrawler
from crawlers.core.crawl_runners import CrawlRunner


if __name__ == "__main__":
    crawl_runner = CrawlRunner()
    cellphone_product_crawler = CellphonesProductCrawler(   
            max_clicks=100,
            wait_time=1.0,
        )
    crawl_runner.register(cellphone_product_crawler)

    crawl_runner.run_all()