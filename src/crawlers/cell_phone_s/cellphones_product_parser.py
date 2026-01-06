import os
import re
import time
import json
import uuid
import datetime
from urllib.parse import urlparse
from typing import Optional, List

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from storage.s3_storage import S3Storage
from datetime import timezone

from crawlers.cell_phone_s.cellphones_config import CellphoneSConfig
from crawlers.core.article_document import (
    ArticleDocument,
    ArticleSource,
    ArticleMetadata,
    ArticleContent,
    ArticleStatus,
)

# =========================
# Parser
# =========================

class CellphonesProductSParser:
    """
    Parser CellphonesS
    - Giữ nguyên Selenium config
    - Crawl listing -> product url
    - Vào từng product page lấy:
        + Thông số kỹ thuật -> content.product
        + Nội dung mô tả -> content.text
    """

    PRICE_RE = re.compile(r"[^\d]")

    def __init__(
        self,
        max_clicks: int = 5,
        wait_time: float = 1.0,
        config: CellphoneSConfig | None = None,
    ):
        self.max_clicks = max_clicks
        self.wait_time = wait_time
        self.config = config
        self.max_worker = max(1, os.cpu_count() - 1)
        print(f"CellphonesProductSParser using {self.max_worker} workers and max_clicks={self.max_clicks}")

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")

        self.driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=chrome_options,
        )

    # ============================================================
    # Utils
    # ============================================================

    def _sleep(self):
        time.sleep(self.wait_time)

    def _clean_price(self, text: Optional[str]) -> Optional[int]:
        if not text:
            return None
        digits = self.PRICE_RE.sub("", text)
        return int(digits) if digits.isdigit() else None

    # ============================================================
    # Listing page
    # ============================================================

    def _click_all_show_more(self):
        clicks = 0
        while clicks < self.max_clicks:
            try:
                btn = WebDriverWait(self.driver, 3).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, ".button__show-more-product")
                    )
                )
                self.driver.execute_script("arguments[0].click();", btn)
                self._sleep()
                clicks += 1
            except Exception:
                break

    def _parse_listing_product(self, html: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")

        link = soup.select_one("a.product__link")
        name = soup.select_one(".product__name h3")
        price = soup.select_one(".product__price--show")
        old_price = soup.select_one(".product__price--through")

        url = link["href"] if link else ""
        domain = urlparse(url).netloc if url else ""

        return {
            "id": str(uuid.uuid4()),
            "url": url,
            "domain": domain,
            "title": name.text.strip() if name else "",
            "price": self._clean_price(price.text) if price else None,
            "original_price": self._clean_price(old_price.text) if old_price else None,
        }

    def _collect_listing_products(self, url: str) -> List[dict]:
        self.driver.get(url)
        self._sleep()
        self._click_all_show_more()

        elements = self.driver.find_elements(
            By.CSS_SELECTOR,
            ".product-info-container.product-item"
        )

        products = []
        for el in elements:
            try:
                products.append(
                    self._parse_listing_product(el.get_attribute("innerHTML"))
                )
            except Exception:
                continue

        return products

    # ============================================================
    # Product detail page
    # ============================================================

    def _parse_technical_specs(self, soup: BeautifulSoup) -> dict:
        specs = {}
        rows = soup.select("table.technical-content tr.technical-content-item")

        for row in rows:
            try:
                key = row.select_one("td:nth-child(1)").get_text(strip=True)
                value = row.select_one("td:nth-child(2)").get_text(" ", strip=True)
                specs[key] = value
            except Exception:
                continue

        return specs

    def _parse_product_text(self, soup: BeautifulSoup) -> str:
        block = soup.select_one("div.ksp-content")
        if not block:
            return ""

        return block.get_text("\n", strip=True)

    def _fetch_product_detail(self, product_url: str) -> dict:
        """
        Fetch chi tiết từng product
        Lỗi thì trả dict rỗng để không block luồng
        """
        try:
            self.driver.get(product_url)
            print(f"Fetching product detail: {product_url}")
            self._sleep()

            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            return {
                "specs": self._parse_technical_specs(soup),
                "text": self._parse_product_text(soup),
            }
        except Exception:
            return {"specs": {}, "text": ""}

    # ============================================================
    # Public API
    # ============================================================

    def fetch_all_products(self, url: str, category: str) -> List[ArticleDocument]:
        """
        Crawl full:
        listing -> từng product -> ArticleDocument
        """

        now = datetime.datetime.now(timezone.utc).isoformat()
        raw_products = self._collect_listing_products(url)

        documents: List[ArticleDocument] = []

        for raw in raw_products:
            detail = self._fetch_product_detail(raw["url"])

            doc = ArticleDocument(
                id=f"doc_{now[:10].replace('-', '')}_{raw['id']}_{uuid.uuid4().hex}",
                source=ArticleSource(
                    url=raw["url"],
                    domain=raw["domain"],
                    crawl_date=now,
                ),
                metadata=ArticleMetadata(
                    title=raw["title"],
                    author="CellphoneS",
                    published_time="",
                    category=category,
                    tags=[],
                    language="vi",
                    content_type="product",
                    version=1,
                    price=raw["price"] or raw["original_price"],
                ),
                content=ArticleContent(
                    raw_html=None,
                    text=detail["text"],
                    summary="",
                    images=[],
                    product=detail["specs"] or None,
                ),
                status=ArticleStatus(
                    parsed=True,
                    cleaned=False,
                    embedded=False,
                    last_updated=now,
                ),
            )

            documents.append(doc)

        return documents
