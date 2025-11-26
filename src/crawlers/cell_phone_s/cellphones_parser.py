import time
import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor, as_completed
from crawlers.cell_phone_s.cellphones_config import CellphoneSConfig


class CellphonesSParser:
    """Parser CellphonesS: crawl theo batch mỗi lần click."""

    def __init__(
        self,
        max_clicks: int = 5,
        wait_time: float = 1.0,
        config: CellphoneSConfig = CellphoneSConfig(),
    ):
        self.max_clicks = max_clicks
        self.wait_time = wait_time
        self.config = config
        self.max_worker = max(1, os.cpu_count() - 1)

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        self.driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=chrome_options,
        )

    # -------------------------------
    # Fetch theo click
    # -------------------------------

    def fetch_articles_by_click(self):
        """
        Mỗi lần click yield ra list article mới xuất hiện
        """
        self.driver.get(self.config.ARTICLE_BASE_URL)
        time.sleep(self.wait_time)

        seen_urls = set()

        for i in range(self.max_clicks):
            print(f"👉 Click {i+1}")

            try:
                btn = self.driver.find_element(By.XPATH, '//button[span[text()="Xem thêm"]]')
                self.driver.execute_script("arguments[0].click();", btn)
                time.sleep(self.wait_time)
            except Exception:
                print("✅ Không còn nút Xem thêm")
                break

            anchors = self.driver.find_elements(By.XPATH, '//a[starts-with(@href,"/sforum/")]')

            new_articles = []

            for a in anchors:
                url = a.get_attribute("href")
                title = a.get_attribute("title") or a.text

                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)
                new_articles.append({
                    "url": url,
                    "title": title,
                    "category": "news"
                })

            if new_articles:
                yield new_articles

        self.driver.quit()

    # -------------------------------
    # Fetch detail
    # -------------------------------

    def fetch_article_detail(self, article: dict) -> dict | None:
        url = article.get("url")
        if not url or self.config.AUTHOR_URL_PREFIX in url:
            return None

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "lxml")
            article_tag = soup.find("article")
            if not article_tag:
                return None

            title_tag = article_tag.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else article.get("title")

            author_tag = article_tag.select_one(f'a[href^="{self.config.AUTHOR_URL_PREFIX}"]')
            author = author_tag.get_text(strip=True) if author_tag else "Không rõ"

            date_tag = article_tag.find("span", string=lambda x: x and "Ngày cập nhật:" in x)
            published_time = date_tag.get_text(strip=True).replace("Ngày cập nhật:", "").strip() if date_tag else None

            paragraphs = [p.get_text(strip=True) for p in article_tag.find_all("p") if p.get_text(strip=True)]
            text = "\n".join(paragraphs)

            article.update({
                "title": title,
                "author": author,
                "published_time": published_time,
                "text": text,
                "raw_html": str(article_tag),
                "images": [],
                "domain": "cellphones.com.vn",
            })

            return article

        except Exception as e:
            print(f"⚠️ Lỗi chi tiết {url}: {e}")
            return None

    def fetch_all_details(self, articles: list[dict], skip_images=True):
        detailed_articles = []
        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(self.fetch_article_detail, art) for art in articles]
            for f in as_completed(futures):
                result = f.result()
                if result:
                    detailed_articles.append(result)
        return detailed_articles
