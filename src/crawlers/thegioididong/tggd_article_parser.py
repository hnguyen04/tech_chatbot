import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from crawlers.thegioididong.tggd_config import TGGDConfig

class TGGDArticleParser:
    """Parse HTML danh sách và chi tiết bài viết nhanh hơn (đa luồng)."""

    def __init__(self, config: TGGDConfig = TGGDConfig()):
        self.config = config

    def parse_list(self, html: str) -> list[dict]:
        """Parse danh sách bài viết từ HTML."""
        soup = BeautifulSoup(html, "lxml")
        articles = []

        for li in soup.select("ul.newslist li"):
            try:
                a_tag = li.select_one("a")
                if not a_tag:
                    continue

                title_tag = a_tag.select_one("h3.titlecom")
                if not title_tag:
                    continue

                title = title_tag.get_text(strip=True)
                href = a_tag.get("href", "")
                full_url = href if href.startswith("http") else f"{self.config.BASE_URL}{href}"

                author_tag = li.select_one(".user-info span")
                author = author_tag.get_text(strip=True) if author_tag else "Không rõ"

                spans = li.select(".timepost span")
                published_time = spans[-1].get_text(strip=True) if spans else ""

                raw_article = {
                    "id": li.get("data-id"),
                    "title": title,
                    "url": full_url,
                    "domain": self.config.DOMAIN,
                    "author": author,
                    "thumbnail": None,
                    "published_time": published_time,
                }

                articles.append(raw_article)
            except Exception as e:
                print(f"⚠️ Parse error in list: {e}")

        return articles

    def fetch_article_detail(self, article: dict, skip_images=True) -> dict:
        """Mở URL thật của bài viết và trích text (bỏ ảnh nếu skip_images=True)."""
        url = article.get("url")
        if not url:
            return article

        try:
            resp = requests.get(url, headers=self.config.DETAILED_API_HEADERS, timeout=10)
            if resp.status_code != 200:
                print(f"⚠️ HTTP {resp.status_code} khi truy cập {url}")
                return article

            soup = BeautifulSoup(resp.text, "lxml")
            article_tag = soup.select_one("article")
            if not article_tag:
                return article

            paragraphs = [p.get_text(strip=True) for p in article_tag.select("p") if p.get_text(strip=True)]
            article["text"] = "\n".join(paragraphs)
            article["raw_html"] = str(article_tag)

            if not skip_images:
                article["images"] = [img.get("src") for img in article_tag.select("img") if img.get("src")]

            title_tag = article_tag.select_one("h1.titledetail")
            if title_tag:
                article["title"] = title_tag.get_text(strip=True)

            user_tag = article_tag.select_one(".userdetail div")
            if user_tag:
                article["author"] = user_tag.get_text(strip=True).replace("Đóng góp bởi", "").strip()

        except Exception as e:
            print(f"⚠️ Lỗi khi parse chi tiết {url}: {e}")

        return article

    def fetch_all_details(self, articles: list[dict], max_workers: int = 8, skip_images=True) -> list[dict]:
        """Tải chi tiết bài viết song song."""
        detailed_articles = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.fetch_article_detail, art, skip_images) for art in articles]
            for future in as_completed(futures):
                try:
                    detailed_articles.append(future.result())
                except Exception as e:
                    print(f"⚠️ Worker error: {e}")
        return detailed_articles