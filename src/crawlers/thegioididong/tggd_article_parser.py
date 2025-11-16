import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from crawlers.thegioididong.tggd_config import TGGDConfig


class TGGDArticleParser:
    """Parser HTML danh sách & chi tiết."""

    def __init__(self, config: TGGDConfig = TGGDConfig()):
        self.config = config

    # -------------------------------
    # Parse list
    # -------------------------------

    def parse_list(self, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "lxml")
        items = soup.select("ul.newslist li")

        return [self.parse_list_item(li) for li in items if self.parse_list_item(li)]

    def parse_list_item(self, li):
        """Parse 1 item trong trang danh sách."""
        try:
            a_tag = li.select_one("a")
            title_tag = li.select_one("h3.titlecom")
            if not a_tag or not title_tag:
                return None

            url = a_tag.get("href", "")
            full_url = url if url.startswith("http") else f"{self.config.BASE_ARTICLE_URL}{url}"

            published_time = ""
            spans = li.select(".timepost span")
            if spans:
                published_time = spans[-1].get_text(strip=True)

            author_tag = li.select_one(".user-info span")

            return {
                "id": li.get("data-id"),
                "title": title_tag.get_text(strip=True),
                "url": full_url,
                "domain": self.config.DOMAIN,
                "author": author_tag.get_text(strip=True) if author_tag else "Không rõ",
                "published_time": published_time,
                "thumbnail": None,
                "category": "news",
                "tags": [],
                "language": "vi",
                "content_type": "article",
            }

        except Exception:
            return None

    # -------------------------------
    # Parse detail
    # -------------------------------

    def fetch_article_detail(self, article: dict, skip_images=True) -> dict:
        """Tải trang chi tiết """
        url = article.get("url")
        if not url:
            return article
        
        try:
            resp = requests.get(url, headers=self.config.DETAILED_API_HEADERS, timeout=10)
            if resp.status_code != 200:
                return article
                        
            return self.apply_detail_html(article, resp.text, skip_images)

        except Exception:
            return article

    def apply_detail_html(self, article: dict, html: str, skip_images=True) -> dict:
        """Parse HTML chi tiết vào article."""
        soup = BeautifulSoup(html, "lxml")
        article_tag = soup.select_one("article")
        if not article_tag:
            return article

        article["raw_html"] = str(article_tag)
        article["text"] = "\n".join(
            p.get_text(strip=True)
            for p in article_tag.select("p")
            if p.get_text(strip=True)
        )

        if not skip_images:
            article["images"] = [
                img.get("src")
                for img in article_tag.select("img")
                if img.get("src")
            ]

        title_tag = article_tag.select_one("h1.titledetail")
        if title_tag:
            article["title"] = title_tag.get_text(strip=True)

        user_tag = article_tag.select_one(".userdetail div")
        if user_tag:
            article["author"] = (
                user_tag.get_text(strip=True)
                .replace("Đóng góp bởi", "")
                .strip()
            )
        
        return article

    # -------------------------------
    # Multi-thread loader
    # -------------------------------

    def fetch_all_details(self, articles: list[dict], max_workers=8, skip_images=True) -> list[dict]:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.fetch_article_detail, art, skip_images)
                for art in articles
            ]
            return [f.result() for f in as_completed(futures)]
