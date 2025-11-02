from datetime import datetime, timezone
import uuid

class ArticleDocument:
    """Tạo tài liệu JSON chuẩn theo cấu trúc yêu cầu."""
    def __init__(self, article: dict):
        self.article = article or {}
        self.now = datetime.now(timezone.utc).isoformat()

    def to_dict(self):
        a = self.article
        return {
            "id": f"doc_{self.now[:10].replace('-', '')}_{a.get('id', 'unknown')}_" + uuid.uuid4().hex,
            "source": {
                "url": a.get("url", ""),
                "domain": a.get("domain", ""),
                "crawl_date": self.now,
            },
            "metadata": {
                "title": a.get("title", ""),
                "author": a.get("author", "Không rõ"),
                "category": "news",
                "tags": [],
                "language": "vi",
                "content_type": "article",
                "version": 1,
                "published_time": a.get("published_time", ""),
            },
            "content": {
                "raw_html": a.get("raw_html", None),
                "text": a.get("text", ""),
                "summary": "",
                "images": a.get("images", []),
            },
            "status": {
                "parsed": True,
                "cleaned": False,
                "embedded": False,
                "last_updated": self.now,
            },
        }