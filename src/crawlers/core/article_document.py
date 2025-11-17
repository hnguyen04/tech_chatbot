from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
import uuid


@dataclass
class ArticleSource:
    url: str
    domain: str
    crawl_date: str


@dataclass
class ArticleMetadata:
    title: str
    author: str
    category: str = "news"
    tags: List[str] = field(default_factory=list)
    language: str = "vi"
    content_type: str = "article"
    version: int = 1
    published_time: str = ""
    price: Optional[str] = None


@dataclass
class ArticleContent:
    raw_html: Optional[str]
    text: str
    summary: str = ""
    images: List[str] = field(default_factory=list)
    product: Optional[dict] = None


@dataclass
class ArticleStatus:
    parsed: bool = True
    cleaned: bool = False
    embedded: bool = False
    last_updated: str = ""


@dataclass
class ArticleDocument:
    id: str
    source: ArticleSource
    metadata: ArticleMetadata
    content: ArticleContent
    status: ArticleStatus

    @staticmethod
    def from_raw(article: dict) -> "ArticleDocument":
        now = datetime.now(timezone.utc).isoformat()
        doc_id = (
            f"doc_{now[:10].replace('-', '')}_{article.get('id', 'unknown')}_{uuid.uuid4().hex}"
        )

        return ArticleDocument(
            id=doc_id,
            source=ArticleSource(
                url=article.get("url", ""),
                domain=article.get("domain", ""),
                crawl_date=now,
            ),
            metadata=ArticleMetadata(
                title=article.get("title", ""),
                author=article.get("author", "Không rõ"),
                published_time=article.get("published_time", ""),
                category=article.get("category", "news"),
                tags=article.get("tags", []),
                language=article.get("language", "vi"),
                content_type=article.get("content_type", "article"),
                version=article.get("version", 1),
            ),
            content=ArticleContent(
                raw_html=article.get("raw_html"),
                text=article.get("text", ""),
                images=article.get("images", []),
                summary=article.get("summary", ""),
            ),
            status=ArticleStatus(
                parsed=True,
                cleaned=article.get("status", {}).get("cleaned", False),
                embedded=article.get("status", {}).get("embedded", False),
                last_updated=now,
            ),
        )

    def to_dict(self):
        """Chuyển sang JSON dict chuẩn."""
        return {
            "id": self.id,
            "source": self.source.__dict__,
            "metadata": self.metadata.__dict__,
            "content": self.content.__dict__,
            "status": self.status.__dict__,
        }
