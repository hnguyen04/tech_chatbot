from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class RawRecord:
    url: str
    title: str
    category: str
    domain: str
    published_time: Optional[str] = None
    crawl_date: Optional[str] = None
    price: Optional[str] = None
    content_text: Optional[str] = None
    images: List[str] = field(default_factory=list)
    content_type: str = "article"  # article | product
    product_blob: Optional[dict] = None  # specs dict if product


@dataclass
class CleanRecord:
    source_url: str
    domain: str
    crawl_date: datetime
    title: str
    category: str
    price: Optional[str]
    content_text: str
    images: List[str]
    content_type: str
    product_blob: Optional[dict] = None


@dataclass
class TitleLLMResult:
    brand: Optional[str] = None
    model: Optional[str] = None
    product_line: Optional[str] = None
    full_title_eng: Optional[str] = None
    category_eng: Optional[str] = None
    llm_processed: bool = False


@dataclass
class SpecItem:
    standardized_key: str
    standardized_key_eng: str
    standardized_value: str
    standardized_value_eng: str
    category: str
    category_eng: str
    numerical_value_list: List[float] = field(default_factory=list)
    unit_list: List[str] = field(default_factory=list)
