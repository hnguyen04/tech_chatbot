import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse
import unicodedata
from typing import Set



def normalize_url(url: str) -> str | None:
    """Strip query params/fragments and ensure scheme exists."""
    if not url or not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if not parsed.scheme:
        parsed = parsed._replace(scheme="https")
    cleaned = parsed._replace(query="", fragment="")
    return urlunparse(cleaned)


def clean_price(price: str | None) -> str | None:
    """Remove currency text/symbols and keep digits only."""
    if not price:
        return None
    digits = re.findall(r"\d+", price)
    if not digits:
        return None
    return "".join(digits)


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r", " ").replace("\t", " ")
    cleaned = cleaned.replace("\n", " ")
    # Ensure punctuation is followed by a space if missing.
    cleaned = re.sub(r"([.,;:!?])(\S)", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def parse_relative_time(value: str | None, now: datetime | None = None) -> datetime | None:
    """Parse strings like '5 ngay truoc', '2 gio truoc', '1 tuan truoc'."""
    if not value or not isinstance(value, str):
        return None
    now = now or datetime.now(timezone.utc)
    lowered = value.lower()
    patterns = [
        (r"(\d+)\s*ngay", "days"),
        (r"(\d+)\s*gio", "hours"),
        (r"(\d+)\s*tuan", "weeks"),
        (r"(\d+)\s*phut", "minutes"),
    ]
    for pattern, unit in patterns:
        m = re.search(pattern, lowered)
        if m:
            amount = int(m.group(1))
            delta_kwargs = {unit: amount}
            return now - timedelta(**delta_kwargs)
    return None

def extract_raw_units(value: str) -> Set[str]:
    """
    Trích raw unit hoàn toàn data-driven
    - KHÔNG dùng predefined unit list
    - Unit = phần còn lại sau khi loại số
    - Giữ nguyên chữ hoa / ký hiệu
    """

    NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:[\.,]\d+)*")

    if not value or not isinstance(value, str):
        return set()

    units = set()

    # Tách theo khoảng trắng và các dấu phân cách thường gặp
    tokens = re.split(r"[\s~–\-]+", value)

    for token in tokens:
        if not NUMBER_PATTERN.search(token):
            continue

        # Bỏ phần số
        unit = NUMBER_PATTERN.sub("", token)

        # Bỏ ký tự bao ngoài
        unit = unit.strip("()[]{}×x*/≤>=:")

        if not unit:
            continue

        # Bỏ token toàn dấu vô nghĩa
        if all(c in ".,%" for c in unit):
            continue

        units.add(unit)

    return units

def normalize_text(text: str) -> str:
    """
    Normalize nhẹ:
    - Cắt mọi thứ sau dấu :
    - Strip space
    - Giữ nguyên chữ hoa, dấu
    """
    if not text:
        return ""

    return text.split(":", 1)[0].strip()

