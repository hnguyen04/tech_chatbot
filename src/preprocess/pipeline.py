from datetime import datetime, timezone
from typing import Iterable, List
from preprocess.cleaners import (
    normalize_url,
    clean_price,
    clean_text,
    parse_relative_time,
)
from preprocess.models import RawRecord, CleanRecord, SpecItem
from preprocess.llm_placeholders import TitleLLMEnricher, SpecsLLMEnricher
from preprocess.db_writer import PostgresWriter
import re
import unicodedata

class PreprocessPipeline:
    """
    ✅ STREAMING PIPELINE
    - Không load toàn bộ JSON
    - Batch nhỏ cho LLM
    - Xử lý hết mọi bản ghi
    """

    def __init__(
        self,
        title_enricher: TitleLLMEnricher | None = None,
        specs_enricher: SpecsLLMEnricher | None = None,
        db_writer: PostgresWriter | None = None,
        batch_size: int = 32, 
    ):
        self.title_enricher = title_enricher or TitleLLMEnricher()
        self.specs_enricher = specs_enricher or SpecsLLMEnricher()
        self.db_writer = db_writer or PostgresWriter()
        self.batch_size = batch_size
        self.spec_key_registry = self.db_writer.load_spec_key_registry()
        self.unit_registry = self.db_writer.load_unit_registry()

    def run(self, raw_docs: Iterable[dict]):
        buffer: List[CleanRecord] = []

        for doc in raw_docs:
            raw = self._to_raw(doc)
            clean = self._clean_record(raw)
            if not clean:
                continue

            buffer.append(clean)

            # ✅ CHANGED: xử lý theo batch
            if len(buffer) >= self.batch_size:
                self._process_batch(buffer)
                buffer.clear()

        if buffer:
            self._process_batch(buffer)

    def _process_batch(self, records: List[CleanRecord]):
        records = self._filter_existing(records)
        if not records:
            return

        # ✅ batch nhỏ → không vượt token Gemini
        title_results = self.title_enricher.run_batch(records)
        product_ids = self.db_writer.insert_products(records, title_results)
        print(f"Successfully inserted {len(product_ids)} products title enriched.")
        
        for pid, rec in zip(product_ids, records):
            print("Processing specs for product ID:", pid)
            if rec.content_type != "product" and not rec.product_blob:
                continue
            specs = self._normalize_specs_from_registry(rec)
            self.db_writer.insert_specs(pid, specs)
            print(f"Inserted specs for product ID {pid} ({len(specs)}).")

        print(f"Inserted batch: {len(product_ids)}")

    def _to_raw(self, doc: dict) -> RawRecord:
        content = doc.get("content", {})
        metadata = doc.get("metadata", {})

        return RawRecord(
            url=doc.get("source", {}).get("url") or doc.get("url") or metadata.get("url"),
            title=metadata.get("title") or doc.get("title") or "",
            category=metadata.get("category") or doc.get("category") or "",
            domain=doc.get("source", {}).get("domain") or doc.get("domain") or "",
            published_time=metadata.get("published_time") or doc.get("published_time"),
            crawl_date=doc.get("source", {}).get("crawl_date") or doc.get("crawl_date"),
            price=metadata.get("price") or doc.get("price"),
            content_text=content.get("text") or doc.get("text"),
            images=content.get("images") or doc.get("images") or [],
            content_type=metadata.get("content_type") or doc.get("content_type") or "article",
            product_blob=content.get("product") or doc.get("product"),
        )

    def _clean_record(self, rec: RawRecord) -> CleanRecord | None:
        normalized_url = normalize_url(rec.url)
        if not normalized_url or not rec.title or not rec.category:
            return None

        crawl_dt = self._resolve_datetime(rec)

        content_type = rec.content_type
        if rec.product_blob and content_type == "article":
            content_type = "product"

        title = rec.title
        if content_type == "product":
            title = f"San Pham: {title}"

        return CleanRecord(
            source_url=normalized_url,
            domain=rec.domain or normalized_url.split("/")[2],
            crawl_date=crawl_dt,
            title=title,
            category=rec.category,
            price=clean_price(rec.price),
            content_text=clean_text(rec.content_text),
            images=rec.images or [],
            content_type=content_type,
            product_blob=rec.product_blob,
        )

    def _resolve_datetime(self, rec: RawRecord):
        now = datetime.now(timezone.utc)

        if rec.content_type == "article":
            rel = parse_relative_time(rec.published_time, now)
            if rel:
                return rel
            try:
                return datetime.fromisoformat(rec.published_time)
            except Exception:
                return now

        if rec.crawl_date:
            try:
                return datetime.fromisoformat(rec.crawl_date)
            except Exception:
                pass

        return now

    def _filter_existing(self, records: List[CleanRecord]) -> List[CleanRecord]:
        urls = [r.source_url for r in records]
        placeholders = ",".join(["%s"] * len(urls))
        sql = f"SELECT source_url FROM products WHERE source_url IN ({placeholders})"

        try:
            self.db_writer.client.cur.execute(sql, urls)
            existing = {row[0] for row in self.db_writer.client.cur.fetchall()}
        except Exception:
            existing = set()

        return [r for r in records if r.source_url not in existing]
    
    def _normalize_key(self, key: str) -> str:
        if not key:
            return ""

        key = unicodedata.normalize("NFKC", key)
        key = key.lower().strip()

        # bỏ dấu :
        key = re.sub(r"[:：]+$", "", key)

        # gom whitespace
        key = re.sub(r"\s+", " ", key)

        return key

    def _extract_numbers_and_units(self, text: str):
        if not text:
            return [], []

        text_norm = text.lower()
        numbers = []
        units = []

        unit_patterns = sorted(
            self.unit_registry.keys(),
            key=len,
            reverse=True
        )
        unit_regex = "|".join(re.escape(u) for u in unit_patterns)

        pattern = re.compile(
            rf"([-+]?\d+(?:[.,]\d+)?)\s*({unit_regex})",
            flags=re.IGNORECASE
        )

        for match in pattern.finditer(text_norm):
            num_str, raw_unit = match.groups()

            try:
                value = float(num_str.replace(",", "."))
                numbers.append(value)
            except ValueError:
                continue

            units.append(self.unit_registry[raw_unit.lower()])

        return numbers, list(dict.fromkeys(units))  

    def _normalize_specs_from_registry(self, rec: CleanRecord) -> List[SpecItem]:
        if not rec.product_blob:
            return []

        items: List[SpecItem] = []

        for raw_key, raw_value in rec.product_blob.items():
            lookup = self._normalize_key(raw_key)
            entry = self.spec_key_registry.get(lookup)

            if not entry:
                continue

            numbers, units = self._extract_numbers_and_units(raw_value)

            items.append(
                SpecItem(
                    standardized_key=entry["standardized_key"],
                    standardized_key_eng=entry["standardized_key_eng"],
                    standardized_value=raw_value,
                    standardized_value_eng=raw_value,
                    category=entry["category"],
                    category_eng=entry["category_eng"],
                    numerical_value_list=numbers,
                    unit_list=units,
                )
            )

        return items