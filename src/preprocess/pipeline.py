import json
from datetime import datetime, timezone
from typing import List
from preprocess.cleaners import normalize_url, clean_price, clean_text, parse_relative_time
from preprocess.models import RawRecord, CleanRecord
from preprocess.llm_placeholders import TitleLLMEnricher, SpecsLLMEnricher
from preprocess.db_writer import PostgresWriter


class PreprocessPipeline:
    """
    1) Clean raw docs
    2) Dedup by source_url in DB
    3) LLM enrich title/category (stub)
    4) LLM normalize specs (stub)
    5) Insert into Postgres
    """

    def __init__(
        self,
        title_enricher: TitleLLMEnricher | None = None,
        specs_enricher: SpecsLLMEnricher | None = None,
        db_writer: PostgresWriter | None = None,
    ):
        self.title_enricher = title_enricher or TitleLLMEnricher()
        self.specs_enricher = specs_enricher or SpecsLLMEnricher()
        self.db_writer = db_writer or PostgresWriter()

    def run(self, raw_docs: List[dict]):
        raw_records = [self._to_raw(rec) for rec in raw_docs]
        cleaned = [r for r in (self._clean_record(r) for r in raw_records) if r]

        if not cleaned:
            print("No valid records after cleaning.")
            return

        cleaned = self._filter_existing(cleaned)
        if not cleaned:
            print("All records already exist in DB.")
            return

        title_results = self.title_enricher.run_batch(cleaned)
        product_ids = self.db_writer.insert_products(cleaned, title_results)

        for pid, rec in zip(product_ids, cleaned):
            if rec.content_type != "product" and not rec.product_blob:
                continue
            specs = self.specs_enricher.run(rec)
            self.db_writer.insert_specs(pid, specs)

        print(f"Inserted {len(product_ids)} rows into products.")

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
            domain=rec.domain or (normalize_url(rec.url) and normalize_url(rec.url).split("/")[2]) or "",
            crawl_date=crawl_dt,
            title=title,
            category=rec.category,
            price=clean_price(rec.price),
            content_text=clean_text(rec.content_text),
            images=rec.images or [],
            content_type=content_type,
            product_blob=rec.product_blob,
        )

    def _resolve_datetime(self, rec: RawRecord) -> datetime:
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
        except Exception as e:
            print(f"Skip dedup (table missing or query failed): {e}")
            existing = set()

        return [r for r in records if r.source_url not in existing]


def run_from_file(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    pipeline = PreprocessPipeline()
    pipeline.run(docs)
