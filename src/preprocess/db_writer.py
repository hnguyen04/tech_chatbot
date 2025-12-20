from typing import Iterable, List, Tuple
from psycopg2.extras import execute_values, Json
from preprocess.models import CleanRecord, TitleLLMResult, SpecItem
from storage.postgres_client import PostgresClient
import unicodedata
import re

MAX_STR_200 = 200
MAX_STR_50 = 50


def _truncate(val: str | None, limit: int) -> str | None:
    if val is None:
        return None
    return val[:limit]


class PostgresWriter:
    def __init__(self, client: PostgresClient | None = None):
        self.client = client or PostgresClient()
        self._spec_key_registry = None
        self._unit_registry = None

    def insert_products(
        self,
        cleaned: List[CleanRecord],
        llm_results: List[TitleLLMResult],
    ) -> List[int]:
        rows = []
        for rec, llm in zip(cleaned, llm_results):
            rows.append(
                (
                    rec.source_url,
                    rec.domain,
                    rec.crawl_date,
                    rec.content_type,
                    llm.brand,
                    llm.model,
                    llm.product_line,
                    rec.title,
                    llm.full_title_eng,
                    rec.category,
                    llm.category_eng,
                    rec.price,
                    rec.content_text,
                    Json(rec.images or []),
                    llm.llm_processed,
                    False,
                )
            )

        sql = """
        INSERT INTO products
        (source_url, domain, crawl_date, content_type, brand, model, product_line, full_title,
        full_title_eng, category, category_eng, price, content_text, images,
        llm_processed, chunked)
        VALUES %s
        ON CONFLICT (source_url) DO NOTHING
        RETURNING id;
        """

        try:
            execute_values(self.client.cur, sql, rows)
            ids = [row[0] for row in self.client.cur.fetchall()]
            self.client.conn.commit()
            return ids
        except Exception as e:
            self.client.conn.rollback()
            raise

    def insert_specs(self, product_id: int, specs: Iterable[SpecItem]):
        rows: List[Tuple] = []
        for item in specs:
            rows.append(
                (
                    product_id,
                    _truncate(item.standardized_key, MAX_STR_200),
                    _truncate(item.standardized_key_eng, MAX_STR_200),
                    _truncate(item.standardized_value, MAX_STR_200),
                    _truncate(item.standardized_value_eng, MAX_STR_200),
                    _truncate(item.category, MAX_STR_50),
                    _truncate(item.category_eng, MAX_STR_50),
                    item.numerical_value_list,
                    item.unit_list,
                )
            )

        if not rows:
            return

        sql = """
        INSERT INTO product_specifications
        (product_id, standardized_key, standardized_key_eng, standardized_value,
        standardized_value_eng, category, category_eng, numerical_value_list, unit_list)
        VALUES %s;
        """
        rows = [
            (
                r[0],
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                r[6],
                Json(r[7] or []),
                Json(r[8] or []),
            )
            for r in rows
        ]
        execute_values(self.client.cur, sql, rows)
        self.client.conn.commit()

    def insert_spec_keys(self, rows):
        sql = """
        INSERT INTO spec_key_registry (raw_key, normalized_key)
        VALUES %s
        ON CONFLICT (normalized_key) DO NOTHING;
        """
        execute_values(self.client.cur, sql, rows)
        self.client.conn.commit()


    def insert_raw_units(self, rows):
        sql = """
        INSERT INTO raw_unit_registry (raw_unit, normalized_unit)
        VALUES %s
        ON CONFLICT (raw_unit) DO NOTHING;
        """
        execute_values(self.client.cur, sql, rows)
        self.client.conn.commit()

    def fetch_spec_keys_missing_enrichment(self, limit: int):
        sql = """
        SELECT id, raw_key, normalized_key
        FROM spec_key_registry
        WHERE standardized_key_eng IS NULL
        LIMIT %s
        """
        with self.client.conn.cursor() as cur:
            cur.execute(sql, (limit,))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


    def update_spec_key_enrichment(self, rows):
        sql = """
        UPDATE spec_key_registry
        SET
            standardized_key_eng = %(standardized_key_eng)s,
            category_vi = %(category_vi)s,
            category_eng = %(category_eng)s
        WHERE id = %(id)s
        """
        with self.client.conn.cursor() as cur:
            cur.executemany(sql, rows)
        self.client.conn.commit()

    def load_spec_key_registry(self):
        """
        Cache:
        {
            normalized_key_lower: {
                standardized_key,
                standardized_key_eng,
                category,
                category_eng
            }
        }

        Key được tạo từ:
        - raw_key
        - normalized_key
        (đều qua cùng normalize function)
        """
        if self._spec_key_registry is not None:
            return self._spec_key_registry

        sql = """
        SELECT raw_key,
            normalized_key,
            standardized_key_eng,
            category_vi,
            category_eng
        FROM spec_key_registry
        """
        with self.client.conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        registry = {}

        def _norm(k: str) -> str:
            k = unicodedata.normalize("NFKC", k)
            k = k.lower().strip()
            k = re.sub(r"[:：]+$", "", k)
            k = re.sub(r"\s+", " ", k)
            return k

        for raw_key, normalized_key, eng, cat_vi, cat_eng in rows:
            entry = {
                "standardized_key": normalized_key,
                "standardized_key_eng": eng,
                "category": cat_vi,
                "category_eng": cat_eng,
            }

            # 🔹 key từ raw_key
            if raw_key:
                registry[_norm(raw_key)] = entry

            # 🔹 key từ normalized_key (rất quan trọng)
            if normalized_key:
                registry[_norm(normalized_key)] = entry

        self._spec_key_registry = registry
        return registry

    def load_unit_registry(self):
        """
        Cache:
        {
            raw_unit_lower: normalized_unit
        }
        """
        if self._unit_registry is not None:
            return self._unit_registry

        sql = """
        SELECT raw_unit, normalized_unit
        FROM raw_unit_registry
        """
        with self.client.conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        registry = {}
        for raw, norm in rows:
            if not raw:
                continue
            key = unicodedata.normalize("NFKC", raw).lower().strip()
            registry[key] = norm

        self._unit_registry = registry
        return registry
    
    def is_json_processed(self, s3_key: str) -> bool:
        sql = """
        SELECT processed
        FROM json_check
        WHERE s3_key = %s
        """
        with self.client.conn.cursor() as cur:
            cur.execute(sql, (s3_key,))
            row = cur.fetchone()
            return bool(row and row[0])
        
    def mark_json_processed(self, s3_key: str):
        sql = """
        INSERT INTO json_check (s3_key, processed)
        VALUES (%s, TRUE)
        ON CONFLICT (s3_key)
        DO UPDATE SET
            processed = TRUE,
            updated_at = NOW();
        """
        with self.client.conn.cursor() as cur:
            cur.execute(sql, (s3_key,))
        self.client.conn.commit()