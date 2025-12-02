from typing import Iterable, List, Tuple
from psycopg2.extras import execute_values, Json
from preprocess.models import CleanRecord, TitleLLMResult, SpecItem
from storage.postgres_client import PostgresClient

MAX_STR_200 = 200
MAX_STR_50 = 50


def _truncate(val: str | None, limit: int) -> str | None:
    if val is None:
        return None
    return val[:limit]


class PostgresWriter:
    def __init__(self, client: PostgresClient | None = None):
        self.client = client or PostgresClient()

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
        RETURNING id;
        """
        execute_values(self.client.cur, sql, rows)
        ids = [row[0] for row in self.client.cur.fetchall()]
        self.client.conn.commit()
        return ids

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
