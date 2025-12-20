
from collections import defaultdict
from typing import List, Dict
from storage.postgres_client import PostgresClient


class PostgresLoader:
    def __init__(self, client: PostgresClient | None = None):
        self.client = client or PostgresClient()

    def load_records_batch(
        self,
        batch_size: int = 100,
        last_id: int = 0,
    ) -> list[dict]:
        """
        Load đúng batch_size records tiếp theo theo id
        """
        sql = """
            SELECT
                id,
                source_url,
                content_type,
                brand,
                model,
                product_line,
                price,
                full_title,
                category,
                category_eng,
                content_text
            FROM products
            WHERE
                llm_processed = TRUE and chunked = FALSE
                AND id > %(last_id)s
            ORDER BY id
            LIMIT %(limit)s
        """

        return self.client.fetch_all(
            sql,
            {
                "limit": batch_size,
                "last_id": last_id,
            },
        )


    # -------------------------
    # SPECS (1-n)
    # -------------------------

    def load_specs_by_product_ids(
        self,
        product_ids: List[int],
    ) -> Dict[int, List[dict]]:
        """
        Load specs cho danh sách product_ids
        Trả về: { product_id: [spec, spec, ...] }
        """
        if not product_ids:
            return {}

        sql = """
            SELECT
                product_id,
                standardized_key,
                standardized_key_eng,
                standardized_value,
                standardized_value_eng,
                category,
                category_eng,
                numerical_value_list,
                unit_list
            FROM product_specifications
            WHERE product_id = ANY(%(product_ids)s)
            ORDER BY product_id
        """

        rows = self.client.fetch_all(
            sql,
            {"product_ids": product_ids},
        )

        grouped: Dict[int, List[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["product_id"]].append(row)

        return grouped
    
    def mark_chunked(self, record_ids: List[int]):
        if not record_ids:
            return
        sql = """
            UPDATE products
            SET chunked = TRUE
            WHERE id = ANY(%s)
        """
        try:
            self.client.cur.execute(sql, (record_ids,))
            self.client.conn.commit()
        except Exception as e:
            self.client.conn.rollback()
            print(f"❌ Failed to mark chunked: {e}")

    # -------------------------
    # CLEANUP
    # -------------------------

    def close(self):
        self.client.close()