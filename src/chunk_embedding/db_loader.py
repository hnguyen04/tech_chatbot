
from collections import defaultdict
from typing import List, Dict
from storage.postgres_client import PostgresClient


class PostgresLoader:
    def __init__(self, client: PostgresClient | None = None):
        self.client = client or PostgresClient()
        self._spec_key_registry = None
        self._unit_registry = None    
    
    def load_products_batch(
        self,
        batch_size: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """
        Load products đã llm_processed theo batch
        """
        sql = """
            SELECT
                id,
                source_url,
                content_type,
                brand,
                model,
                product_line,
                full_title,
                category,
                category_eng,
                content_text
            FROM products
            WHERE llm_processed = TRUE
            ORDER BY id
            LIMIT %(limit)s OFFSET %(offset)s
        """
        return self.client.fetch_all(
            sql,
            {
                "limit": batch_size,
                "offset": offset,
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

    # -------------------------
    # CLEANUP
    # -------------------------

    def close(self):
        self.client.close()