import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

class PostgresClient:
    def __init__(self):
        # Explicitly load .env from project root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        env_path = project_root / '.env'
        load_dotenv(dotenv_path=env_path, override=True)

        print(f"Loaded .env from {env_path}")

        self.host = os.getenv("POSTGRES_DB_HOST")
        self.port = os.getenv("POSTGRES_DB_PORT")
        self.dbname = os.getenv("POSTGRES_DB_NAME")
        self.user = os.getenv("POSTGRES_DB_USER")
        self.password = os.getenv("POSTGRES_DB_PASSWORD")
        self.conn = None
        self.cur = None
        
        # Only connect if credentials are provided
        if self.host and self.dbname and self.user and self.password:
            try:
                self.conn = psycopg2.connect(
                    host=self.host, port=self.port, dbname=self.dbname, user=self.user, password=self.password,
                    cursor_factory=RealDictCursor, 
                )
                self.cur = self.conn.cursor()
                print(f"Connected successfully to Postgres at {self.host}:{self.port}/{self.dbname}")
            except Exception as e:
                print(f"Warning: Failed to connect to Postgres: {e}")
                print("Metadata enrichment will be disabled")
                self.conn = None
                self.cur = None
        else:
            print("PostgreSQL credentials not provided - metadata enrichment will be disabled")
    
    def fetch_all(self, sql: str, params: dict | None = None) -> list[dict]:
        if not self.cur:
            raise RuntimeError("PostgreSQL connection not established")
        self.cur.execute(sql, params or {})
        return self.cur.fetchall()

    def execute(self, sql: str, params: dict | None = None):
        if not self.cur:
            raise RuntimeError("PostgreSQL connection not established")
        self.cur.execute(sql, params or {})
        self.conn.commit()

    def executemany_values(self, sql: str, values: list[tuple]):
        if not self.cur:
            raise RuntimeError("PostgreSQL connection not established")
        execute_values(self.cur, sql, values)
        self.conn.commit()

    def search_keyword(self, query: str, limit: int = 20, products_only: bool = False) -> list[dict]:
        """
        Perform keyword search using PostgreSQL Full-Text Search with ILIKE fallback
        
        Args:
            query: Search query
            limit: Max results
            products_only: If True, only return content_type='product' (not articles)
            
        Returns:
            List of products with rank score
        """
        if not self.cur:
            return []
            
        # Using websearch_to_tsquery for flexible query parsing
        # Search in title, brand, model, and content
        # BOOST: products > articles, exact title/model match > fuzzy match, products WITH SPECS > without
        content_type_filter = "AND content_type = 'product'" if products_only else ""
        
        sql = f"""
            SELECT p.id, p.full_title, p.category, p.brand, p.model, p.price, 
                   p.source_url, p.content_text, p.content_type,
                   ts_rank(
                       setweight(to_tsvector('simple', coalesce(p.full_title, '')), 'A') || 
                       setweight(to_tsvector('simple', coalesce(p.brand, '')), 'B') ||
                       setweight(to_tsvector('simple', coalesce(p.model, '')), 'B') ||
                       setweight(to_tsvector('simple', coalesce(p.content_text, '')), 'C'),
                       websearch_to_tsquery('simple', %s)
                   ) 
                   * CASE WHEN p.content_type = 'product' THEN 10.0 ELSE 1.0 END
                   * CASE WHEN lower(p.full_title) LIKE lower(%s) THEN 100.0
                          WHEN lower(p.model) LIKE lower(%s) THEN 50.0 
                          ELSE 1.0 END
                   * CASE WHEN spec_count > 0 THEN 100.0 ELSE 1.0 END
                   as rank
            FROM products p
            LEFT JOIN (
                SELECT product_id, COUNT(*) as spec_count 
                FROM product_specifications 
                GROUP BY product_id
            ) ps ON p.id = ps.product_id
            WHERE (
                to_tsvector('simple', coalesce(p.full_title, '')) || 
                to_tsvector('simple', coalesce(p.brand, '')) ||
                to_tsvector('simple', coalesce(p.model, '')) ||
                to_tsvector('simple', coalesce(p.content_text, ''))
            ) @@ websearch_to_tsquery('simple', %s)
            {content_type_filter}
            ORDER BY rank DESC
            LIMIT %s
        """
        
        # Create LIKE pattern for exact match boost (e.g., '%S24 Ultra%')
        like_pattern = f'%{query}%'
        
        try:
            self.cur.execute(sql, (query, like_pattern, like_pattern, query, limit))
            results = self.cur.fetchall()
            
            # If full-text search returns nothing, fallback to ILIKE pattern search
            if not results:
                results = self._search_keyword_ilike(query, limit, products_only)
            
            return results
        except Exception as e:
            # This is normal for complex Vietnamese queries - silently fallback to ILIKE
            return self._search_keyword_ilike(query, limit, products_only)
    
    def _search_keyword_ilike(self, query: str, limit: int = 20, products_only: bool = False) -> list[dict]:
        """
        Fallback ILIKE-based search when full-text search fails or returns no results.
        Better for exact product names like 'iPhone 15', 'S24 Ultra'.
        """
        if not self.cur:
            return []
        
        content_type_filter = "AND content_type = 'product'" if products_only else ""
        like_pattern = f'%{query}%'
        
        sql = f"""
            SELECT p.id, p.full_title, p.category, p.brand, p.model, p.price, 
                   p.source_url, p.content_text, p.content_type,
                   CASE 
                       WHEN lower(p.full_title) ILIKE lower(%s) THEN 100
                       WHEN lower(p.model) ILIKE lower(%s) THEN 50
                       WHEN lower(p.brand) ILIKE lower(%s) THEN 25
                       ELSE 1
                   END 
                   * CASE WHEN p.content_type = 'product' THEN 10 ELSE 1 END
                   * CASE WHEN ps.spec_count > 0 THEN 10 ELSE 1 END
                   as rank
            FROM products p
            LEFT JOIN (
                SELECT product_id, COUNT(*) as spec_count 
                FROM product_specifications 
                GROUP BY product_id
            ) ps ON p.id = ps.product_id
            WHERE (
                p.full_title ILIKE %s 
                OR p.model ILIKE %s 
                OR p.brand ILIKE %s
            )
            {content_type_filter}
            ORDER BY rank DESC
            LIMIT %s
        """
        
        try:
            self.cur.execute(sql, (like_pattern, like_pattern, like_pattern, 
                                   like_pattern, like_pattern, like_pattern, limit))
            return self.cur.fetchall()
        except Exception as e:
            print(f"Error in ILIKE fallback search: {e}")
            return []

    def fetch_product_by_id(self, product_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a single product by its ID.
        
        Args:
            product_id: Product ID
            
        Returns:
            Product dict or None if not found
        """
        if not self.cur:
            return None
            
        sql = """
            SELECT id, full_title, brand, model, price, category, 
                   content_text, source_url, images
            FROM products 
            WHERE id = %s
        """
        
        try:
            self.cur.execute(sql, (product_id,))
            result = self.cur.fetchone()
            return dict(result) if result else None
        except Exception as e:
            print(f"Error fetching product {product_id}: {e}")
            return None

    def fetch_product_specs(self, product_id: int) -> List[Dict[str, Any]]:
        """
        Fetch specifications for a product.
        
        Args:
            product_id: Product ID
            
        Returns:
            List of specification dicts
        """
        if not self.cur:
            return []
            
        sql = """
            SELECT standardized_key, standardized_value, 
                   category, numerical_value_list, unit_list
            FROM product_specifications
            WHERE product_id = %s
            ORDER BY category, standardized_key
        """
        
        try:
            self.cur.execute(sql, (product_id,))
            results = self.cur.fetchall()
            return [dict(r) for r in results]
        except Exception as e:
            print(f"Error fetching specs for product {product_id}: {e}")
            return []

    def fetch_product_with_specs(self, product_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a product with its specifications combined.
        
        Args:
            product_id: Product ID
            
        Returns:
            Product dict with 'specifications' key, or None
        """
        product = self.fetch_product_by_id(product_id)
        if product:
            product['specifications'] = self.fetch_product_specs(product_id)
        return product

    def compare_products(self, product_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Fetch multiple products with specs for side-by-side comparison.
        
        Args:
            product_ids: List of product IDs to compare
            
        Returns:
            List of product dicts with specifications
        """
        results = []
        # Limit to 5 products for reasonable comparison
        for pid in product_ids[:5]:
            product = self.fetch_product_with_specs(pid)
            if product:
                results.append(product)
        return results

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()