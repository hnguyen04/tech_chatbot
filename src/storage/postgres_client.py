import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import os
from pathlib import Path
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

    def search_keyword(self, query: str, limit: int = 20) -> list[dict]:
        """
        Perform keyword search using PostgreSQL Full-Text Search
        
        Args:
            query: Search query
            limit: Max results
            
        Returns:
            List of products with rank score
        """
        if not self.cur:
            return []
            
        # Using websearch_to_tsquery for flexible query parsing
        # Search in title, brand, model, and content
        sql = """
            SELECT id, full_title, category, brand, model, price, source_url, content_text,
                   ts_rank(
                       setweight(to_tsvector('simple', coalesce(full_title, '')), 'A') || 
                       setweight(to_tsvector('simple', coalesce(brand, '')), 'B') ||
                       setweight(to_tsvector('simple', coalesce(model, '')), 'B') ||
                       setweight(to_tsvector('simple', coalesce(content_text, '')), 'C'),
                       websearch_to_tsquery('simple', %s)
                   ) as rank
            FROM products
            WHERE (
                to_tsvector('simple', coalesce(full_title, '')) || 
                to_tsvector('simple', coalesce(brand, '')) ||
                to_tsvector('simple', coalesce(model, '')) ||
                to_tsvector('simple', coalesce(content_text, ''))
            ) @@ websearch_to_tsquery('simple', %s)
            ORDER BY rank DESC
            LIMIT %s
        """
        
        try:
            self.cur.execute(sql, (query, query, limit))
            results = self.cur.fetchall()
            return results
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()