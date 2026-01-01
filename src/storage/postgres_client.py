import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import os
from dotenv import load_dotenv

class PostgresClient:
    def __init__(self):
        load_dotenv()
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

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()


# Usage
pg_client = PostgresClient()