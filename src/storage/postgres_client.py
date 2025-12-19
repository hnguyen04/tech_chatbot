import psycopg2
from psycopg2.extras import execute_values
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
        try:
            self.conn = psycopg2.connect(
                host=self.host, port=self.port, dbname=self.dbname, user=self.user, password=self.password
            )
            self.cur = self.conn.cursor()
            print(f"✅ Connected successfully to Postgres at {self.host}:{self.port}/{self.dbname}")
        except Exception as e:
            print(f"❌ Failed to connect to Postgres: {e}")
    
    def fetch_all(self, sql: str, params: dict | None = None) -> list[dict]:
        self.cur.execute(sql, params or {})
        return self.cur.fetchall()

    def execute(self, sql: str, params: dict | None = None):
        self.cur.execute(sql, params or {})
        self.conn.commit()

    def executemany_values(self, sql: str, values: list[tuple]):
        execute_values(self.cur, sql, values)
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()


# Usage
pg_client = PostgresClient(
)