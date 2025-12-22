import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")


def init_task_status_table():
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS task_status (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress JSONB,
                    result JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
        conn.commit()
    print("✅ task_status table ensured.")
