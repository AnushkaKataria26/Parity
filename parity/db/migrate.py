import sqlite3
import os
from .schema import SCHEMA_STATEMENTS

def apply_schema(conn: sqlite3.Connection) -> None:
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
    conn.commit()

def upsert_repo(conn: sqlite3.Connection, name: str, path: str, commit_sha: str | None) -> int:
    abs_path = os.path.abspath(path)
    conn.execute(
        """INSERT INTO repos (name, path, last_ingested_commit_sha) 
           VALUES (?, ?, ?) 
           ON CONFLICT(path) DO UPDATE SET 
           name=excluded.name, 
           last_ingested_commit_sha=excluded.last_ingested_commit_sha""",
        (name, abs_path, commit_sha)
    )
    conn.commit()
    
    cursor = conn.execute("SELECT id FROM repos WHERE path = ?", (abs_path,))
    return cursor.fetchone()["id"]
