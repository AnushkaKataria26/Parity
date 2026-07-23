import os
from parity.db.connection import get_connection
from parity.db.migrate import apply_schema, upsert_repo

def test_apply_schema(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    apply_schema(conn)
    
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row["name"] for row in cursor.fetchall()]
    
    expected_tables = ["repos", "code_chunks", "doc_chunks", "claims", "verification_results"]
    for t in expected_tables:
        assert t in tables
        
def test_apply_schema_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    apply_schema(conn)
    apply_schema(conn) # Should not raise
    
def test_upsert_repo(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    apply_schema(conn)
    
    repo_path = "/fake/repo"
    id1 = upsert_repo(conn, "repo", repo_path, "sha1")
    id2 = upsert_repo(conn, "repo", repo_path, "sha2")
    
    assert id1 == id2
    
    cursor = conn.execute("SELECT COUNT(*) as count FROM repos")
    assert cursor.fetchone()["count"] == 1
    
    cursor = conn.execute("SELECT last_ingested_commit_sha FROM repos WHERE path = ?", (os.path.abspath(repo_path),))
    assert cursor.fetchone()["last_ingested_commit_sha"] == "sha1"

def test_upsert_repo_path_normalization(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(str(db_path))
    apply_schema(conn)
    
    cwd = os.getcwd()
    rel_path = "some/dir"
    abs_path = os.path.abspath(rel_path)
    
    id1 = upsert_repo(conn, "repo", rel_path, "sha1")
    id2 = upsert_repo(conn, "repo", abs_path, "sha2")
    
    assert id1 == id2
    
    cursor = conn.execute("SELECT COUNT(*) as count FROM repos")
    assert cursor.fetchone()["count"] == 1
