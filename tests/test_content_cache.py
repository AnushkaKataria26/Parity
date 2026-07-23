import sqlite3
import pytest
from parity.versioning.content_cache import compute_file_hash, needs_reprocessing, update_cache_entry

def test_compute_file_hash(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello")
    hash1 = compute_file_hash(str(f))
    
    f.write_text("hello")
    hash2 = compute_file_hash(str(f))
    
    f.write_text("world")
    hash3 = compute_file_hash(str(f))
    
    assert hash1 == hash2
    assert hash1 != hash3

def test_needs_reprocessing():
    conn = sqlite3.connect(":memory:")
    conn.execute('''
        CREATE TABLE file_cache (
            repo_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            last_processed_commit_sha TEXT,
            last_processed_at TEXT,
            PRIMARY KEY (repo_id, file_path, file_type)
        )
    ''')
    
    # Not in cache -> needs reprocessing
    assert needs_reprocessing(conn, 1, "main.py", "code", "hash1") is True
    
    # Update cache
    update_cache_entry(conn, 1, "main.py", "code", "hash1", "sha1")
    
    # Same hash -> no reprocessing
    assert needs_reprocessing(conn, 1, "main.py", "code", "hash1") is False
    
    # Different hash -> needs reprocessing
    assert needs_reprocessing(conn, 1, "main.py", "code", "hash2") is True
    
    # Different file type -> needs reprocessing
    assert needs_reprocessing(conn, 1, "main.py", "doc", "hash1") is True
    
    # Different file path -> needs reprocessing
    assert needs_reprocessing(conn, 1, "other.py", "code", "hash1") is True
