import hashlib
from datetime import datetime, timezone
import sqlite3

def compute_file_hash(file_path: str) -> str:
    """
    Computes SHA256 of the raw file bytes.
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks for memory efficiency on large files
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_cached_hash(conn: sqlite3.Connection, repo_id: int, file_path: str, file_type: str) -> str | None:
    """
    Returns the content_hash from file_cache, or None if no entry exists.
    """
    cursor = conn.execute(
        "SELECT content_hash FROM file_cache WHERE repo_id=? AND file_path=? AND file_type=?",
        (repo_id, file_path, file_type)
    )
    row = cursor.fetchone()
    if row:
        # Depending on row factory, it might be a tuple or sqlite3.Row
        return row[0] if isinstance(row, tuple) or (hasattr(row, 'keys') and not isinstance(row, dict)) else row['content_hash']
    return None

def update_cache_entry(conn: sqlite3.Connection, repo_id: int, file_path: str, file_type: str, content_hash: str, commit_sha: str | None) -> None:
    """
    Upserts a row into file_cache.
    """
    now = datetime.now(timezone.utc).isoformat()
    
    conn.execute(
        '''
        INSERT INTO file_cache (repo_id, file_path, file_type, content_hash, last_processed_commit_sha, last_processed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(repo_id, file_path, file_type) DO UPDATE SET
            content_hash=excluded.content_hash,
            last_processed_commit_sha=excluded.last_processed_commit_sha,
            last_processed_at=excluded.last_processed_at
        ''',
        (repo_id, file_path, file_type, content_hash, commit_sha, now)
    )

def needs_reprocessing(conn: sqlite3.Connection, repo_id: int, file_path: str, file_type: str, current_hash: str) -> bool:
    """
    Returns True if the file needs reprocessing (i.e. not in cache, or hash differs).
    
    Note: A file whose content hash is unchanged but which was already deleted from code_chunks/doc_chunks
    by some out-of-band DB edit is not specially detected here — this cache only tracks file-content staleness,
    not DB-row presence.
    """
    cached_hash = get_cached_hash(conn, repo_id, file_path, file_type)
    return cached_hash != current_hash
