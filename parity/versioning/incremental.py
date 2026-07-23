import os
import sqlite3
import logging
from typing import Set, Tuple, Any

from parity.versioning.git_diff import get_current_commit_sha, get_changed_files_since
from parity.versioning.content_cache import compute_file_hash, needs_reprocessing, update_cache_entry

def run_incremental_chunking(
    conn: sqlite3.Connection, 
    repo_id: int, 
    repo_path: str, 
    file_type: str, 
    discover_fn, 
    extract_fn, 
    store_file_fn,
    delete_file_fn,
    store_all_fn
) -> Tuple[dict, Set[str]]:
    """
    Shared incremental orchestration logic.
    Returns (summary_dict, candidate_files_set).
    """
    current_sha = get_current_commit_sha(repo_path)
    
    cursor = conn.execute("SELECT last_ingested_commit_sha FROM repos WHERE id = ?", (repo_id,))
    row = cursor.fetchone()
    stored_sha = row[0] if row else None
    
    # Check if this is truly the first run (no chunks yet)
    chunk_table = "code_chunks" if file_type == "code" else "doc_chunks"
    cursor = conn.execute(f"SELECT COUNT(*) FROM {chunk_table} WHERE repo_id = ?", (repo_id,))
    count = cursor.fetchone()[0]
    
    if count == 0:
        is_full_rescan = True
        changed_files = set()
    else:
        changed_files, is_full_rescan = get_changed_files_since(repo_path, stored_sha)
        
    all_files = discover_fn(repo_path)
    
    summary = {
        "is_full_rescan": is_full_rescan,
        "changed_detected": 0,
        "skipped": 0,
        "reprocessed": 0,
        "deleted": 0,
        "total_chunks_now": 0
    }
    
    if is_full_rescan:
        all_chunks = []
        files_scanned = 0
        files_skipped = 0
        
        for file_path in all_files:
            files_scanned += 1
            chunks_tuple = extract_fn(file_path, repo_path)
            # extract_fn returns (chunks, is_skipped) for code, but only chunks for docs?
            # Wait, doc_chunker returns list of chunks. Code returns (chunks, is_skipped).
            # We'll normalize this wrapper in the caller or here. Let's assume extract_fn returns what we need to store.
            if isinstance(chunks_tuple, tuple):
                chunks, is_skipped = chunks_tuple
                if is_skipped:
                    files_skipped += 1
            else:
                chunks = chunks_tuple
            
            all_chunks.extend(chunks)
            
            rel_path = os.path.relpath(file_path, repo_path).replace(os.sep, '/')
            try:
                content_hash = compute_file_hash(file_path)
                update_cache_entry(conn, repo_id, rel_path, file_type, content_hash, current_sha)
            except FileNotFoundError:
                pass
                
        store_all_fn(conn, repo_id, all_chunks)
        summary["reprocessed"] = files_scanned
        summary["skipped_syntax"] = files_skipped # custom for code
        
        # update repo sha
        if current_sha:
            conn.execute("UPDATE repos SET last_ingested_commit_sha = ? WHERE id = ?", (current_sha, repo_id))
            conn.commit()
            
        cursor = conn.execute(f"SELECT COUNT(*) FROM {chunk_table} WHERE repo_id = ?", (repo_id,))
        summary["total_chunks_now"] = cursor.fetchone()[0]
        summary["all_chunks_obj"] = all_chunks
        return summary, set([os.path.relpath(f, repo_path).replace(os.sep, '/') for f in all_files])
        
    # Incremental path
    summary["changed_detected"] = len(changed_files)
    
    # intersect changed_files with discovered files to apply exclusion rules
    discovered_rel_paths = {os.path.relpath(f, repo_path).replace(os.sep, '/'): f for f in all_files}
    candidate_files = changed_files.intersection(discovered_rel_paths.keys())
    
    # Also handle deleted files (in changed_files but not in discovered files, AND were in DB)
    deleted_candidates = changed_files - discovered_rel_paths.keys()
    for rel_path in deleted_candidates:
        delete_file_fn(conn, repo_id, rel_path)
        conn.execute("DELETE FROM file_cache WHERE repo_id = ? AND file_path = ? AND file_type = ?", (repo_id, rel_path, file_type))
        summary["deleted"] += 1
        
    all_chunks_for_stats = []
    
    for rel_path in candidate_files:
        abs_path = discovered_rel_paths[rel_path]
        
        if not os.path.exists(abs_path):
            delete_file_fn(conn, repo_id, rel_path)
            conn.execute("DELETE FROM file_cache WHERE repo_id = ? AND file_path = ? AND file_type = ?", (repo_id, rel_path, file_type))
            summary["deleted"] += 1
            continue
            
        content_hash = compute_file_hash(abs_path)
        if not needs_reprocessing(conn, repo_id, rel_path, file_type, content_hash):
            logging.info(f"Skipping {rel_path}, hash unchanged")
            summary["skipped"] += 1
            continue
            
        chunks_tuple = extract_fn(abs_path, repo_path)
        is_syntax_skipped = False
        if isinstance(chunks_tuple, tuple):
            chunks, is_syntax_skipped = chunks_tuple
        else:
            chunks = chunks_tuple
            
        if not is_syntax_skipped:
            store_file_fn(conn, repo_id, rel_path, chunks)
            all_chunks_for_stats.extend(chunks)
            update_cache_entry(conn, repo_id, rel_path, file_type, content_hash, current_sha)
            summary["reprocessed"] += 1
        else:
            summary["skipped"] += 1
            # Still update cache so we don't keep trying to parse a broken file if it doesn't change
            update_cache_entry(conn, repo_id, rel_path, file_type, content_hash, current_sha)
            
    if current_sha:
        conn.execute("UPDATE repos SET last_ingested_commit_sha = ? WHERE id = ?", (current_sha, repo_id))
        
    conn.commit()
    
    cursor = conn.execute(f"SELECT COUNT(*) FROM {chunk_table} WHERE repo_id = ?", (repo_id,))
    summary["total_chunks_now"] = cursor.fetchone()[0]
    summary["all_chunks_obj"] = all_chunks_for_stats
    
    return summary, changed_files
