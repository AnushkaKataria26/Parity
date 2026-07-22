import sqlite3
import os
import json
from typing import List

from parity.chunking.ast_chunker import CodeChunk

def store_chunks(conn: sqlite3.Connection, repo_id: int, chunks: List[CodeChunk]) -> int:
    """
    Stores code chunks into the `code_chunks` table and writes their bodies
    (source_text and docstring) to JSON files on disk.
    
    Re-running this for the same repo_id replaces all existing chunks for that repo.
    """
    # Raw source text doesn't belong in a relational row queried repeatedly,
    # and keeping it file-keyed by chunk id avoids bloating the DB.
    base_dir = os.path.join("data", "code_chunk_bodies", str(repo_id))
    os.makedirs(base_dir, exist_ok=True)
    
    with conn:
        # Before inserting new chunks for this repo, delete existing code_chunks rows.
        # This makes the operation idempotent.
        conn.execute("DELETE FROM code_chunks WHERE repo_id = ?", (repo_id,))
        
        count = 0
        for chunk in chunks:
            cursor = conn.execute(
                '''
                INSERT INTO code_chunks (repo_id, file_path, symbol_name, symbol_type, start_line, end_line, ast_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (repo_id, chunk.file_path, chunk.symbol_name, chunk.symbol_type, 
                 chunk.start_line, chunk.end_line, chunk.ast_hash)
            )
            chunk_id = cursor.lastrowid
            
            # Write companion artifact
            json_path = os.path.join(base_dir, f"{chunk_id}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "source_text": chunk.source_text,
                    "docstring": chunk.docstring
                }, f, ensure_ascii=False, indent=2)
            
            count += 1
            
    return count

from parity.chunking.doc_chunker import DocChunk

def store_doc_chunks(conn: sqlite3.Connection, repo_id: int, chunks: List[DocChunk]) -> int:
    """
    Stores doc chunks into the `doc_chunks` table and writes their bodies
    (code_blocks, start_line, end_line) to JSON files on disk.
    
    Re-running this for the same repo_id replaces all existing chunks for that repo.
    """
    base_dir = os.path.join("data", "doc_chunk_bodies", str(repo_id))
    os.makedirs(base_dir, exist_ok=True)
    
    with conn:
        conn.execute("DELETE FROM doc_chunks WHERE repo_id = ?", (repo_id,))
        
        count = 0
        for chunk in chunks:
            # Handle heading truncation
            heading = chunk.heading_path
            if len(heading) > 500:
                print(f"Warning: truncating extremely long heading path ({len(heading)} chars) to 500 chars")
                heading = heading[:497] + "..."
                
            cursor = conn.execute(
                '''
                INSERT INTO doc_chunks (repo_id, file_path, heading, text)
                VALUES (?, ?, ?, ?)
                ''',
                (repo_id, chunk.file_path, heading, chunk.text)
            )
            chunk_id = cursor.lastrowid
            
            # Write companion artifact
            json_path = os.path.join(base_dir, f"{chunk_id}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "code_blocks": chunk.code_blocks,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line
                }, f, ensure_ascii=False, indent=2)
            
            count += 1
            
    return count
