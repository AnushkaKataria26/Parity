import os
import shutil
import tempfile
import sqlite3
import traceback
from typing import Dict, Any

from parity.evaluation.fault_injection import select_fault_targets, inject_faults
from parity.db.migrate import upsert_repo
from parity.chunking.ast_chunker import discover_python_files, extract_chunks_from_file
from parity.db.chunk_ops import store_chunks
from parity.vectorstore.chroma_client import get_chroma_client, get_or_create_collections
from parity.embedding.model import embed_repo
from parity.retrieval.retriever import retrieve_for_repo
from parity.verification.verify import verify_repo

def run_fault_injection_eval(conn: sqlite3.Connection, repo_id: int, repo_path: str, config: Dict[str, Any], n_faults: int = 15) -> Dict[str, Any]:
    # 1. Copy the entire repo to a fresh tmp directory (excluding .git)
    tmp_dir = tempfile.mkdtemp(prefix="parity_eval_")
    repo_name = os.path.basename(repo_path)
    repo_copy_path = os.path.join(tmp_dir, repo_name)
    
    shutil.copytree(repo_path, repo_copy_path, ignore=shutil.ignore_patterns('.git'))
    
    temp_repo_id = None
    
    try:
        # 2. Select and inject faults
        targets = select_fault_targets(conn, repo_id, n_faults)
        if not targets:
            return {"n_faults": 0, "detected": 0, "missed": 0, "detection_rate": 0.0, "missed_details": []}
            
        injected_faults = inject_faults(conn, repo_id, repo_copy_path, targets)
        actual_n_faults = len(injected_faults)
        
        # 3. Register the copy as a separate, temporary repo row
        # This is essential so the eval run's chunks/claims/verification_results don't collide
        # with or overwrite the real repo's existing data in Phases 1-7's tables.
        temp_repo_id = upsert_repo(conn, name=f"{repo_name}_eval_copy", path=repo_copy_path, commit_sha=None)
        
        # 4. Run the full pipeline against this temporary repo id:
        # chunk-code --full
        python_files = discover_python_files(repo_copy_path)
        all_chunks = []
        for file_path in python_files:
            chunks, _ = extract_chunks_from_file(file_path, repo_copy_path)
            all_chunks.extend(chunks)
        store_chunks(conn, temp_repo_id, all_chunks)
        
        # embed
        client = get_chroma_client(config["chroma_persist_dir"])
        code_col, doc_col = get_or_create_collections(client)
        embed_repo(conn, temp_repo_id, client, code_col, doc_col)
        
        # copy claims and retrieval_results from the real repo's run instead of re-extracting
        cursor = conn.cursor()
        
        # Copy doc_chunks
        cursor.execute("SELECT id, file_path, heading, text, embedding_id FROM doc_chunks WHERE repo_id = ?", (repo_id,))
        doc_rows = cursor.fetchall()
        
        doc_chunk_map = {} # old_id -> new_id
        for d in doc_rows:
            cursor.execute(
                "INSERT INTO doc_chunks (repo_id, file_path, heading, text, embedding_id) VALUES (?, ?, ?, ?, ?)",
                (temp_repo_id, d[1], d[2], d[3], d[4])
            )
            doc_chunk_map[d[0]] = cursor.lastrowid
            
        # Copy claims
        claim_map = {} # old_id -> new_id
        for old_doc_id, new_doc_id in doc_chunk_map.items():
            cursor.execute("SELECT id, claim_text, claim_type, referenced_symbol_guess FROM claims WHERE doc_chunk_id = ?", (old_doc_id,))
            claims = cursor.fetchall()
            for c in claims:
                cursor.execute(
                    "INSERT INTO claims (doc_chunk_id, claim_text, claim_type, referenced_symbol_guess) VALUES (?, ?, ?, ?)",
                    (new_doc_id, c[1], c[2], c[3])
                )
                claim_map[c[0]] = cursor.lastrowid
                
        conn.commit()
        
        # Run retrieve
        retrieve_for_repo(conn, temp_repo_id, code_col, "BAAI/bge-small-en-v1.5", 5)
        
        # Run verify
        verify_repo(conn, temp_repo_id, repo_copy_path, config["ollama_model"], config["ollama_host"])
        
        # 5. Check results for injected faults
        detected = 0
        missed = 0
        missed_details = []
        
        for fault in injected_faults:
            # Look up whether its target claim ended up with status == "Contradicted"
            # We trace through the temp repo's code_chunks row -> symbol_name matching the fault's symbol_name
            # and file_path matching.
            
            # Find the new chunk id for this symbol in the temp repo
            cursor.execute("SELECT id FROM code_chunks WHERE repo_id = ? AND file_path = ? AND symbol_name = ?", (temp_repo_id, fault.file_path, fault.symbol_name))
            chunk_row = cursor.fetchone()
            if not chunk_row:
                # Should not happen unless chunking broke
                missed += 1
                missed_details.append({
                    "fault_type": fault.fault_type,
                    "symbol_name": fault.symbol_name,
                    "actual_outcome": "Chunk missing in temp repo"
                })
                continue
                
            new_chunk_id = chunk_row[0]
            
            # Find claims targeting this chunk via retrieval_results and check their verification status
            cursor.execute("""
                SELECT vr.status, c.claim_type, vr.actual_value, vr.claimed_value
                FROM verification_results vr
                JOIN claims c ON vr.claim_id = c.id
                JOIN retrieval_results rr ON rr.claim_id = c.id
                WHERE rr.matched_code_chunk_id = ? 
                  AND vr.matched_code_chunk_id = ?
            """, (new_chunk_id, new_chunk_id))
            
            v_rows = cursor.fetchall()
            
            # Was ANY claim covering this symbol contradicted? (For rename_param, it's signature. For change_default, default_value)
            # Actually, the fault targets a specific kind of claim, let's just see if ANY claim of the right type was contradicted.
            is_contradicted = False
            actual_outcome = "No matching claim verified"
            
            target_claim_type = "signature" if fault.fault_type == "rename_param" else "default_value"
            
            for v_row in v_rows:
                status, claim_type, actual, claimed = v_row
                if claim_type == target_claim_type:
                    actual_outcome = status
                    if status == "Contradicted":
                        is_contradicted = True
                        break
                        
            if is_contradicted:
                detected += 1
            else:
                missed += 1
                missed_details.append({
                    "fault_type": fault.fault_type,
                    "symbol_name": fault.symbol_name,
                    "actual_outcome": actual_outcome
                })
                
        detection_rate = (detected / actual_n_faults) * 100 if actual_n_faults > 0 else 0.0
        
        return {
            "n_faults": actual_n_faults,
            "detected": detected,
            "missed": missed,
            "detection_rate": detection_rate,
            "missed_details": missed_details
        }
        
    finally:
        # 8. Clean up
        if temp_repo_id is not None:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM verification_results 
                WHERE claim_id IN (SELECT id FROM claims WHERE doc_chunk_id IN (SELECT id FROM doc_chunks WHERE repo_id = ?))
            """, (temp_repo_id,))
            
            cursor.execute("""
                DELETE FROM retrieval_results 
                WHERE claim_id IN (SELECT id FROM claims WHERE doc_chunk_id IN (SELECT id FROM doc_chunks WHERE repo_id = ?))
            """, (temp_repo_id,))
            
            cursor.execute("DELETE FROM claims WHERE doc_chunk_id IN (SELECT id FROM doc_chunks WHERE repo_id = ?)", (temp_repo_id,))
            cursor.execute("DELETE FROM doc_chunks WHERE repo_id = ?", (temp_repo_id,))
            cursor.execute("DELETE FROM code_chunks WHERE repo_id = ?", (temp_repo_id,))
            cursor.execute("DELETE FROM file_cache WHERE repo_id = ?", (temp_repo_id,))
            cursor.execute("DELETE FROM repos WHERE id = ?", (temp_repo_id,))
            conn.commit()
            
            # Also clean up chunk bodies
            code_body_dir = f"data/code_chunk_bodies/{temp_repo_id}"
            if os.path.exists(code_body_dir):
                shutil.rmtree(code_body_dir, ignore_errors=True)
                
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
