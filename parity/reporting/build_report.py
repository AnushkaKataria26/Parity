import os
import json
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DriftEntry:
    doc_file_path: str
    doc_heading: str
    doc_start_line: Optional[int]
    doc_end_line: Optional[int]
    claim_text: str
    claim_type: str
    status: str
    claimed_value: Optional[str]
    actual_value: Optional[str]
    matched_symbol: Optional[str]
    matched_file_path: Optional[str]
    matched_start_line: Optional[int]
    matched_end_line: Optional[int]

@dataclass
class DriftReport:
    repo_name: str
    repo_path: str
    generated_at: str
    commit_sha: Optional[str]
    entries_by_file: Dict[str, List[DriftEntry]]
    totals: Dict[str, int]

def build_drift_report(conn, repo_id: int) -> DriftReport:
    cursor = conn.execute("SELECT name, path, last_ingested_commit_sha FROM repos WHERE id = ?", (repo_id,))
    row = cursor.fetchone()
    if not row:
        raise ValueError(f"Repo ID {repo_id} not found")
        
    repo_name, repo_path, commit_sha = row
    generated_at = datetime.now(timezone.utc).isoformat()
    
    query = """
        SELECT 
            vr.status, vr.claimed_value, vr.actual_value,
            c.claim_text, c.claim_type,
            d.id as doc_chunk_id, d.file_path as doc_file_path, d.heading as doc_heading,
            cc.symbol_name, cc.file_path as matched_file_path, cc.start_line, cc.end_line
        FROM verification_results vr
        JOIN claims c ON vr.claim_id = c.id
        JOIN doc_chunks d ON c.doc_chunk_id = d.id
        LEFT JOIN code_chunks cc ON vr.matched_code_chunk_id = cc.id
        WHERE d.repo_id = ?
    """
    cursor = conn.execute(query, (repo_id,))
    results = cursor.fetchall()
    
    totals = {"verified": 0, "contradicted": 0, "unverifiable": 0}
    entries = []
    
    for row in results:
        status = row[0]
        claimed_value = row[1]
        actual_value = row[2]
        claim_text = row[3]
        claim_type = row[4]
        doc_chunk_id = row[5]
        doc_file_path = row[6]
        doc_heading = row[7]
        matched_symbol = row[8]
        matched_file_path = row[9]
        matched_start_line = row[10]
        matched_end_line = row[11]
        
        key = status.lower()
        if key in totals:
            totals[key] += 1
            
        doc_start_line = None
        doc_end_line = None
        json_path = os.path.join("data", "doc_chunk_bodies", str(repo_id), f"{doc_chunk_id}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    chunk_data = json.load(f)
                    doc_start_line = chunk_data.get("start_line")
                    doc_end_line = chunk_data.get("end_line")
            except Exception:
                pass
                
        entry = DriftEntry(
            doc_file_path=doc_file_path,
            doc_heading=doc_heading if doc_heading is not None else "",
            doc_start_line=doc_start_line,
            doc_end_line=doc_end_line,
            claim_text=claim_text,
            claim_type=claim_type,
            status=status,
            claimed_value=claimed_value,
            actual_value=actual_value,
            matched_symbol=matched_symbol,
            matched_file_path=matched_file_path,
            matched_start_line=matched_start_line,
            matched_end_line=matched_end_line
        )
        entries.append(entry)
        
    severity_rank = {"Contradicted": 0, "Unverifiable": 1, "Verified": 2}
    
    def entry_sort_key(e: DriftEntry):
        rank = severity_rank.get(e.status, 99)
        line = e.doc_start_line if e.doc_start_line is not None else float('inf')
        return (rank, line)
        
    grouped: Dict[str, List[DriftEntry]] = {}
    for e in entries:
        grouped.setdefault(e.doc_file_path, []).append(e)
        
    for file_path in grouped:
        grouped[file_path].sort(key=entry_sort_key)
        
    def file_sort_key(file_path: str):
        min_rank = min(severity_rank.get(e.status, 99) for e in grouped[file_path])
        return (min_rank, file_path)
        
    sorted_files = sorted(grouped.keys(), key=file_sort_key)
    entries_by_file = {f: grouped[f] for f in sorted_files}
    
    return DriftReport(
        repo_name=repo_name,
        repo_path=repo_path,
        generated_at=generated_at,
        commit_sha=commit_sha,
        entries_by_file=entries_by_file,
        totals=totals
    )
