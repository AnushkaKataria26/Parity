import csv
import logging
import random
from typing import Dict, Any

def export_claims_for_labeling(conn, repo_id: int, output_path: str) -> int:
    cursor = conn.cursor()
    cursor.execute("SELECT id, heading, text FROM doc_chunks WHERE repo_id = ?", (repo_id,))
    rows = cursor.fetchall()
    
    # Cap at ~40-50 chunks
    max_chunks = 50
    if len(rows) > max_chunks:
        random.seed(42)
        rows = random.sample(rows, max_chunks)
        
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_chunk_id", "heading", "text", "expected_claim_count", "expected_claim_types", "notes"])
        for row in rows:
            writer.writerow([row[0], row[1] or "", row[2] or "", "", "", ""])
            
    logging.info(f"Exported {len(rows)} chunks for labeling to {output_path}")
    return len(rows)

def score_extraction(conn, repo_id: int, labeled_path: str) -> Dict[str, Any]:
    """
    Computes a coarse precision/recall proxy based on chunk-level counts.
    Not a strict claim-level IR precision/recall.
    """
    cursor = conn.cursor()
    
    labeled_data = []
    skipped = 0
    
    with open(labeled_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "doc_chunk_id" not in row or "expected_claim_count" not in row:
                skipped += 1
                logging.warning(f"Skipping row missing required columns: {row}")
                continue
                
            try:
                chunk_id = int(row["doc_chunk_id"])
                # If they left it blank, skip it as unlabeled
                if not row["expected_claim_count"].strip():
                    skipped += 1
                    continue
                    
                expected_count = int(row["expected_claim_count"])
                labeled_data.append((chunk_id, expected_count))
            except ValueError:
                skipped += 1
                logging.warning(f"Skipping malformed row (non-integer): {row}")
                
    if skipped > 0:
        logging.warning(f"Skipped {skipped} malformed or unlabeled rows in {labeled_path}")
        
    zero_empty_correct = 0
    zero_empty_total = 0
    
    nonzero_correct = 0
    nonzero_total = 0
    
    exact_match = 0
    within_one = 0
    
    for chunk_id, expected_count in labeled_data:
        cursor.execute("SELECT COUNT(*) FROM claims WHERE doc_chunk_id = ?", (chunk_id,))
        actual_count = cursor.fetchone()[0]
        
        if expected_count == 0:
            zero_empty_total += 1
            if actual_count == 0:
                zero_empty_correct += 1
        else:
            nonzero_total += 1
            if actual_count >= 1:
                nonzero_correct += 1
                
        if actual_count == expected_count:
            exact_match += 1
            
        if abs(actual_count - expected_count) <= 1:
            within_one += 1
            
    total_labeled = len(labeled_data)
    
    return {
        "labeled_count": total_labeled,
        "zero_empty_correct": zero_empty_correct,
        "zero_empty_total": zero_empty_total,
        "nonzero_correct": nonzero_correct,
        "nonzero_total": nonzero_total,
        "exact_match": exact_match,
        "within_one": within_one
    }
