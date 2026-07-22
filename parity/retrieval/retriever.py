import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, List

from parity.embedding.model import embed_texts

SIMILARITY_FLOOR = 0.55
AMBIGUITY_MARGIN = 0.05

@dataclass
class RetrievalCandidate:
    code_chunk_id: int
    symbol_name: str
    score: float

@dataclass
class RetrievalResult:
    claim_id: int
    matched_code_chunk_id: Optional[int]
    match_status: str
    top_k: List[RetrievalCandidate]

def retrieve_for_claim(claim_row: dict, doc_chunk_row: dict, code_collection, model_name: str, top_k: int = 5) -> RetrievalResult:
    """
    Retrieves candidate code chunks for a claim and disambiguates to find the best match.
    
    We don't skip retrieval for "behavior" claims because Phase 7's report benefits from a 
    matched-symbol citation even for behavior claims, keeping the logic simple.
    """
    claim_id = claim_row["id"]
    claim_text = claim_row["claim_text"]
    guess = claim_row.get("referenced_symbol_guess")
    
    query_text = claim_text
    if guess:
        query_text = f"{claim_text}\nSymbol: {guess}"
        
    embeddings = embed_texts([query_text], model_name=model_name, batch_size=1)
    vec = embeddings[0] if embeddings else None
    
    if vec is None:
        logging.warning(f"Warning: Empty embedding for claim {claim_id}")
        return RetrievalResult(claim_id, None, "no_match", [])
        
    repo_id = doc_chunk_row["repo_id"]
    
    results = code_collection.query(
        query_embeddings=[vec],
        n_results=top_k,
        where={"repo_id": repo_id}
    )
    
    candidates = []
    
    ids_list = results.get("ids", [])
    distances_list = results.get("distances", [])
    metadatas_list = results.get("metadatas", [])
    
    if not ids_list or not ids_list[0]:
        return RetrievalResult(claim_id, None, "no_match", [])
        
    retrieved_ids = ids_list[0]
    retrieved_dists = distances_list[0]
    retrieved_metas = metadatas_list[0]
    
    for i in range(len(retrieved_ids)):
        chroma_id = retrieved_ids[i]
        dist = retrieved_dists[i]
        meta = retrieved_metas[i]
        
        c_id = int(chroma_id.replace("code_chunk_", ""))
        symbol_name = meta.get("symbol_name", "")
        
        # Chroma hnsw:space is set to "cosine". 
        # For cosine space, Chroma returns cosine distance d. 
        # So we convert distance to similarity explicitly: similarity = 1 - d
        score = 1.0 - dist
        
        candidates.append(RetrievalCandidate(
            code_chunk_id=c_id,
            symbol_name=symbol_name,
            score=score
        ))
        
    # Sort candidates by score descending
    candidates.sort(key=lambda x: x.score, reverse=True)
    
    match_status = "no_match"
    matched_id = None
    
    exact_matches = []
    if guess:
        # Case sensitive exact match or trailing segment match
        for c in candidates:
            if c.symbol_name == guess or c.symbol_name.endswith(f".{guess}"):
                exact_matches.append(c)
                
        # Case insensitive pass if no exact match
        if not exact_matches:
            guess_lower = guess.lower()
            for c in candidates:
                if c.symbol_name.lower() == guess_lower or c.symbol_name.lower().endswith(f".{guess_lower}"):
                    exact_matches.append(c)
                    
    if exact_matches:
        if len(exact_matches) == 1:
            match_status = "matched"
            matched_id = exact_matches[0].code_chunk_id
        else:
            # Sort exact matches by score descending
            exact_matches.sort(key=lambda x: x.score, reverse=True)
            top_score = exact_matches[0].score
            second_score = exact_matches[1].score
            if top_score - second_score > AMBIGUITY_MARGIN:
                match_status = "matched"
                matched_id = exact_matches[0].code_chunk_id
            else:
                match_status = "ambiguous"
                matched_id = None
    else:
        # Embedding similarity alone
        top_score = candidates[0].score
        if top_score > SIMILARITY_FLOOR:
            if len(candidates) > 1:
                second_score = candidates[1].score
                if top_score - second_score > AMBIGUITY_MARGIN:
                    match_status = "matched"
                    matched_id = candidates[0].code_chunk_id
                else:
                    match_status = "ambiguous"
                    matched_id = None
            else:
                # Only 1 candidate
                match_status = "matched"
                matched_id = candidates[0].code_chunk_id
        else:
            match_status = "no_match"
            matched_id = None
            
    return RetrievalResult(
        claim_id=claim_id,
        matched_code_chunk_id=matched_id,
        match_status=match_status,
        top_k=candidates
    )

def retrieve_for_repo(conn, repo_id: int, code_collection, model_name: str, top_k: int = 5) -> dict:
    import sqlite3
    # Use row factory temporarily to get keys easily if not already
    old_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT c.*, d.repo_id, d.file_path, d.heading, d.text as doc_text
        FROM claims c
        JOIN doc_chunks d ON c.doc_chunk_id = d.id
        WHERE d.repo_id = ?
    """, (repo_id,))
    
    rows = cursor.fetchall()
    
    # Restore factory for general use (if needed)
    conn.row_factory = old_factory
    
    cursor = conn.cursor()
    # Delete existing retrieval_results for claims belonging to this repo
    cursor.execute("""
        DELETE FROM retrieval_results 
        WHERE claim_id IN (
            SELECT c.id FROM claims c 
            JOIN doc_chunks d ON c.doc_chunk_id = d.id 
            WHERE d.repo_id = ?
        )
    """, (repo_id,))
    
    total_claims = len(rows)
    matched = 0
    ambiguous = 0
    no_match = 0
    
    for row_tuple in rows:
        claim_row = {
            "id": row_tuple["id"],
            "doc_chunk_id": row_tuple["doc_chunk_id"],
            "claim_text": row_tuple["claim_text"],
            "claim_type": row_tuple["claim_type"],
            "referenced_symbol_guess": row_tuple["referenced_symbol_guess"]
        }
        
        doc_chunk_row = {
            "id": row_tuple["doc_chunk_id"],
            "repo_id": row_tuple["repo_id"],
            "file_path": row_tuple["file_path"],
            "heading": row_tuple["heading"],
            "text": row_tuple["doc_text"]
        }
        
        res = retrieve_for_claim(claim_row, doc_chunk_row, code_collection, model_name, top_k)
        
        top_k_json = json.dumps([asdict(c) for c in res.top_k])
        now_str = datetime.now(timezone.utc).isoformat()
        
        cursor.execute("""
            INSERT INTO retrieval_results (claim_id, matched_code_chunk_id, match_status, top_k_json, retrieved_at)
            VALUES (?, ?, ?, ?, ?)
        """, (res.claim_id, res.matched_code_chunk_id, res.match_status, top_k_json, now_str))
        
        if res.match_status == "matched":
            matched += 1
        elif res.match_status == "ambiguous":
            ambiguous += 1
        else:
            no_match += 1
            
    conn.commit()
    
    return {
        "total_claims": total_claims,
        "matched": matched,
        "ambiguous": ambiguous,
        "no_match": no_match
    }
