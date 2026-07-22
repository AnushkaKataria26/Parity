import json
import sqlite3
import pytest
from unittest.mock import MagicMock, patch

from parity.retrieval.retriever import (
    retrieve_for_claim,
    retrieve_for_repo,
    RetrievalCandidate,
    SIMILARITY_FLOOR,
    AMBIGUITY_MARGIN
)

class MockChromaCollection:
    def __init__(self, ids=None, distances=None, metadatas=None):
        self._ids = ids or []
        self._distances = distances or []
        self._metadatas = metadatas or []
        
    def query(self, query_embeddings, n_results, where=None):
        # Truncate to n_results
        return {
            "ids": [self._ids[:n_results]] if self._ids else [],
            "distances": [self._distances[:n_results]] if self._distances else [],
            "metadatas": [self._metadatas[:n_results]] if self._metadatas else []
        }

@patch("parity.retrieval.retriever.embed_texts")
def test_exact_single_symbol_match(mock_embed):
    mock_embed.return_value = [[0.1, 0.2, 0.3]]
    
    # 3 candidates
    ids = ["code_chunk_10", "code_chunk_11", "code_chunk_12"]
    # distances corresponding to similarities: 0.9, 0.8, 0.7
    distances = [0.1, 0.2, 0.3]
    metadatas = [
        {"symbol_name": "something_else"},
        {"symbol_name": "my_exact_match"},
        {"symbol_name": "another_thing"}
    ]
    
    collection = MockChromaCollection(ids, distances, metadatas)
    
    claim_row = {
        "id": 1,
        "claim_text": "This calls my_exact_match.",
        "referenced_symbol_guess": "my_exact_match"
    }
    
    doc_chunk_row = {"repo_id": 1}
    
    result = retrieve_for_claim(claim_row, doc_chunk_row, collection, "dummy")
    
    assert result.match_status == "matched"
    assert result.matched_code_chunk_id == 11
    
@patch("parity.retrieval.retriever.embed_texts")
def test_exact_trailing_segment_match(mock_embed):
    mock_embed.return_value = [[0.1]]
    
    ids = ["code_chunk_1"]
    distances = [0.2]
    metadatas = [{"symbol_name": "Client.connect"}]
    
    collection = MockChromaCollection(ids, distances, metadatas)
    
    claim_row = {
        "id": 1,
        "claim_text": "Connects to server",
        "referenced_symbol_guess": "connect"
    }
    
    result = retrieve_for_claim(claim_row, {"repo_id": 1}, collection, "dummy")
    
    assert result.match_status == "matched"
    assert result.matched_code_chunk_id == 1

@patch("parity.retrieval.retriever.embed_texts")
def test_exact_match_case_insensitive(mock_embed):
    mock_embed.return_value = [[0.1]]
    
    ids = ["code_chunk_2"]
    distances = [0.2]
    metadatas = [{"symbol_name": "retry"}]
    
    collection = MockChromaCollection(ids, distances, metadatas)
    
    claim_row = {
        "id": 1,
        "claim_text": "Retries connection",
        "referenced_symbol_guess": "Retry"
    }
    
    result = retrieve_for_claim(claim_row, {"repo_id": 1}, collection, "dummy")
    
    assert result.match_status == "matched"
    assert result.matched_code_chunk_id == 2

@patch("parity.retrieval.retriever.embed_texts")
def test_multiple_exact_matches_clear_margin(mock_embed):
    mock_embed.return_value = [[0.1]]
    
    ids = ["code_chunk_1", "code_chunk_2"]
    # Distances 0.1, 0.3 -> Scores 0.9, 0.7. Margin = 0.2 > 0.05
    distances = [0.1, 0.3]
    metadatas = [{"symbol_name": "retry"}, {"symbol_name": "Client.retry"}]
    
    collection = MockChromaCollection(ids, distances, metadatas)
    
    claim_row = {
        "id": 1,
        "claim_text": "Retries connection",
        "referenced_symbol_guess": "retry"
    }
    
    result = retrieve_for_claim(claim_row, {"repo_id": 1}, collection, "dummy")
    
    assert result.match_status == "matched"
    assert result.matched_code_chunk_id == 1

@patch("parity.retrieval.retriever.embed_texts")
def test_multiple_exact_matches_ambiguous(mock_embed):
    mock_embed.return_value = [[0.1]]
    
    ids = ["code_chunk_1", "code_chunk_2"]
    # Distances 0.1, 0.12 -> Scores 0.9, 0.88. Margin = 0.02 < 0.05
    distances = [0.1, 0.12]
    metadatas = [{"symbol_name": "retry"}, {"symbol_name": "Client.retry"}]
    
    collection = MockChromaCollection(ids, distances, metadatas)
    
    claim_row = {
        "id": 1,
        "claim_text": "Retries connection",
        "referenced_symbol_guess": "retry"
    }
    
    result = retrieve_for_claim(claim_row, {"repo_id": 1}, collection, "dummy")
    
    assert result.match_status == "ambiguous"
    assert result.matched_code_chunk_id is None

@patch("parity.retrieval.retriever.embed_texts")
def test_no_exact_match_matched(mock_embed):
    mock_embed.return_value = [[0.1]]
    
    ids = ["code_chunk_1", "code_chunk_2"]
    # Distances 0.2 (score 0.8), 0.3 (score 0.7). Margin = 0.1 > 0.05, top > 0.55
    distances = [0.2, 0.3]
    metadatas = [{"symbol_name": "do_connect"}, {"symbol_name": "something_else"}]
    
    collection = MockChromaCollection(ids, distances, metadatas)
    
    claim_row = {
        "id": 1,
        "claim_text": "Connects to server",
        "referenced_symbol_guess": None
    }
    
    result = retrieve_for_claim(claim_row, {"repo_id": 1}, collection, "dummy")
    
    assert result.match_status == "matched"
    assert result.matched_code_chunk_id == 1

@patch("parity.retrieval.retriever.embed_texts")
def test_no_exact_match_no_match_below_floor(mock_embed):
    mock_embed.return_value = [[0.1]]
    
    ids = ["code_chunk_1", "code_chunk_2"]
    # Distances 0.6 (score 0.4), 0.7 (score 0.3). top < 0.55
    distances = [0.6, 0.7]
    metadatas = [{"symbol_name": "do_connect"}, {"symbol_name": "something_else"}]
    
    collection = MockChromaCollection(ids, distances, metadatas)
    
    claim_row = {
        "id": 1,
        "claim_text": "Connects to server",
        "referenced_symbol_guess": None
    }
    
    result = retrieve_for_claim(claim_row, {"repo_id": 1}, collection, "dummy")
    
    assert result.match_status == "no_match"
    assert result.matched_code_chunk_id is None

@patch("parity.retrieval.retriever.embed_texts")
def test_no_exact_match_ambiguous(mock_embed):
    mock_embed.return_value = [[0.1]]
    
    ids = ["code_chunk_1", "code_chunk_2"]
    # Distances 0.2 (score 0.8), 0.22 (score 0.78). Margin = 0.02 < 0.05
    distances = [0.2, 0.22]
    metadatas = [{"symbol_name": "do_connect"}, {"symbol_name": "connect"}]
    
    collection = MockChromaCollection(ids, distances, metadatas)
    
    claim_row = {
        "id": 1,
        "claim_text": "Connects to server",
        "referenced_symbol_guess": None # Let's say no guess to trigger fallback
    }
    
    result = retrieve_for_claim(claim_row, {"repo_id": 1}, collection, "dummy")
    
    assert result.match_status == "ambiguous"
    assert result.matched_code_chunk_id is None

@patch("parity.retrieval.retriever.embed_texts")
def test_only_one_candidate(mock_embed):
    mock_embed.return_value = [[0.1]]
    
    ids = ["code_chunk_1"]
    # Distance 0.2 (score 0.8) -> above floor
    distances = [0.2]
    metadatas = [{"symbol_name": "do_connect"}]
    
    collection = MockChromaCollection(ids, distances, metadatas)
    
    claim_row = {
        "id": 1,
        "claim_text": "Connects to server",
        "referenced_symbol_guess": None
    }
    
    result = retrieve_for_claim(claim_row, {"repo_id": 1}, collection, "dummy", top_k=5)
    
    assert result.match_status == "matched"
    assert result.matched_code_chunk_id == 1
    assert len(result.top_k) == 1

@patch("parity.retrieval.retriever.embed_texts")
def test_zero_candidates(mock_embed):
    mock_embed.return_value = [[0.1]]
    
    collection = MockChromaCollection([], [], [])
    
    claim_row = {
        "id": 1,
        "claim_text": "Connects to server",
        "referenced_symbol_guess": None
    }
    
    result = retrieve_for_claim(claim_row, {"repo_id": 1}, collection, "dummy")
    
    assert result.match_status == "no_match"
    assert result.matched_code_chunk_id is None
    assert len(result.top_k) == 0

@patch("parity.retrieval.retriever.retrieve_for_claim")
def test_retrieve_for_repo_aggregates_and_replaces(mock_rfc):
    conn = sqlite3.connect(":memory:")
    
    # Setup schema
    from parity.db.schema import SCHEMA_STATEMENTS
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
        
    conn.execute("INSERT INTO repos (name, path) VALUES ('test', '/test')")
    conn.execute("INSERT INTO doc_chunks (repo_id, file_path, text) VALUES (1, 'doc.md', 'text')")
    # Insert 3 claims
    conn.execute("INSERT INTO claims (doc_chunk_id, claim_text, claim_type) VALUES (1, 'claim 1', 'behavior')")
    conn.execute("INSERT INTO claims (doc_chunk_id, claim_text, claim_type) VALUES (1, 'claim 2', 'behavior')")
    conn.execute("INSERT INTO claims (doc_chunk_id, claim_text, claim_type) VALUES (1, 'claim 3', 'behavior')")
    
    # Pre-populate retrieval_results with a stale row for this repo
    conn.execute("INSERT INTO retrieval_results (claim_id, matched_code_chunk_id, match_status, top_k_json, retrieved_at) VALUES (1, 99, 'matched', '[]', '2024-01-01')")
    
    # Mock responses
    def side_effect(claim_row, *args, **kwargs):
        from parity.retrieval.retriever import RetrievalResult, RetrievalCandidate
        if claim_row["id"] == 1:
            return RetrievalResult(1, 100, "matched", [RetrievalCandidate(100, "f1", 0.9)])
        elif claim_row["id"] == 2:
            return RetrievalResult(2, None, "ambiguous", [])
        else:
            return RetrievalResult(3, None, "no_match", [])
            
    mock_rfc.side_effect = side_effect
    
    collection = MockChromaCollection()
    
    summary = retrieve_for_repo(conn, 1, collection, "dummy")
    
    assert summary["total_claims"] == 3
    assert summary["matched"] == 1
    assert summary["ambiguous"] == 1
    assert summary["no_match"] == 1
    
    # Verify DB
    cursor = conn.cursor()
    cursor.execute("SELECT claim_id, match_status, top_k_json FROM retrieval_results ORDER BY claim_id")
    rows = cursor.fetchall()
    
    assert len(rows) == 3
    assert rows[0][0] == 1
    assert rows[0][1] == "matched"
    assert "f1" in rows[0][2]
    
    assert rows[1][0] == 2
    assert rows[1][1] == "ambiguous"
    
    assert rows[2][0] == 3
    assert rows[2][1] == "no_match"
