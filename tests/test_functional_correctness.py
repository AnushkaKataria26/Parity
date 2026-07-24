import os
import sqlite3
import pytest
from unittest.mock import patch

from parity.chunking.ast_chunker import extract_chunks_from_file
from parity.chunking.doc_chunker import extract_chunks_from_markdown
from parity.embedding.model import build_code_chunk_embedding_text, build_doc_chunk_embedding_text
from parity.retrieval.retriever import retrieve_for_claim
from parity.verification.resolve import ResolvedSymbol
from parity.verification.verify import (
    verify_signature_claim,
    verify_default_value_claim,
    verify_return_type_claim
)
from parity.reporting.build_report import build_drift_report

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

def test_ast_code_chunker_correctness():
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures/functional/fixture_a.py")
    chunks, _ = extract_chunks_from_file(fixture_path, os.path.dirname(fixture_path))
    
    assert len(chunks) == 9
    
    symbols = {c.symbol_name: c for c in chunks}
    
    assert "top_level_fn" in symbols
    assert "top_level_fn.inner_fn" in symbols
    assert "Outer" in symbols
    assert "Outer.__init__" in symbols
    assert "Outer.static_method" in symbols
    assert "Outer.prop" in symbols
    assert "Outer.Inner" in symbols
    assert "Outer.Inner.inner_method" in symbols
    assert "async_top_level" in symbols
    
    inner_fn = symbols["top_level_fn.inner_fn"]
    assert inner_fn.symbol_type == "function"
    
    static_method = symbols["Outer.static_method"]
    assert static_method.symbol_type == "method"
    
    prop = symbols["Outer.prop"]
    assert prop.symbol_type == "method"
    
    # In fixture_a.py, @property is on line 23
    assert prop.start_line == 23
    
    async_top = symbols["async_top_level"]
    assert async_top.symbol_type == "async_function"
    
    top_level = symbols["top_level_fn"]
    assert top_level.docstring == "Top level docstring."
    
    outer = symbols["Outer"]
    inner_class = symbols["Outer.Inner"]
    assert outer.ast_hash != inner_class.ast_hash

def test_doc_chunker_correctness():
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures/functional/fixture_b.md")
    chunks = list(extract_chunks_from_markdown(fixture_path, os.path.dirname(fixture_path)))
    
    assert len(chunks) == 5
    
    paths = [c.heading_path for c in chunks]
    assert paths == [
        "MyProject",
        "MyProject > Installation",
        "MyProject > Installation > Requirements",
        "MyProject > Usage",
        "MyProject > Usage"
    ]
    
    req_chunk = chunks[2]
    assert req_chunk.heading_path == "MyProject > Installation > Requirements"
    
    usage1_chunk = chunks[3]
    usage2_chunk = chunks[4]
    
    assert usage1_chunk.heading_path == "MyProject > Usage"
    assert usage2_chunk.heading_path == "MyProject > Usage"
    
    assert len(usage1_chunk.code_blocks) == 1
    assert "# this is a heading-looking comment: # Fake Heading" in usage1_chunk.code_blocks[0]
    
    assert "# Fake Heading" not in usage1_chunk.text

def test_embedding_text_construction_correctness():
    chunk_row = {
        "symbol_name": "Outer.static_method",
        "symbol_type": "method"
    }
    body_json = {
        "source_text": "@staticmethod\ndef static_method(y):\n    return y",
        "docstring": None
    }
    
    code_text = build_code_chunk_embedding_text(chunk_row, body_json)
    assert "method Outer.static_method" in code_text
    assert "def static_method(y):" in code_text
    assert "@staticmethod" not in code_text
    assert "return y" not in code_text
    assert "Represent this sentence for retrieval:" not in code_text
    
    doc_chunk_row = {
        "heading": "MyProject > Usage",
        "text": "Some usage prose."
    }
    doc_body = {}
    doc_text = build_doc_chunk_embedding_text(doc_chunk_row, doc_body)
    assert doc_text == "MyProject > Usage\\nSome usage prose."
    assert "Represent this sentence for retrieval:" not in doc_text

@pytest.mark.parametrize("guess, candidates_data, expected_status, expected_matched_symbol", [
    ("retry", [("retry", 0.60)], "matched", "retry"),
    ("retry", [("utils.retry", 0.60)], "matched", "utils.retry"),
    ("Retry", [("retry", 0.60)], "matched", "retry"),
    ("retry", [("A.retry", 0.70), ("B.retry", 0.68)], "ambiguous", None), # margin 0.02 <= 0.05
    ("retry", [("A.retry", 0.75), ("B.retry", 0.68)], "matched", "A.retry"), # margin 0.07 > 0.05
    (None, [("connect", 0.80), ("disconnect", 0.50)], "matched", "connect"), # clears 0.55 floor, margin 0.30 > 0.05
    (None, [("connect", 0.50)], "no_match", None), # below floor
    (None, [("connect", 0.60), ("disconnect", 0.58)], "ambiguous", None), # margin 0.02 <= 0.05
    ("foo", [("bar", 0.90)], "matched", "bar"), # falls through to embedding-only, single candidate
    ("foo", [], "no_match", None),
    (None, [("only_one", 0.56)], "matched", "only_one"), # single candidate clears floor
    (None, [("only_one", 0.54)], "no_match", None), # single candidate below floor
])
@patch("parity.retrieval.retriever.embed_texts")
def test_retrieval_disambiguation(mock_embed, guess, candidates_data, expected_status, expected_matched_symbol):
    mock_embed.return_value = [[0.1, 0.2, 0.3]]
    
    ids = [f"code_chunk_{i}" for i in range(len(candidates_data))]
    distances = [1.0 - c[1] for c in candidates_data]
    metadatas = [{"symbol_name": c[0]} for c in candidates_data]
    
    collection = MockChromaCollection(ids, distances, metadatas)
    
    claim_row = {
        "id": 1,
        "claim_text": "text",
        "referenced_symbol_guess": guess
    }
    
    result = retrieve_for_claim(claim_row, {"repo_id": 1}, collection, "dummy")
    
    assert result.match_status == expected_status
    if expected_matched_symbol is None:
        assert result.matched_code_chunk_id is None
    else:
        # Check that matched_code_chunk_id corresponds to expected_matched_symbol
        found = False
        for i, c in enumerate(candidates_data):
            if c[0] == expected_matched_symbol:
                assert result.matched_code_chunk_id == i
                found = True
                break
        assert found

@pytest.mark.parametrize("claimed_params, actual_params, has_kwargs, expected_status", [
    (["timeout"], [{"name": "timeout", "kind": "POSITIONAL_OR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True}, {"name": "retries", "kind": "POSITIONAL_OR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True}], False, "Verified"),
    (["timeout", "bogus"], [{"name": "timeout", "kind": "POSITIONAL_OR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True}, {"name": "retries", "kind": "POSITIONAL_OR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True}], False, "Contradicted"),
    (["bogus"], [{"name": "timeout", "kind": "POSITIONAL_OR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True}], True, "Unverifiable"),
    ([], [{"name": "timeout", "kind": "POSITIONAL_OR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True}], False, "Unverifiable"),
])
def test_verify_signature_claim_logic(claimed_params, actual_params, has_kwargs, expected_status):
    if has_kwargs:
        actual_params = actual_params + [{"name": "kwargs", "kind": "VAR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True}]
        
    resolved = ResolvedSymbol("function", actual_params, None, True)
    res = verify_signature_claim({"param_names": claimed_params}, resolved)
    assert res.status == expected_status

@pytest.mark.parametrize("claimed_param, claimed_default, actual_params, has_kwargs, expected_status, expected_actual", [
    ("timeout", "30", [{"name": "timeout", "kind": "POSITIONAL_OR_KEYWORD", "has_default": True, "default_repr": "30", "is_literal": True}], False, "Verified", "30"),
    ("timeout", "30", [{"name": "timeout", "kind": "POSITIONAL_OR_KEYWORD", "has_default": True, "default_repr": "60", "is_literal": True}], False, "Contradicted", "60"),
    ("timeout", "30", [{"name": "timeout", "kind": "POSITIONAL_OR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True}], False, "Contradicted", "no default"),
    ("timeout", "30", [], False, "Contradicted", "parameter not found"),
    ("timeout", "30", [], True, "Unverifiable", None),
    ("timeout", "DEFAULT_TIMEOUT", [{"name": "timeout", "kind": "POSITIONAL_OR_KEYWORD", "has_default": True, "default_repr": "DEFAULT_TIMEOUT", "is_literal": False}], False, "Unverifiable", "DEFAULT_TIMEOUT"),
    ("timeout", "None", [{"name": "timeout", "kind": "POSITIONAL_OR_KEYWORD", "has_default": True, "default_repr": "None", "is_literal": True}], False, "Verified", "None"),
    ("timeout", "null", [{"name": "timeout", "kind": "POSITIONAL_OR_KEYWORD", "has_default": True, "default_repr": "None", "is_literal": True}], False, "Verified", "None"),
    ("timeout", "30", [{"name": "timeout", "kind": "POSITIONAL_OR_KEYWORD", "has_default": True, "default_repr": "30.0", "is_literal": True}], False, "Verified", "30.0"),
])
def test_verify_default_value_claim_logic(claimed_param, claimed_default, actual_params, has_kwargs, expected_status, expected_actual):
    if has_kwargs:
        actual_params = actual_params + [{"name": "kwargs", "kind": "VAR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True}]
        
    resolved = ResolvedSymbol("function", actual_params, None, True)
    res = verify_default_value_claim({"param_name": claimed_param, "default": claimed_default}, resolved)
    assert res.status == expected_status
    if expected_actual is not None:
        assert res.actual_value == expected_actual

@pytest.mark.parametrize("claimed_type, actual_return, expected_status", [
    ("list", "List[Result]", "Verified"),
    ("dict", "List[Result]", "Contradicted"),
    ("Result", None, "Unverifiable"),
])
def test_verify_return_type_claim_logic(claimed_type, actual_return, expected_status):
    resolved = ResolvedSymbol("function", [], actual_return, True)
    res = verify_return_type_claim({"return_type": claimed_type}, resolved)
    assert res.status == expected_status

def test_drift_report_aggregation_sort_order(tmp_db_conn):
    cursor = tmp_db_conn.cursor()
    
    # Setup some test data in tmp_db_conn
    cursor.execute("INSERT INTO repos (name, path, last_ingested_commit_sha) VALUES (?, ?, ?)", ("test_repo", "/test", "now"))
    repo_id = cursor.lastrowid
    
    cursor.execute("INSERT INTO doc_chunks (repo_id, file_path, heading, text) VALUES (?, ?, ?, ?)", (repo_id, "file_a.md", "h1", "text"))
    chunk_a = cursor.lastrowid
    cursor.execute("INSERT INTO doc_chunks (repo_id, file_path, heading, text) VALUES (?, ?, ?, ?)", (repo_id, "file_b.md", "h1", "text"))
    chunk_b = cursor.lastrowid
    cursor.execute("INSERT INTO doc_chunks (repo_id, file_path, heading, text) VALUES (?, ?, ?, ?)", (repo_id, "file_c.md", "h1", "text"))
    chunk_c = cursor.lastrowid
    
    def insert_claim(chunk_id, status, text="test claim"):
        cursor.execute("INSERT INTO claims (doc_chunk_id, claim_type, claim_text) VALUES (?, ?, ?)", (chunk_id, "signature", text))
        claim_id = cursor.lastrowid
        cursor.execute("INSERT INTO retrieval_results (claim_id, match_status, top_k_json, retrieved_at) VALUES (?, 'matched', '[]', 'now')", (claim_id,))
        cursor.execute("INSERT INTO verification_results (claim_id, status, verified_at) VALUES (?, ?, 'now')", (claim_id, status))
        
    # file_a.md: 1 Contradicted, 2 Verified
    insert_claim(chunk_a, "Contradicted")
    insert_claim(chunk_a, "Verified")
    insert_claim(chunk_a, "Verified")
    
    # file_b.md: 2 Unverifiable
    insert_claim(chunk_b, "Unverifiable")
    insert_claim(chunk_b, "Unverifiable")
    
    # file_c.md: 3 Verified
    insert_claim(chunk_c, "Verified")
    insert_claim(chunk_c, "Verified")
    insert_claim(chunk_c, "Verified")
    
    tmp_db_conn.commit()
    
    report = build_drift_report(tmp_db_conn, repo_id)
    
    file_keys = list(report.entries_by_file.keys())
    assert file_keys == ["file_a.md", "file_b.md", "file_c.md"]
    
    entries_a = report.entries_by_file["file_a.md"]
    assert entries_a[0].status == "Contradicted"
    assert entries_a[1].status == "Verified"
    assert entries_a[2].status == "Verified"
    
    assert report.totals["contradicted"] == 1
    assert report.totals["unverifiable"] == 2
    assert report.totals["verified"] == 5
    
    # Check verbose / non-verbose logic directly if report object exposes it, 
    # but the requirement states "Non-verbose text render omits file_c.md's section entirely; verbose render includes it."
    # Let's test render text if possible
    from parity.reporting.render_text import render_text_report
    
    text_verbose = render_text_report(report, verbose=True)
    text_non_verbose = render_text_report(report, verbose=False)
    
    assert "file_c.md" in text_verbose
    assert "file_c.md" not in text_non_verbose
