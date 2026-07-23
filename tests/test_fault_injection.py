import os
import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from parity.evaluation.fault_injection import select_fault_targets, inject_faults, apply_fault_to_file, InjectedFault
from parity.evaluation.run_fault_eval import run_fault_injection_eval
from parity.db.migrate import apply_schema

@pytest.fixture
def mem_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    yield conn
    conn.close()

def test_select_fault_targets(mem_db):
    conn = mem_db
    conn.execute("INSERT INTO repos (id, name, path, last_ingested_commit_sha) VALUES (1, 'test', '/tmp/test', 'sha')")
    
    # 2 chunks
    conn.execute("INSERT INTO code_chunks (id, repo_id, file_path, symbol_name, symbol_type, start_line, end_line) VALUES (1, 1, 'f1.py', 'func1', 'function', 1, 5)")
    conn.execute("INSERT INTO code_chunks (id, repo_id, file_path, symbol_name, symbol_type, start_line, end_line) VALUES (2, 1, 'f2.py', 'func2', 'function', 6, 10)")
    
    conn.execute("INSERT INTO doc_chunks (id, repo_id, file_path, text) VALUES (1, 1, 'd1.md', 'text')")
    
    # Claim 1: signature (matched to chunk 1)
    conn.execute("INSERT INTO claims (id, doc_chunk_id, claim_text, claim_type) VALUES (1, 1, 'claim 1', 'signature')")
    conn.execute("INSERT INTO retrieval_results (id, claim_id, matched_code_chunk_id, match_status, top_k_json, retrieved_at) VALUES (1, 1, 1, 'matched', '[]', 'time')")
    
    # Claim 2: behavior (matched to chunk 2) - should be ignored
    conn.execute("INSERT INTO claims (id, doc_chunk_id, claim_text, claim_type) VALUES (2, 1, 'claim 2', 'behavior')")
    conn.execute("INSERT INTO retrieval_results (id, claim_id, matched_code_chunk_id, match_status, top_k_json, retrieved_at) VALUES (2, 2, 2, 'matched', '[]', 'time')")
    
    # Claim 3: default_value (matched to chunk 1)
    conn.execute("INSERT INTO claims (id, doc_chunk_id, claim_text, claim_type) VALUES (3, 1, 'claim 3', 'default_value')")
    conn.execute("INSERT INTO retrieval_results (id, claim_id, matched_code_chunk_id, match_status, top_k_json, retrieved_at) VALUES (3, 3, 1, 'matched', '[]', 'time')")
    
    targets = select_fault_targets(conn, 1, n=5)
    
    # chunk 1 is eligible. Chunk 2 is not (behavior).
    # Since we grouped by chunk_id, there should be exactly 1 target.
    assert len(targets) == 1
    assert targets[0]["chunk_id"] == 1
    
    # Check fallback when asking for more than available
    targets2 = select_fault_targets(conn, 1, n=10)
    assert len(targets2) == 1

def test_inject_faults_safety_check(mem_db):
    mem_db.execute("INSERT INTO repos (id, name, path, last_ingested_commit_sha) VALUES (1, 'test', '/tmp/real_repo', 'sha')")
    
    with pytest.raises(ValueError, match="repo_copy_path must never be the original"):
        inject_faults(mem_db, 1, "/tmp/real_repo", [])
        
def test_apply_fault_to_file_rename_param(tmp_path):
    f_path = tmp_path / "test.py"
    original_code = "def foo(data):\n    metadata = 1\n    return data\n"
    f_path.write_text(original_code, encoding="utf-8")
    
    # Mutate data -> data_mutated, should NOT touch metadata
    mutated_snippet = "def foo(data_mutated):\n    metadata = 1\n    return data_mutated\n"
    
    fault = InjectedFault(
        fault_id="1", file_path="test.py", symbol_name="foo",
        fault_type="rename_param", original_snippet=original_code,
        mutated_snippet=mutated_snippet, target_claim_hint="",
        start_line=1, end_line=3
    )
    
    apply_fault_to_file(str(f_path), fault)
    
    res = f_path.read_text(encoding="utf-8")
    assert "def foo(data_mutated):" in res
    assert "metadata = 1" in res
    assert "return data_mutated" in res
    assert "return metadata" not in res
    assert res == mutated_snippet

def test_apply_fault_to_file_change_default(tmp_path):
    f_path = tmp_path / "test.py"
    original_code = "def foo(a=1):\n    return a\n"
    f_path.write_text(original_code, encoding="utf-8")
    
    mutated_snippet = "def foo(a=43):\n    return a\n"
    
    fault = InjectedFault(
        fault_id="1", file_path="test.py", symbol_name="foo",
        fault_type="change_default", original_snippet=original_code,
        mutated_snippet=mutated_snippet, target_claim_hint="",
        start_line=1, end_line=2
    )
    
    apply_fault_to_file(str(f_path), fault)
    
    res = f_path.read_text(encoding="utf-8")
    assert res == mutated_snippet

@patch("parity.evaluation.run_fault_eval.select_fault_targets")
@patch("parity.evaluation.run_fault_eval.inject_faults")
@patch("parity.evaluation.run_fault_eval.discover_python_files")
@patch("parity.evaluation.run_fault_eval.extract_chunks_from_file")
@patch("parity.evaluation.run_fault_eval.store_chunks")
@patch("parity.evaluation.run_fault_eval.get_chroma_client")
@patch("parity.evaluation.run_fault_eval.get_or_create_collections")
@patch("parity.evaluation.run_fault_eval.embed_repo")
@patch("parity.evaluation.run_fault_eval.retrieve_for_repo")
@patch("parity.evaluation.run_fault_eval.verify_repo")
def test_run_fault_injection_eval_cleanup_on_exception(
    mock_verify, mock_retrieve, mock_embed, mock_get_coll, mock_get_client,
    mock_store, mock_extract, mock_discover, mock_inject, mock_select, mem_db, tmp_path
):
    mem_db.execute("INSERT INTO repos (id, name, path, last_ingested_commit_sha) VALUES (1, 'test', '/tmp/test_real', 'sha')")
    
    mock_select.return_value = [{"chunk_id": 1, "symbol_name": "foo", "file_path": "foo.py"}]
    mock_inject.return_value = [InjectedFault("1", "foo.py", "foo", "rename_param", "", "", "", 1, 1)]
    mock_discover.return_value = []
    mock_extract.return_value = ([], False)
    mock_get_coll.return_value = (MagicMock(), MagicMock())
    
    mock_verify.side_effect = Exception("Forced crash")
    
    config = {"chroma_persist_dir": "", "ollama_model": "", "ollama_host": ""}
    
    with pytest.raises(Exception, match="Forced crash"):
        run_fault_injection_eval(mem_db, 1, str(tmp_path), config, n_faults=1)
        
    # Check cleanup
    # There should only be the original repo left
    cursor = mem_db.execute("SELECT id FROM repos")
    repos = cursor.fetchall()
    assert len(repos) == 1
    assert repos[0][0] == 1
