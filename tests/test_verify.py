import os
import sqlite3
import pytest
from unittest.mock import patch

from parity.verification.resolve import ResolvedSymbol
from parity.verification.verify import (
    verify_signature_claim,
    verify_default_value_claim,
    verify_env_var_claim,
    verify_return_type_claim,
    verify_repo
)
from parity.db.schema import SCHEMA_STATEMENTS

def test_verify_signature_claim():
    resolved = ResolvedSymbol("static", [
        {"name": "host", "kind": "POSITIONAL_OR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True},
        {"name": "port", "kind": "POSITIONAL_OR_KEYWORD", "has_default": True, "default_repr": "8080", "is_literal": True}
    ], None, True)
    
    # Present -> Verified
    res = verify_signature_claim({"param_names": ["host"]}, resolved)
    assert res.status == "Verified"
    
    # Absent, no kwargs -> Contradicted
    res = verify_signature_claim({"param_names": ["timeout"]}, resolved)
    assert res.status == "Contradicted"
    
    # Absent, kwargs present -> Unverifiable
    resolved_kwargs = ResolvedSymbol("static", [
        {"name": "host", "kind": "POSITIONAL_OR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True},
        {"name": "kwargs", "kind": "VAR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True}
    ], None, True)
    res = verify_signature_claim({"param_names": ["timeout"]}, resolved_kwargs)
    assert res.status == "Unverifiable"

def test_verify_default_value_claim():
    resolved = ResolvedSymbol("static", [
        {"name": "timeout", "kind": "POSITIONAL_OR_KEYWORD", "has_default": True, "default_repr": "30", "is_literal": True},
        {"name": "retry", "kind": "POSITIONAL_OR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True},
        {"name": "host", "kind": "POSITIONAL_OR_KEYWORD", "has_default": True, "default_repr": "CONST", "is_literal": False},
        {"name": "kwargs", "kind": "VAR_KEYWORD", "has_default": False, "default_repr": None, "is_literal": True}
    ], None, True)
    
    # Matching default -> Verified
    res = verify_default_value_claim({"param_name": "timeout", "default": "30"}, resolved)
    assert res.status == "Verified"
    
    # Mismatched default -> Contradicted
    res = verify_default_value_claim({"param_name": "timeout", "default": "60"}, resolved)
    assert res.status == "Contradicted"
    
    # Param doesn't exist, kwargs present -> Unverifiable
    res = verify_default_value_claim({"param_name": "missing", "default": "60"}, resolved)
    assert res.status == "Unverifiable"
    
    resolved_no_kwargs = ResolvedSymbol("static", [
        {"name": "timeout", "kind": "POSITIONAL_OR_KEYWORD", "has_default": True, "default_repr": "30", "is_literal": True}
    ], None, True)
    
    # Param doesn't exist, no kwargs -> Contradicted
    res = verify_default_value_claim({"param_name": "missing", "default": "60"}, resolved_no_kwargs)
    assert res.status == "Contradicted"
    
    # Param exists, no default -> Contradicted
    res = verify_default_value_claim({"param_name": "retry", "default": "True"}, resolved)
    assert res.status == "Contradicted"
    
    # Non-literal default -> Unverifiable
    res = verify_default_value_claim({"param_name": "host", "default": "localhost"}, resolved)
    assert res.status == "Unverifiable"

def test_verify_env_var_claim(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    
    file1 = repo_dir / "main.py"
    file1.write_text("import os\n", encoding="utf-8")
    
    file2 = repo_dir / "settings.py"
    file2.write_text("import os\nos.environ.get('DEBUG_MODE')\n", encoding="utf-8")
    
    # Found in matched file -> Verified
    res = verify_env_var_claim({"var_name": "DEBUG_MODE"}, str(repo_dir), "settings.py")
    assert res.status == "Verified"
    
    # Found in different file -> Verified
    res = verify_env_var_claim({"var_name": "DEBUG_MODE"}, str(repo_dir), "main.py")
    assert res.status == "Verified"
    
    # Not found anywhere -> Unverifiable
    res = verify_env_var_claim({"var_name": "MISSING_VAR"}, str(repo_dir), "main.py")
    assert res.status == "Unverifiable"

def test_verify_return_type_claim():
    resolved = ResolvedSymbol("static", [], "List[int]", True)
    
    # Substring match -> Verified
    res = verify_return_type_claim({"return_type": "list"}, resolved)
    assert res.status == "Verified"
    
    # No match -> Contradicted
    res = verify_return_type_claim({"return_type": "dict"}, resolved)
    assert res.status == "Contradicted"
    
    # No annotation -> Unverifiable
    resolved_no_ann = ResolvedSymbol("static", [], None, True)
    res = verify_return_type_claim({"return_type": "list"}, resolved_no_ann)
    assert res.status == "Unverifiable"

@patch("parity.verification.verify.extract_claimed_value")
def test_verify_repo_integration(mock_extract, tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    
    (repo_dir / "mod.py").write_text("def f(x=1):\n    pass\n", encoding="utf-8")
    
    db_path = tmp_path / "parity.db"
    conn = sqlite3.connect(db_path)
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
        
    conn.execute("INSERT INTO repos (name, path) VALUES ('repo', ?)", (str(repo_dir),))
    
    # Code chunk 1
    conn.execute("INSERT INTO code_chunks (repo_id, file_path, symbol_name, symbol_type, start_line, end_line) VALUES (1, 'mod.py', 'f', 'function', 1, 2)")
    
    # Doc chunk 1
    conn.execute("INSERT INTO doc_chunks (repo_id, file_path, text) VALUES (1, 'doc.md', 'text')")
    
    # Claim 1: Verified (default matches)
    conn.execute("INSERT INTO claims (doc_chunk_id, claim_text, claim_type) VALUES (1, 'default 1', 'default_value')")
    # Claim 2: Contradicted (default mismatch)
    conn.execute("INSERT INTO claims (doc_chunk_id, claim_text, claim_type) VALUES (1, 'default 2', 'default_value')")
    # Claim 3: Behavior -> Unverifiable
    conn.execute("INSERT INTO claims (doc_chunk_id, claim_text, claim_type) VALUES (1, 'behaves well', 'behavior')")
    # Claim 4: Ambiguous -> Unverifiable
    conn.execute("INSERT INTO claims (doc_chunk_id, claim_text, claim_type) VALUES (1, 'ambiguous', 'default_value')")
    
    # Retrieval results
    conn.execute("INSERT INTO retrieval_results (claim_id, matched_code_chunk_id, match_status, top_k_json, retrieved_at) VALUES (1, 1, 'matched', '[]', 'time')")
    conn.execute("INSERT INTO retrieval_results (claim_id, matched_code_chunk_id, match_status, top_k_json, retrieved_at) VALUES (2, 1, 'matched', '[]', 'time')")
    conn.execute("INSERT INTO retrieval_results (claim_id, matched_code_chunk_id, match_status, top_k_json, retrieved_at) VALUES (3, 1, 'matched', '[]', 'time')")
    conn.execute("INSERT INTO retrieval_results (claim_id, matched_code_chunk_id, match_status, top_k_json, retrieved_at) VALUES (4, NULL, 'ambiguous', '[]', 'time')")
    
    def extract_side_effect(text, ctype, model, host):
        if "1" in text:
            return {"param_name": "x", "default": "1"}
        elif "2" in text:
            return {"param_name": "x", "default": "2"}
        return {}
        
    mock_extract.side_effect = extract_side_effect
    
    stats = verify_repo(conn, 1, str(repo_dir), "model", "host")
    
    assert stats["total"] == 4
    assert stats["verified"] == 1
    assert stats["contradicted"] == 1
    assert stats["unverifiable"] == 2
    
    # Verify mock not called for ambiguous or behavior
    assert mock_extract.call_count == 2
    
    cursor = conn.execute("SELECT status FROM verification_results ORDER BY claim_id")
    statuses = [r[0] for r in cursor.fetchall()]
    assert statuses == ["Verified", "Contradicted", "Unverifiable", "Unverifiable"]
    
    # Run twice
    stats2 = verify_repo(conn, 1, str(repo_dir), "model", "host")
    assert stats2 == stats
    cursor = conn.execute("SELECT COUNT(*) FROM verification_results")
    assert cursor.fetchone()[0] == 4  # fully replaced, not duplicated
