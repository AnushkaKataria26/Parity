import sqlite3
import pytest
import os
import json
from parity.reporting.build_report import build_drift_report

@pytest.fixture
def db_conn(tmp_path):
    conn = sqlite3.connect(":memory:")
    # Basic schema
    conn.execute("""CREATE TABLE repos (
        id INTEGER PRIMARY KEY, name TEXT, path TEXT, last_ingested_commit_sha TEXT
    )""")
    conn.execute("""CREATE TABLE code_chunks (
        id INTEGER PRIMARY KEY, repo_id INTEGER, file_path TEXT, symbol_name TEXT, symbol_type TEXT, start_line INTEGER, end_line INTEGER
    )""")
    conn.execute("""CREATE TABLE doc_chunks (
        id INTEGER PRIMARY KEY, repo_id INTEGER, file_path TEXT, heading TEXT, text TEXT
    )""")
    conn.execute("""CREATE TABLE claims (
        id INTEGER PRIMARY KEY, doc_chunk_id INTEGER, claim_text TEXT, claim_type TEXT
    )""")
    conn.execute("""CREATE TABLE verification_results (
        id INTEGER PRIMARY KEY, claim_id INTEGER, matched_code_chunk_id INTEGER, status TEXT, actual_value TEXT, claimed_value TEXT, verified_at TEXT
    )""")
    
    conn.execute("INSERT INTO repos (id, name, path, last_ingested_commit_sha) VALUES (1, 'test', '/test', 'abc12345')")
    
    # Doc files
    conn.execute("INSERT INTO doc_chunks (id, repo_id, file_path, heading, text) VALUES (1, 1, 'doc1.md', 'h1', 'text')")
    conn.execute("INSERT INTO doc_chunks (id, repo_id, file_path, heading, text) VALUES (2, 1, 'doc2.md', 'h2', 'text')")
    
    # Code chunks
    conn.execute("INSERT INTO code_chunks (id, repo_id, file_path, symbol_name, symbol_type, start_line, end_line) VALUES (1, 1, 'code1.py', 'func1', 'function', 10, 20)")
    conn.execute("INSERT INTO code_chunks (id, repo_id, file_path, symbol_name, symbol_type, start_line, end_line) VALUES (2, 1, 'code2.py', 'func2', 'function', 30, 40)")
    
    # Claims
    conn.execute("INSERT INTO claims (id, doc_chunk_id, claim_text, claim_type) VALUES (1, 1, 'claim 1', 'signature')")
    conn.execute("INSERT INTO claims (id, doc_chunk_id, claim_text, claim_type) VALUES (2, 1, 'claim 2', 'behavior')")
    conn.execute("INSERT INTO claims (id, doc_chunk_id, claim_text, claim_type) VALUES (3, 2, 'claim 3', 'return_type')")
    conn.execute("INSERT INTO claims (id, doc_chunk_id, claim_text, claim_type) VALUES (4, 1, 'claim 4', 'signature')")
    
    # Verification
    # Doc 1 has Verified, Unverifiable, Contradicted
    conn.execute("INSERT INTO verification_results (claim_id, matched_code_chunk_id, status, actual_value, claimed_value, verified_at) VALUES (1, 1, 'Verified', 'x', 'x', 'now')")
    conn.execute("INSERT INTO verification_results (claim_id, matched_code_chunk_id, status, actual_value, claimed_value, verified_at) VALUES (2, NULL, 'Unverifiable', NULL, NULL, 'now')")
    conn.execute("INSERT INTO verification_results (claim_id, matched_code_chunk_id, status, actual_value, claimed_value, verified_at) VALUES (4, 2, 'Contradicted', 'y', 'x', 'now')")
    
    # Doc 2 has only Unverifiable
    conn.execute("INSERT INTO verification_results (claim_id, matched_code_chunk_id, status, actual_value, claimed_value, verified_at) VALUES (3, 2, 'Unverifiable', NULL, NULL, 'now')")
    
    return conn

def test_build_report_grouping_and_sorting(db_conn, monkeypatch, tmp_path):
    # Mock companion JSON files
    os.makedirs(tmp_path / "data" / "doc_chunk_bodies" / "1", exist_ok=True)
    with open(tmp_path / "data" / "doc_chunk_bodies" / "1" / "1.json", "w") as f:
        json.dump({"start_line": 5, "end_line": 10}, f)
    with open(tmp_path / "data" / "doc_chunk_bodies" / "1" / "2.json", "w") as f:
        json.dump({"start_line": 15, "end_line": 20}, f)
        
    monkeypatch.chdir(tmp_path)
    
    report = build_drift_report(db_conn, 1)
    
    assert report.repo_name == "test"
    assert report.commit_sha == "abc12345"
    assert report.totals == {"verified": 1, "contradicted": 1, "unverifiable": 2}
    
    # Check file ordering: doc1.md has Contradicted (0), doc2.md has Unverifiable (1)
    file_order = list(report.entries_by_file.keys())
    assert file_order == ["doc1.md", "doc2.md"]
    
    # Check doc1.md sorting: Contradicted first, then Unverifiable, then Verified
    doc1_entries = report.entries_by_file["doc1.md"]
    assert len(doc1_entries) == 3
    assert doc1_entries[0].status == "Contradicted"
    assert doc1_entries[1].status == "Unverifiable"
    assert doc1_entries[2].status == "Verified"
    
    # Check doc_start_line
    assert doc1_entries[0].doc_start_line == 5
    
    # Check code matched fields
    assert doc1_entries[0].matched_symbol == "func2"
    assert doc1_entries[0].matched_file_path == "code2.py"
    assert doc1_entries[0].matched_start_line == 30
    
    assert doc1_entries[1].matched_symbol is None

def test_build_report_alphabetical_tiebreak(db_conn):
    db_conn.execute("INSERT INTO doc_chunks (id, repo_id, file_path, heading, text) VALUES (3, 1, 'a_doc.md', 'h', 'text')")
    db_conn.execute("INSERT INTO claims (id, doc_chunk_id, claim_text, claim_type) VALUES (5, 3, 'claim 5', 'signature')")
    db_conn.execute("INSERT INTO verification_results (claim_id, matched_code_chunk_id, status, actual_value, claimed_value, verified_at) VALUES (5, 1, 'Contradicted', 'a', 'b', 'now')")
    
    report = build_drift_report(db_conn, 1)
    file_order = list(report.entries_by_file.keys())
    # a_doc.md and doc1.md both have Contradicted. a_doc.md comes first alphabetically.
    assert file_order[0] == "a_doc.md"
    assert file_order[1] == "doc1.md"

def test_build_report_no_commit_sha(db_conn):
    db_conn.execute("UPDATE repos SET last_ingested_commit_sha = NULL WHERE id = 1")
    report = build_drift_report(db_conn, 1)
    assert report.commit_sha is None

def test_build_report_missing_companion_json(db_conn, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    report = build_drift_report(db_conn, 1)
    # doc1.md -> doc_start_line should be None
    assert report.entries_by_file["doc1.md"][0].doc_start_line is None

def test_build_report_zero_results(db_conn):
    db_conn.execute("DELETE FROM verification_results")
    report = build_drift_report(db_conn, 1)
    assert report.totals == {"verified": 0, "contradicted": 0, "unverifiable": 0}
    assert report.entries_by_file == {}
