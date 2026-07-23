import pytest
import sqlite3
import os
import json
import sys
from parity.cli.main import cmd_report

class DummyArgs:
    def __init__(self, repo_path, config=None, verbose=False, json_out=None, text_out=None):
        self.repo_path = repo_path
        self.config = config
        self.verbose = verbose
        self.json_out = json_out
        self.text_out = text_out
        
@pytest.fixture
def test_env(tmp_path, monkeypatch):
    # Setup repo structure
    repo_path = tmp_path / "myrepo"
    repo_path.mkdir()
    
    # Setup db
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    
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
    
    conn.execute("INSERT INTO repos (id, name, path, last_ingested_commit_sha) VALUES (1, 'myrepo', ?, 'abc12345')", (str(repo_path),))
    conn.commit()
    
    # Setup config
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write(f"db_path: {db_path}\n")
        
    return repo_path, config_path, conn

def test_cmd_report_with_data(test_env, tmp_path, capsys):
    repo_path, config_path, conn = test_env
    
    conn.execute("INSERT INTO doc_chunks (id, repo_id, file_path, text) VALUES (1, 1, 'doc.md', 'txt')")
    conn.execute("INSERT INTO claims (id, doc_chunk_id, claim_text, claim_type) VALUES (1, 1, 'claim', 'signature')")
    conn.execute("INSERT INTO verification_results (claim_id, status, verified_at) VALUES (1, 'Contradicted', 'now')")
    conn.commit()
    
    json_out = tmp_path / "out" / "report.json"
    text_out = tmp_path / "out" / "report.txt"
    
    args = DummyArgs(str(repo_path), config=str(config_path), json_out=str(json_out), text_out=str(text_out))
    
    with pytest.raises(SystemExit) as excinfo:
        cmd_report(args)
        
    assert excinfo.value.code == 0
    
    out, err = capsys.readouterr()
    assert "Parity Drift Report — myrepo" in out
    
    # check files
    assert json_out.exists()
    assert text_out.exists()
    
    with open(json_out, "r") as f:
        data = json.load(f)
        assert data["repo_name"] == "myrepo"
        
    with open(text_out, "r") as f:
        txt = f.read()
        assert "Parity Drift Report" in txt

def test_cmd_report_zero_results(test_env, capsys):
    repo_path, config_path, conn = test_env
    # No verification results seeded
    
    args = DummyArgs(str(repo_path), config=str(config_path))
    
    with pytest.raises(SystemExit) as excinfo:
        cmd_report(args)
        
    assert excinfo.value.code == 0
    
    out, err = capsys.readouterr()
    assert "Warning: no verification results found" in err
    assert "No drift detected" in out

def test_cmd_report_deterministic_runs(test_env, tmp_path):
    repo_path, config_path, conn = test_env
    
    conn.execute("INSERT INTO doc_chunks (id, repo_id, file_path, text) VALUES (1, 1, 'doc.md', 'txt')")
    conn.execute("INSERT INTO claims (id, doc_chunk_id, claim_text, claim_type) VALUES (1, 1, 'claim', 'signature')")
    conn.execute("INSERT INTO verification_results (claim_id, status, verified_at) VALUES (1, 'Contradicted', 'now')")
    conn.commit()
    
    json_out = tmp_path / "report.json"
    args = DummyArgs(str(repo_path), config=str(config_path), json_out=str(json_out))
    
    with pytest.raises(SystemExit):
        cmd_report(args)
        
    with open(json_out, "r") as f:
        run1 = json.load(f)
        
    # run again
    with pytest.raises(SystemExit):
        cmd_report(args)
        
    with open(json_out, "r") as f:
        run2 = json.load(f)
        
    # delete generated_at for comparison
    del run1["generated_at"]
    del run2["generated_at"]
    
    assert run1 == run2
