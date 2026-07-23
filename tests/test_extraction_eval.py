import os
import sqlite3
import pytest
import csv
from parity.evaluation.extraction_eval import export_claims_for_labeling, score_extraction
from parity.db.migrate import apply_schema

@pytest.fixture
def mem_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    yield conn
    conn.close()

def test_export_claims_for_labeling_basic(mem_db, tmp_path):
    conn = mem_db
    conn.execute("INSERT INTO repos (id, name, path, last_ingested_commit_sha) VALUES (1, 'test', '/tmp/test', 'sha')")
    conn.execute("INSERT INTO doc_chunks (id, repo_id, file_path, heading, text) VALUES (1, 1, 'test.md', 'h1', 't1')")
    conn.execute("INSERT INTO doc_chunks (id, repo_id, file_path, heading, text) VALUES (2, 1, 'test.md', 'h2', 't2')")
    
    out_file = tmp_path / "out.csv"
    count = export_claims_for_labeling(conn, 1, str(out_file))
    
    assert count == 2
    assert out_file.exists()
    
    with open(out_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 3 # header + 2
        assert rows[0] == ["doc_chunk_id", "heading", "text", "expected_claim_count", "expected_claim_types", "notes"]
        assert rows[1][0] == "1"
        assert rows[2][0] == "2"

def test_export_claims_for_labeling_empty(mem_db, tmp_path):
    conn = mem_db
    out_file = tmp_path / "out.csv"
    count = export_claims_for_labeling(conn, 1, str(out_file))
    
    assert count == 0
    with open(out_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0] == ["doc_chunk_id", "heading", "text", "expected_claim_count", "expected_claim_types", "notes"]

def test_score_extraction(mem_db, tmp_path):
    conn = mem_db
    conn.execute("INSERT INTO repos (id, name, path, last_ingested_commit_sha) VALUES (1, 'test', '/tmp/test', 'sha')")
    conn.execute("INSERT INTO doc_chunks (id, repo_id, file_path, text) VALUES (1, 1, 'f1', 't1')")
    conn.execute("INSERT INTO doc_chunks (id, repo_id, file_path, text) VALUES (2, 1, 'f2', 't2')")
    
    # doc chunk 1 has 2 claims in DB
    conn.execute("INSERT INTO claims (id, doc_chunk_id, claim_text, claim_type) VALUES (1, 1, 'c1', 'signature')")
    conn.execute("INSERT INTO claims (id, doc_chunk_id, claim_text, claim_type) VALUES (2, 1, 'c2', 'default_value')")
    
    # doc chunk 2 has 0 claims in DB
    
    label_file = tmp_path / "labels.csv"
    with open(label_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_chunk_id", "expected_claim_count"])
        writer.writerow(["1", "2"]) # match exact
        writer.writerow(["2", "0"]) # match exact
        writer.writerow(["3", "1"]) # missing chunk, will get 0 actual -> within_one=1, nonzero_correct=0
        writer.writerow(["4", "bad"]) # malformed
        writer.writerow(["5", ""]) # skipped
        writer.writerow(["missing_cols"]) # missing cols
        
    res = score_extraction(conn, 1, str(label_file))
    
    assert res["labeled_count"] == 3
    assert res["zero_empty_total"] == 1
    assert res["zero_empty_correct"] == 1
    assert res["nonzero_total"] == 2
    assert res["nonzero_correct"] == 1
    assert res["exact_match"] == 2
    assert res["within_one"] == 3 # actual=0 vs exp=1 is within 1
