import os
import json
import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from parity.embedding.model import (
    embed_texts,
    build_code_chunk_embedding_text,
    build_doc_chunk_embedding_text,
    embed_repo
)
import chromadb

@pytest.fixture
def mock_encode():
    # Mock encode returns a 384-dim array of 0.1s for each input text
    def encode_side_effect(texts, **kwargs):
        import numpy as np
        return np.array([[0.1] * 384 for _ in texts])
    
    mock = MagicMock()
    mock.side_effect = encode_side_effect
    return mock

@pytest.fixture
def mock_get_model(mock_encode):
    with patch("parity.embedding.model.get_embedding_model") as mock_get:
        mock_model = MagicMock()
        mock_model.encode = mock_encode
        mock_get.return_value = mock_model
        yield mock_get

def test_embed_texts_normal(mock_get_model):
    texts = ["hello", "world"]
    res = embed_texts(texts)
    assert len(res) == 2
    assert len(res[0]) == 384
    assert res[0][0] == 0.1

def test_embed_texts_with_empty(mock_get_model):
    texts = ["hello", "", "  ", "world"]
    res = embed_texts(texts)
    assert len(res) == 4
    assert res[1] is None
    assert res[2] is None
    assert len(res[0]) == 384
    assert len(res[3]) == 384

def test_embed_texts_all_empty(mock_get_model):
    texts = ["", "   "]
    res = embed_texts(texts)
    assert len(res) == 2
    assert res[0] is None
    assert res[1] is None

def test_build_code_chunk_embedding_text_with_docstring():
    chunk_row = {"symbol_type": "function", "symbol_name": "my_func"}
    body_json = {
        "docstring": "This is a docstring.",
        "source_text": "def my_func(a):\n    '''This is a docstring.'''\n    return a"
    }
    text = build_code_chunk_embedding_text(chunk_row, body_json)
    assert "function my_func" in text
    assert "This is a docstring." in text
    assert "def my_func(a):" in text

def test_build_code_chunk_embedding_text_no_docstring():
    chunk_row = {"symbol_type": "function", "symbol_name": "my_func"}
    body_json = {
        "docstring": None,
        "source_text": "@decorator\ndef my_func(a):\n    pass"
    }
    text = build_code_chunk_embedding_text(chunk_row, body_json)
    assert "function my_func" in text
    assert "def my_func(a):" in text
    assert "pass" not in text

def test_build_code_chunk_embedding_text_missing_json(caplog):
    chunk_row = {"symbol_type": "function", "symbol_name": "my_func"}
    text = build_code_chunk_embedding_text(chunk_row, {})
    # Falls back gracefully to just the symbol type and name
    assert text == "function my_func"

def setup_test_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE repos (id INTEGER PRIMARY KEY, name TEXT, path TEXT UNIQUE, last_ingested_commit_sha TEXT);
        CREATE TABLE code_chunks (
            id INTEGER PRIMARY KEY, repo_id INTEGER, file_path TEXT, 
            symbol_name TEXT, symbol_type TEXT, start_line INTEGER, 
            end_line INTEGER, embedding_id TEXT, ast_hash TEXT
        );
        CREATE TABLE doc_chunks (
            id INTEGER PRIMARY KEY, repo_id INTEGER, file_path TEXT, 
            heading TEXT, text TEXT, embedding_id TEXT
        );
    """)
    return conn

@pytest.fixture
def test_db_and_chroma(tmp_path):
    db_path = tmp_path / "test.db"
    conn = setup_test_db(db_path)
    
    # insert a repo
    conn.execute("INSERT INTO repos (id, name, path) VALUES (1, 'test', 'test_path')")
    
    # insert chunks
    conn.execute("INSERT INTO code_chunks (id, repo_id, file_path, symbol_name, symbol_type, start_line, end_line) VALUES (1, 1, 'a.py', 'func1', 'function', 1, 5)")
    conn.execute("INSERT INTO code_chunks (id, repo_id, file_path, symbol_name, symbol_type, start_line, end_line) VALUES (2, 1, 'b.py', 'func2', 'function', 1, 5)")
    
    conn.execute("INSERT INTO doc_chunks (id, repo_id, file_path, heading, text) VALUES (1, 1, 'a.md', 'Heading 1', 'Some text')")
    
    conn.commit()
    
    # create bodies
    os.makedirs(tmp_path / "data" / "code_chunk_bodies" / "1", exist_ok=True)
    with open(tmp_path / "data" / "code_chunk_bodies" / "1" / "1.json", "w") as f:
        json.dump({"docstring": "doc", "source_text": "def func1():\n pass"}, f)
    with open(tmp_path / "data" / "code_chunk_bodies" / "1" / "2.json", "w") as f:
        json.dump({"docstring": "doc2", "source_text": "def func2():\n pass"}, f)
        
    client = chromadb.EphemeralClient()
    code_col = client.get_or_create_collection("code_chunks")
    doc_col = client.get_or_create_collection("doc_chunks")
    
    # patch os.path.join to point to tmp_path for data dir
    orig_join = os.path.join
    with patch("parity.embedding.model.os.path.join") as mock_join:
        def side_effect(*args):
            if args[0] == "data":
                return str(tmp_path / "/".join(args))
            return orig_join(*args)
        mock_join.side_effect = side_effect
        
        yield conn, client, code_col, doc_col

def test_embed_repo_success(test_db_and_chroma, mock_get_model):
    conn, client, code_col, doc_col = test_db_and_chroma
    
    res = embed_repo(conn, 1, client, code_col, doc_col)
    assert res["code_chunks_embedded"] == 2
    assert res["doc_chunks_embedded"] == 1
    assert res["code_fallback_count"] == 0
    
    # verify db writes
    cursor = conn.cursor()
    cursor.execute("SELECT embedding_id FROM code_chunks WHERE id = 1")
    assert cursor.fetchone()[0] == "code_chunk_1"
    
    # verify chroma counts
    assert code_col.count() == 2
    assert doc_col.count() == 1
    
    # verify metadata
    res_code = code_col.get(where={"repo_id": 1})
    assert len(res_code["metadatas"]) == 2
    assert res_code["metadatas"][0]["repo_id"] == 1
    assert res_code["metadatas"][0]["symbol_name"] in ("func1", "func2")

def test_embed_repo_rerun_clears_old(test_db_and_chroma, mock_get_model):
    conn, client, code_col, doc_col = test_db_and_chroma
    
    # Run once
    embed_repo(conn, 1, client, code_col, doc_col)
    
    # Simulate re-chunking which gives new ids
    conn.execute("DELETE FROM code_chunks")
    conn.execute("INSERT INTO code_chunks (id, repo_id, file_path, symbol_name, symbol_type, start_line, end_line) VALUES (3, 1, 'c.py', 'func3', 'function', 1, 5)")
    conn.commit()
    
    # Run again
    embed_repo(conn, 1, client, code_col, doc_col)
    
    # Check chroma count for repo_id = 1
    res_code = code_col.get(where={"repo_id": 1})
    assert len(res_code["ids"]) == 1
    assert res_code["ids"][0] == "code_chunk_3"
    assert code_col.count() == 1

def test_embed_repo_zero_code_chunks(tmp_path, mock_get_model):
    db_path = tmp_path / "test.db"
    conn = setup_test_db(db_path)
    conn.execute("INSERT INTO repos (id, name, path) VALUES (1, 'test', 'test_path')")
    conn.execute("INSERT INTO doc_chunks (id, repo_id, file_path, heading, text) VALUES (1, 1, 'a.md', 'Heading 1', 'Some text')")
    conn.commit()
    
    client = chromadb.EphemeralClient()
    code_col = client.get_or_create_collection("code_chunks")
    doc_col = client.get_or_create_collection("doc_chunks")
    
    res = embed_repo(conn, 1, client, code_col, doc_col)
    assert res["code_chunks_embedded"] == 0
    assert res["doc_chunks_embedded"] == 1
