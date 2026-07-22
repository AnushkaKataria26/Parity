import pytest
import sqlite3
import os

from parity.retrieval.retriever import retrieve_for_claim
from parity.embedding.model import embed_texts
from parity.vectorstore.chroma_client import get_chroma_client, get_or_create_collections

@pytest.mark.slow
def test_retrieval_integration(tmp_path):
    # Setup SQLite
    db_path = tmp_path / "parity.db"
    conn = sqlite3.connect(db_path)
    
    from parity.db.schema import SCHEMA_STATEMENTS
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
        
    conn.execute("INSERT INTO repos (name, path) VALUES ('test', '/test')")
    repo_id = 1
    
    # Hand-crafted functions
    functions = [
        {"symbol_name": "calculate_tax", "docstring": "Calculates the tax amount.", "sig": "def calculate_tax(amount):"},
        {"symbol_name": "process_payment", "docstring": "Processes the user's payment securely.", "sig": "def process_payment(user_id, amount):"},
        {"symbol_name": "send_email", "docstring": "Sends an email to the user.", "sig": "def send_email(to, subject, body):"},
        {"symbol_name": "connect_db", "docstring": "Establishes a database connection.", "sig": "def connect_db(uri):"},
        {"symbol_name": "retry_operation", "docstring": "Retries a failing operation with exponential backoff.", "sig": "def retry_operation(func, max_retries):"}
    ]
    
    # Insert code chunks
    for i, fn in enumerate(functions, start=1):
        conn.execute("""
            INSERT INTO code_chunks (repo_id, file_path, symbol_name, symbol_type, start_line, end_line)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (repo_id, "main.py", fn["symbol_name"], "function", 1, 5))
        
    conn.commit()
    
    # Setup Chroma
    chroma_dir = str(tmp_path / "chroma")
    client = get_chroma_client(chroma_dir)
    code_col, _ = get_or_create_collections(client)
    
    # Embed code chunks
    texts_to_embed = []
    for fn in functions:
        text = f"function {fn['symbol_name']}\n{fn['docstring']}\n{fn['sig']}"
        texts_to_embed.append(text)
        
    embeddings = embed_texts(texts_to_embed, model_name="BAAI/bge-small-en-v1.5")
    
    chroma_ids = []
    metas = []
    
    for i, emb in enumerate(embeddings, start=1):
        chroma_ids.append(f"code_chunk_{i}")
        metas.append({"repo_id": repo_id, "symbol_name": functions[i-1]["symbol_name"]})
        
    code_col.add(ids=chroma_ids, embeddings=embeddings, metadatas=metas)
    
    # Test 1: claim with clear exact match
    claim_row = {
        "id": 1,
        "claim_text": "The system processes user payments securely.",
        "referenced_symbol_guess": "process_payment"
    }
    
    res1 = retrieve_for_claim(claim_row, {"repo_id": repo_id}, code_col, "BAAI/bge-small-en-v1.5")
    assert res1.match_status == "matched"
    assert res1.matched_code_chunk_id == 2 # process_payment is ID 2
    
    # Test 2: claim with no symbol guess but clear semantics
    claim_row_2 = {
        "id": 2,
        "claim_text": "If an operation fails, it is retried with exponential backoff.",
        "referenced_symbol_guess": None
    }
    
    res2 = retrieve_for_claim(claim_row_2, {"repo_id": repo_id}, code_col, "BAAI/bge-small-en-v1.5")
    # Semantic match should find retry_operation which is ID 5
    assert res2.match_status == "matched"
    assert res2.matched_code_chunk_id == 5
