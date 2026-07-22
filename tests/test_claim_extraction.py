import json
import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from parity.extraction.extractor import parse_and_validate_claims, extract_claims_for_chunk, store_claims
from parity.extraction.prompts import ExtractedClaim
from parity.llm.ollama_client import LLMCallError

def test_parse_valid_json():
    raw = json.dumps([
        {"claim_text": "hello", "claim_type": "signature", "referenced_symbol_guess": "foo"}
    ])
    claims, errors = parse_and_validate_claims(raw)
    assert len(errors) == 0
    assert len(claims) == 1
    assert claims[0].claim_text == "hello"
    assert claims[0].claim_type == "signature"
    assert claims[0].referenced_symbol_guess == "foo"

def test_parse_code_fences():
    raw = "```json\n" + json.dumps([
        {"claim_text": "hello", "claim_type": "signature", "referenced_symbol_guess": "foo"}
    ]) + "\n```"
    claims, errors = parse_and_validate_claims(raw)
    assert len(errors) == 0
    assert len(claims) == 1

def test_parse_trailing_prose():
    raw = json.dumps([
        {"claim_text": "hello", "claim_type": "signature", "referenced_symbol_guess": "foo"}
    ]) + "\nHere is the json you requested."
    claims, errors = parse_and_validate_claims(raw)
    assert len(errors) == 0
    assert len(claims) == 1

def test_parse_garbage():
    raw = "This is not json."
    claims, errors = parse_and_validate_claims(raw)
    assert len(claims) == 0
    assert len(errors) == 1
    assert "failed to parse JSON" in errors[0]

def test_parse_missing_claim_text():
    raw = json.dumps([
        {"claim_type": "signature", "referenced_symbol_guess": "foo"},
        {"claim_text": "hello", "claim_type": "signature", "referenced_symbol_guess": "foo"}
    ])
    claims, errors = parse_and_validate_claims(raw)
    assert len(claims) == 1
    assert len(errors) == 1
    assert "missing required keys" in errors[0]

def test_parse_invalid_type():
    raw = json.dumps([
        {"claim_text": "hello", "claim_type": "nonsense_type", "referenced_symbol_guess": "foo"}
    ])
    claims, errors = parse_and_validate_claims(raw)
    assert len(errors) == 0
    assert len(claims) == 1
    assert claims[0].claim_type == "behavior"

def test_parse_referenced_number():
    raw = json.dumps([
        {"claim_text": "hello", "claim_type": "signature", "referenced_symbol_guess": 42}
    ])
    claims, errors = parse_and_validate_claims(raw)
    assert len(errors) == 0
    assert len(claims) == 1
    assert claims[0].referenced_symbol_guess == "42"

def test_parse_wrong_case_type():
    raw = json.dumps([
        {"claim_text": "hello", "claim_type": "SIGNATURE", "referenced_symbol_guess": "foo"}
    ])
    claims, errors = parse_and_validate_claims(raw)
    assert len(errors) == 0
    assert claims[0].claim_type == "signature"

def test_parse_empty_array():
    raw = "[]"
    claims, errors = parse_and_validate_claims(raw)
    assert len(claims) == 0
    assert len(errors) == 0

@patch("parity.extraction.extractor.extract_claims_raw")
def test_extract_for_chunk_retry_success(mock_extract):
    # First call returns garbage, second returns valid JSON
    mock_extract.side_effect = [
        "garbage",
        json.dumps([{"claim_text": "hello", "claim_type": "signature", "referenced_symbol_guess": "foo"}])
    ]
    chunk = {"id": 1, "heading": "h", "text": "t"}
    claims, p_fail, l_error = extract_claims_for_chunk(chunk, "model", "host")
    assert len(claims) == 1
    assert p_fail is False
    assert l_error is False
    assert mock_extract.call_count == 2
    assert "Your previous response was not valid JSON" in mock_extract.call_args_list[1][1].get('retry_message', '')

@patch("parity.extraction.extractor.extract_claims_raw")
def test_extract_for_chunk_retry_fail(mock_extract):
    mock_extract.side_effect = ["garbage", "garbage"]
    chunk = {"id": 1, "heading": "h", "text": "t"}
    claims, p_fail, l_error = extract_claims_for_chunk(chunk, "model", "host")
    assert len(claims) == 0
    assert p_fail is True
    assert l_error is False
    assert mock_extract.call_count == 2

@patch("parity.extraction.extractor.extract_claims_raw")
def test_extract_for_chunk_llm_error(mock_extract):
    mock_extract.side_effect = LLMCallError("connection error")
    chunk = {"id": 1, "heading": "h", "text": "t"}
    claims, p_fail, l_error = extract_claims_for_chunk(chunk, "model", "host")
    assert len(claims) == 0
    assert p_fail is False
    assert l_error is True
    assert mock_extract.call_count == 1

@patch("parity.extraction.extractor.extract_claims_raw")
def test_extract_for_chunk_legit_empty(mock_extract):
    mock_extract.return_value = "[]"
    chunk = {"id": 1, "heading": "h", "text": "t"}
    claims, p_fail, l_error = extract_claims_for_chunk(chunk, "model", "host")
    assert len(claims) == 0
    assert p_fail is False
    assert l_error is False
    assert mock_extract.call_count == 1

def test_store_claims():
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_chunk_id INTEGER,
        claim_text TEXT,
        claim_type TEXT,
        referenced_symbol_guess TEXT
    )""")
    claims1 = [ExtractedClaim("hello", "signature", "foo")]
    store_claims(conn, 1, claims1)
    
    claims2 = [ExtractedClaim("world", "behavior", "bar"), ExtractedClaim("test", "env_var", None)]
    store_claims(conn, 1, claims2)
    
    cursor = conn.execute("SELECT claim_text FROM claims WHERE doc_chunk_id = 1")
    rows = cursor.fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "world"
    assert rows[1][0] == "test"
