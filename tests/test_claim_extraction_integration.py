import pytest
import sqlite3
import json

from parity.extraction.extractor import extract_claims_for_chunk
from parity.llm.ollama_client import check_ollama_reachable, check_model_available

# This uses whatever the configured model is, assuming the environment has 'llama3' by default or similar.
# Since we are doing a real call, let's just use 'llama3:latest' or check availability.
MODEL = "llama3:latest"
HOST = "http://localhost:11434"

@pytest.mark.slow
def test_integration_extraction_real_ollama():
    if not check_ollama_reachable(HOST):
        pytest.skip(f"Ollama daemon not reachable at {HOST}")
    if not check_model_available(MODEL, HOST):
        pytest.skip(f"Model {MODEL} not available on {HOST}")
        
    chunk = {
        "id": 1,
        "heading": "API Reference / retry",
        "text": "The retry function accepts a max_attempts parameter and returns a bool."
    }
    
    claims, p_fail, l_error = extract_claims_for_chunk(chunk, MODEL, HOST)
    
    assert l_error is False
    assert p_fail is False
    assert len(claims) >= 1
    
    # We should see at least a signature claim
    types = [c.claim_type for c in claims]
    assert "signature" in types
