import pytest
from unittest.mock import patch
from parity.verification.claim_value_extraction import extract_claimed_value

def test_extract_claimed_value_signature():
    with patch('parity.verification.claim_value_extraction.call_ollama_json', return_value='{"param_names": ["host", "port"], "param_count": 2}') as mock_call:
        res = extract_claimed_value("takes host and port", "signature", "test-model", "http://localhost:11434")
        assert res == {"param_names": ["host", "port"], "param_count": 2}
        mock_call.assert_called_once()

def test_extract_claimed_value_default_value():
    with patch('parity.verification.claim_value_extraction.call_ollama_json', return_value='{"param_name": "timeout", "default": "None"}'):
        res = extract_claimed_value("timeout is None", "default_value", "test-model", "http://localhost:11434")
        assert res == {"param_name": "timeout", "default": "None"}

def test_extract_claimed_value_env_var():
    with patch('parity.verification.claim_value_extraction.call_ollama_json', return_value='{"var_name": "DEBUG"}'):
        res = extract_claimed_value("reads DEBUG env var", "env_var", "test-model", "http://localhost:11434")
        assert res == {"var_name": "DEBUG"}

def test_extract_claimed_value_return_type():
    with patch('parity.verification.claim_value_extraction.call_ollama_json', return_value='{"return_type": "list"}'):
        res = extract_claimed_value("returns a list", "return_type", "test-model", "http://localhost:11434")
        assert res == {"return_type": "list"}

def test_extract_claimed_value_retry_success():
    responses = ["not json", '{"return_type": "dict"}']
    with patch('parity.verification.claim_value_extraction.call_ollama_json', side_effect=responses) as mock_call:
        res = extract_claimed_value("returns a dict", "return_type", "test-model", "http://localhost:11434")
        assert res == {"return_type": "dict"}
        assert mock_call.call_count == 2

def test_extract_claimed_value_retry_fail():
    responses = ["not json", "still not json"]
    with patch('parity.verification.claim_value_extraction.call_ollama_json', side_effect=responses) as mock_call:
        res = extract_claimed_value("returns a dict", "return_type", "test-model", "http://localhost:11434")
        assert res == {}
        assert mock_call.call_count == 2
