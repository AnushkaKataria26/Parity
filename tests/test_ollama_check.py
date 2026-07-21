import requests
import pytest
from unittest.mock import patch, Mock
from parity.llm.ollama_client import check_ollama_reachable, check_model_available

def test_check_ollama_reachable_unreachable():
    # Should not raise an exception
    assert check_ollama_reachable("http://localhost:1", timeout=0.1) == False

@patch("requests.get")
def test_check_model_available_exact(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "llama3:8b"}]}
    mock_get.return_value = mock_response
    
    assert check_model_available("llama3:8b", "http://fake") == True

@patch("requests.get")
def test_check_model_available_prefix(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "llama3:8b-instruct-q4_0"}]}
    mock_get.return_value = mock_response
    
    assert check_model_available("llama3:8b", "http://fake") == True

@patch("requests.get")
def test_check_model_available_mismatch(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "llama3:8b"}]}
    mock_get.return_value = mock_response
    
    assert check_model_available("mistral:7b", "http://fake") == False

@patch("requests.get")
def test_check_ollama_reachable_connection_error(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")
    assert check_ollama_reachable("http://fake") == False
