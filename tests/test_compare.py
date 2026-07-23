import pytest
from parity.verification.compare import normalize_value, compare_default_values

def test_normalize_value_numeric():
    assert normalize_value("30") == "30.0"
    assert normalize_value("30.0") == "30.0"
    
def test_normalize_value_none():
    assert normalize_value("None") == "none"
    assert normalize_value("none") == "none"
    assert normalize_value("null") == "none"
    assert normalize_value("NIL") == "none"

def test_normalize_value_quotes():
    assert normalize_value('"utf-8"') == "utf-8"
    assert normalize_value("'utf-8'") == "utf-8"
    assert normalize_value("utf-8") == "utf-8"
    assert normalize_value(' "utf-8" ') == "utf-8"

def test_normalize_value_booleans():
    assert normalize_value("True") == "true"
    assert normalize_value("false") == "false"
    assert normalize_value("1") == "true"
    assert normalize_value("0") == "false"

def test_compare_default_values():
    assert compare_default_values("30", "30.0") is True
    assert compare_default_values("None", "null") is True
    assert compare_default_values("'utf-8'", '"utf-8"') is True
    assert compare_default_values("True", "true") is True
    
    assert compare_default_values("30", "31") is False
    assert compare_default_values("None", "false") is False
    assert compare_default_values("utf-8", "ascii") is False
