import os
import sys
import pytest
from parity.chunking.ast_chunker import extract_chunks_from_file
from parity.verification.resolve import _resolve_symbol_static_from_source, resolve_symbol_dynamic
from parity.verification.verify import verify_signature_claim

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "ast_edge_cases")

@pytest.fixture
def add_fixtures_to_path():
    sys.path.insert(0, FIXTURES_DIR)
    yield
    sys.path.remove(FIXTURES_DIR)

def assert_param(param_dict, name, kind, has_default, default_repr=None, is_literal=None):
    assert param_dict["name"] == name
    assert param_dict["kind"] == kind
    assert param_dict["has_default"] == has_default
    if has_default:
        assert param_dict["default_repr"] == default_repr
    if is_literal is not None:
        assert param_dict["is_literal"] == is_literal

def test_decorator_stacking():
    file_path = os.path.join(FIXTURES_DIR, "fixture_decorators.py")
    chunks, _ = extract_chunks_from_file(file_path, FIXTURES_DIR)
    chunk_map = {c.symbol_name: c for c in chunks}
    
    call_chunk = chunk_map["Service.call"]
    assert call_chunk.start_line == 15
    assert call_chunk.end_line == 19
    assert call_chunk.symbol_type == "method"
    
    assert "@logged" in call_chunk.source_text
    assert "@retry(3)" in call_chunk.source_text
    assert "@staticmethod" in call_chunk.source_text
    
    assert "retry.decorator" in chunk_map
    assert "logged.wrapper" in chunk_map


def test_parameter_kinds(add_fixtures_to_path):
    file_path = os.path.join(FIXTURES_DIR, "fixture_param_kinds.py")
    chunks, _ = extract_chunks_from_file(file_path, FIXTURES_DIR)
    chunk_map = {c.symbol_name: c for c in chunks}
    
    sym = _resolve_symbol_static_from_source(chunk_map["full_signature"].source_text, "function")
    params = sym.parameters
    assert_param(params[0], "a", "POSITIONAL_ONLY", False)
    assert_param(params[1], "b", "POSITIONAL_ONLY", False)
    assert_param(params[2], "c", "POSITIONAL_OR_KEYWORD", False)
    assert_param(params[3], "d", "POSITIONAL_OR_KEYWORD", True, "4")
    assert_param(params[4], "args", "VAR_POSITIONAL", False)
    assert_param(params[5], "e", "KEYWORD_ONLY", False)
    assert_param(params[6], "f", "KEYWORD_ONLY", True, "6")
    assert_param(params[7], "kwargs", "VAR_KEYWORD", False)
    
    sym = _resolve_symbol_static_from_source(chunk_map["only_pos_only"].source_text, "function")
    params = sym.parameters
    assert len(params) == 2
    assert_param(params[0], "a", "POSITIONAL_ONLY", False)
    assert_param(params[1], "b", "POSITIONAL_ONLY", False)
    
    sym = _resolve_symbol_static_from_source(chunk_map["only_kw_only"].source_text, "function")
    params = sym.parameters
    assert len(params) == 2
    assert_param(params[0], "a", "KEYWORD_ONLY", False)
    assert_param(params[1], "b", "KEYWORD_ONLY", True, "2")
    
    sym = _resolve_symbol_static_from_source(chunk_map["star_args_only"].source_text, "function")
    params = sym.parameters
    assert len(params) == 1
    assert_param(params[0], "args", "VAR_POSITIONAL", False)
    
    sym = _resolve_symbol_static_from_source(chunk_map["star_kwargs_only"].source_text, "function")
    params = sym.parameters
    assert len(params) == 1
    assert_param(params[0], "kwargs", "VAR_KEYWORD", False)
    
    sym = _resolve_symbol_static_from_source(chunk_map["no_params"].source_text, "function")
    assert sym.parameters == []
    
    cache = {}
    dyn_full = resolve_symbol_dynamic("fixture_param_kinds", "full_signature", FIXTURES_DIR, cache)
    assert dyn_full.parameters == _resolve_symbol_static_from_source(chunk_map["full_signature"].source_text, "function").parameters

    for fn in ["only_pos_only", "only_kw_only", "star_args_only", "star_kwargs_only", "no_params"]:
        dyn = resolve_symbol_dynamic("fixture_param_kinds", fn, FIXTURES_DIR, cache)
        stat = _resolve_symbol_static_from_source(chunk_map[fn].source_text, "function")
        assert dyn.parameters == stat.parameters


def test_kw_defaults():
    file_path = os.path.join(FIXTURES_DIR, "fixture_kwdefaults.py")
    chunks, _ = extract_chunks_from_file(file_path, FIXTURES_DIR)
    chunk_map = {c.symbol_name: c for c in chunks}
    
    sym = _resolve_symbol_static_from_source(chunk_map["mixed_kw_defaults"].source_text, "function")
    params = sym.parameters
    assert len(params) == 5
    assert_param(params[0], "a", "KEYWORD_ONLY", False)
    assert_param(params[1], "b", "KEYWORD_ONLY", True, "2")
    assert_param(params[2], "c", "KEYWORD_ONLY", False)
    assert_param(params[3], "d", "KEYWORD_ONLY", True, "4")
    assert_param(params[4], "e", "KEYWORD_ONLY", False)


def test_nonliteral_defaults():
    file_path = os.path.join(FIXTURES_DIR, "fixture_nonliteral_defaults.py")
    chunks, _ = extract_chunks_from_file(file_path, FIXTURES_DIR)
    chunk_map = {c.symbol_name: c for c in chunks}
    
    sym1 = _resolve_symbol_static_from_source(chunk_map["f1"].source_text, "function")
    assert_param(sym1.parameters[0], "x", "POSITIONAL_OR_KEYWORD", True, "TIMEOUT", is_literal=False)
    
    sym2 = _resolve_symbol_static_from_source(chunk_map["f2"].source_text, "function")
    assert_param(sym2.parameters[0], "x", "POSITIONAL_OR_KEYWORD", True, "logging.INFO", is_literal=False)
    
    sym3 = _resolve_symbol_static_from_source(chunk_map["f3"].source_text, "function")
    assert_param(sym3.parameters[0], "x", "POSITIONAL_OR_KEYWORD", True, "os.environ.get('X')", is_literal=False)
    
    sym4 = _resolve_symbol_static_from_source(chunk_map["f4"].source_text, "function")
    assert_param(sym4.parameters[0], "x", "POSITIONAL_OR_KEYWORD", True, "[1, 2, 3]", is_literal=True)
    
    sym5 = _resolve_symbol_static_from_source(chunk_map["f5"].source_text, "function")
    assert_param(sym5.parameters[0], "x", "POSITIONAL_OR_KEYWORD", True, "(1, 2)", is_literal=True)
    
    sym6 = _resolve_symbol_static_from_source(chunk_map["f6"].source_text, "function")
    assert_param(sym6.parameters[0], "x", "POSITIONAL_OR_KEYWORD", True, "{1: 2}", is_literal=True)
    
    sym7 = _resolve_symbol_static_from_source(chunk_map["f7"].source_text, "function")
    assert_param(sym7.parameters[0], "x", "POSITIONAL_OR_KEYWORD", True, "-5", is_literal=True)
    
    sym8 = _resolve_symbol_static_from_source(chunk_map["f8"].source_text, "function")
    assert_param(sym8.parameters[0], "x", "POSITIONAL_OR_KEYWORD", True, "1 + 2", is_literal=False)


def test_overloads():
    file_path = os.path.join(FIXTURES_DIR, "fixture_overloads.py")
    chunks, _ = extract_chunks_from_file(file_path, FIXTURES_DIR)
    chunk_names = [c.symbol_name for c in chunks]
    
    assert "process" in chunk_names
    assert "process#2" in chunk_names
    assert "process#3" in chunk_names
    
    process_indices = [i for i, name in enumerate(chunk_names) if name.startswith("process")]
    assert chunk_names[process_indices[0]] == "process"
    assert chunk_names[process_indices[1]] == "process#2"
    assert chunk_names[process_indices[2]] == "process#3"
    
    assert "conditional_fn" in chunk_names
    assert "conditional_fn#2" in chunk_names


def test_deep_nesting():
    file_path = os.path.join(FIXTURES_DIR, "fixture_deep_nesting.py")
    chunks, _ = extract_chunks_from_file(file_path, FIXTURES_DIR)
    chunk_map = {c.symbol_name: c for c in chunks}
    
    assert len(chunks) == 6
    assert "level1" in chunk_map
    assert "level1.level2" in chunk_map
    assert "level1.level2.level3" in chunk_map
    assert "level1.level2.level3.Level4" in chunk_map
    assert "level1.level2.level3.Level4.level5" in chunk_map
    assert "level1.level2.level3.Level4.level5.level6" in chunk_map


def test_wraps_behavior(add_fixtures_to_path):
    cache = {}
    
    doc_dyn = resolve_symbol_dynamic("fixture_wraps_behavior", "documented_func", FIXTURES_DIR, cache)
    params = doc_dyn.parameters
    assert len(params) == 2
    assert_param(params[0], "a", "POSITIONAL_OR_KEYWORD", False)
    assert_param(params[1], "b", "POSITIONAL_OR_KEYWORD", True, "5")
    
    undoc_dyn = resolve_symbol_dynamic("fixture_wraps_behavior", "undocumented_wrapper_func", FIXTURES_DIR, cache)
    params = undoc_dyn.parameters
    assert len(params) == 2
    assert_param(params[0], "args", "VAR_POSITIONAL", False)
    assert_param(params[1], "kwargs", "VAR_KEYWORD", False)
    
    claim = {"param_names": ["a", "b"]}
    # verify_signature_claim signature: (claimed: dict, resolved: ResolvedSymbol) -> VerificationOutcome
    result = verify_signature_claim(claim, undoc_dyn)
    assert result.status == "Unverifiable"

