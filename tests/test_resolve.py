import json
import os
import sys
import pytest
from parity.verification.resolve import (
    module_path_from_file,
    resolve_symbol_dynamic,
    resolve_symbol_static
)

def test_module_path_from_file():
    assert module_path_from_file("pkg/sub/mod.py", "/repo") == "pkg.sub.mod"
    assert module_path_from_file("pkg/sub/__init__.py", "/repo") == "pkg.sub"
    assert module_path_from_file("pkg/sub/mod.txt", "/repo") is None

@pytest.fixture
def repo_fixture(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    
    # Create a simple fixture module
    mod_content = """
import functools

def plain_func(a, b=2):
    pass

def my_decorator(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

@my_decorator
def wrapped_func(c, d=4):
    pass

def bad_decorator(f):
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

@bad_decorator
def bad_wrapped_func(e):
    pass

class MyClass:
    def __init__(self, x):
        pass
        
    def method(self, y):
        pass

def outer():
    def inner(z):
        pass
    return inner
    
SOME_CONST = 42
def func_with_const(timeout=SOME_CONST):
    pass

def complex_func(pos_only, /, pos_or_kw, *args, kw_only=5, **kwargs):
    pass
"""
    mod_file = repo_dir / "my_module.py"
    mod_file.write_text(mod_content, encoding="utf-8")
    
    bad_mod_content = """
raise ImportError("simulated failure")
"""
    bad_mod_file = repo_dir / "bad_module.py"
    bad_mod_file.write_text(bad_mod_content, encoding="utf-8")
    
    # Add repo_dir to sys.path so we can import from it dynamically
    sys.path.insert(0, str(repo_dir))
    yield str(repo_dir)
    sys.path.pop(0)

def test_resolve_symbol_dynamic(repo_fixture):
    cache = {}
    
    # Plain func
    res = resolve_symbol_dynamic("my_module", "plain_func", repo_fixture, cache)
    assert res is not None
    assert res.resolution_method == "dynamic"
    assert len(res.parameters) == 2
    assert res.parameters[0]["name"] == "a"
    assert res.parameters[1]["name"] == "b"
    assert res.parameters[1]["has_default"] is True
    assert res.parameters[1]["default_repr"] == "2"

    # functools.wraps
    res = resolve_symbol_dynamic("my_module", "wrapped_func", repo_fixture, cache)
    assert res is not None
    assert len(res.parameters) == 2
    assert res.parameters[0]["name"] == "c"
    
    # without functools.wraps
    res = resolve_symbol_dynamic("my_module", "bad_wrapped_func", repo_fixture, cache)
    assert res is not None
    assert len(res.parameters) == 2
    assert res.parameters[0]["name"] == "args"
    assert res.parameters[1]["name"] == "kwargs"
    
    # Class __init__
    res = resolve_symbol_dynamic("my_module", "MyClass", repo_fixture, cache)
    assert res is not None
    assert len(res.parameters) == 1
    assert res.parameters[0]["name"] == "x"  # self stripped
    
    # Nested function
    res = resolve_symbol_dynamic("my_module", "outer.inner", repo_fixture, cache)
    assert res is None
    
    # ImportError
    res = resolve_symbol_dynamic("bad_module", "anything", repo_fixture, cache)
    assert res is None
    assert cache["bad_module"] is None
    
def test_resolve_symbol_static(tmp_path):
    # Prepare dummy DB bodies
    bodies_dir = tmp_path / "data" / "code_chunk_bodies" / "1"
    bodies_dir.mkdir(parents=True)
    
    complex_text = "def complex_func(pos_only, /, pos_or_kw, *args, kw_only=5, **kwargs):\n    pass"
    (bodies_dir / "101.json").write_text(json.dumps({"text": complex_text}), encoding="utf-8")
    
    const_text = "def func_with_const(timeout=SOME_CONST):\n    pass"
    (bodies_dir / "102.json").write_text(json.dumps({"text": const_text}), encoding="utf-8")
    
    class_text = "class MyClass:\n    def __init__(self, x):\n        pass"
    (bodies_dir / "103.json").write_text(json.dumps({"text": class_text}), encoding="utf-8")
    
    # Override os.getcwd for the test, or just monkeypatch open?
    # Since resolve_symbol_static hardcodes "data/code_chunk_bodies/...", 
    # we should chdir to tmp_path
    old_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        # Complex signature
        res = resolve_symbol_static(101, 1, "complex_func", "function")
        assert res is not None
        assert res.resolution_method == "static"
        kinds = [p["kind"] for p in res.parameters]
        assert "POSITIONAL_ONLY" in kinds
        assert "POSITIONAL_OR_KEYWORD" in kinds
        assert "VAR_POSITIONAL" in kinds
        assert "KEYWORD_ONLY" in kinds
        assert "VAR_KEYWORD" in kinds
        
        # Non-literal default
        res2 = resolve_symbol_static(102, 1, "func_with_const", "function")
        assert res2 is not None
        p = res2.parameters[0]
        assert p["name"] == "timeout"
        assert p["has_default"] is True
        assert p["is_literal"] is False
        assert p["default_repr"] == "SOME_CONST"
        
        # Self stripped
        res3 = resolve_symbol_static(103, 1, "MyClass", "class")
        assert res3 is not None
        assert len(res3.parameters) == 1
        assert res3.parameters[0]["name"] == "x"
    finally:
        os.chdir(old_cwd)
