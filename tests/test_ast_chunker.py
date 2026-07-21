import os
import sqlite3
import json
import pytest
from parity.chunking.ast_chunker import (
    CodeChunk, 
    discover_python_files, 
    extract_chunks_from_file
)
from parity.db.chunk_ops import store_chunks

def test_extract_simple_top_level(tmp_path):
    code = """
def my_func():
    pass

class MyClass:
    def method_one(self):
        pass
    def method_two(self):
        pass
"""
    f = tmp_path / "test.py"
    f.write_text(code)
    
    chunks, skipped = extract_chunks_from_file(str(f), str(tmp_path))
    assert not skipped
    assert len(chunks) == 4
    
    func_chunk = next(c for c in chunks if c.symbol_name == "my_func")
    assert func_chunk.symbol_type == "function"
    assert func_chunk.start_line == 2
    assert func_chunk.end_line == 3
    
    class_chunk = next(c for c in chunks if c.symbol_name == "MyClass")
    assert class_chunk.symbol_type == "class"
    assert class_chunk.start_line == 5
    assert class_chunk.end_line == 9
    
    m1_chunk = next(c for c in chunks if c.symbol_name == "MyClass.method_one")
    assert m1_chunk.symbol_type == "method"
    
    m2_chunk = next(c for c in chunks if c.symbol_name == "MyClass.method_two")
    assert m2_chunk.symbol_type == "method"

def test_extract_nested(tmp_path):
    code = """
def outer():
    def inner():
        pass
    class InnerClass:
        def inner_method(self):
            pass
"""
    f = tmp_path / "nested.py"
    f.write_text(code)
    
    chunks, skipped = extract_chunks_from_file(str(f), str(tmp_path))
    assert not skipped
    assert len(chunks) == 4
    
    assert any(c.symbol_name == "outer" and c.symbol_type == "function" for c in chunks)
    assert any(c.symbol_name == "outer.inner" and c.symbol_type == "function" for c in chunks)
    assert any(c.symbol_name == "outer.InnerClass" and c.symbol_type == "class" for c in chunks)
    assert any(c.symbol_name == "outer.InnerClass.inner_method" and c.symbol_type == "method" for c in chunks)

def test_extract_decorated_and_async(tmp_path):
    code = """
class AsyncThing:
    @staticmethod
    @property
    def decorated_prop():
        pass
        
    async def async_meth(self):
        pass

async def async_func():
    pass
"""
    f = tmp_path / "deco.py"
    f.write_text(code)
    
    chunks, skipped = extract_chunks_from_file(str(f), str(tmp_path))
    assert not skipped
    
    prop_chunk = next(c for c in chunks if c.symbol_name == "AsyncThing.decorated_prop")
    assert prop_chunk.symbol_type == "method"
    assert prop_chunk.start_line == 3 # First decorator line
    assert "@staticmethod" in prop_chunk.source_text
    
    am_chunk = next(c for c in chunks if c.symbol_name == "AsyncThing.async_meth")
    assert am_chunk.symbol_type == "async_method"
    
    af_chunk = next(c for c in chunks if c.symbol_name == "async_func")
    assert af_chunk.symbol_type == "async_function"

def test_syntax_error(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def f(:\n pass")
    
    chunks, skipped = extract_chunks_from_file(str(f), str(tmp_path))
    assert skipped
    assert len(chunks) == 0

def test_encoding_error(tmp_path):
    f = tmp_path / "bad_enc.py"
    # Write non-UTF-8 bytes
    f.write_bytes(b'def f():\n    return "\xff"\n')
    
    chunks, skipped = extract_chunks_from_file(str(f), str(tmp_path))
    assert not skipped
    assert len(chunks) == 1
    assert chunks[0].symbol_name == "f"

def test_empty_file(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("")
    
    chunks, skipped = extract_chunks_from_file(str(f), str(tmp_path))
    assert not skipped
    assert len(chunks) == 0

def test_duplicate_names(tmp_path):
    code = """
def foo():
    pass
def foo():
    pass
"""
    f = tmp_path / "dup.py"
    f.write_text(code)
    
    chunks, skipped = extract_chunks_from_file(str(f), str(tmp_path))
    assert not skipped
    assert len(chunks) == 2
    assert chunks[0].symbol_name == "foo"
    assert chunks[1].symbol_name == "foo#2"

def test_discover_python_files(tmp_path):
    # Setup tree
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignore.py").write_text("pass")
    
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "ignore.py").write_text("pass")
    
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ignore.py").write_text("pass")
    
    (tmp_path / "src" / "deep" / "dir").mkdir(parents=True)
    f = tmp_path / "src" / "deep" / "dir" / "found.py"
    f.write_text("pass")
    
    files = discover_python_files(str(tmp_path))
    assert len(files) == 1
    assert str(f) in files

def test_ast_hash(tmp_path):
    code1 = """
def foo():
    x = 1
    return x
"""
    code2 = """
def foo():
    x   =  1
    return   x
"""
    code3 = """
def foo():
    y = 1
    return y
"""
    f1 = tmp_path / "f1.py"
    f1.write_text(code1)
    f2 = tmp_path / "f2.py"
    f2.write_text(code2)
    f3 = tmp_path / "f3.py"
    f3.write_text(code3)
    
    c1, _ = extract_chunks_from_file(str(f1), str(tmp_path))
    c2, _ = extract_chunks_from_file(str(f2), str(tmp_path))
    c3, _ = extract_chunks_from_file(str(f3), str(tmp_path))
    
    assert c1[0].ast_hash == c2[0].ast_hash
    assert c1[0].ast_hash != c3[0].ast_hash

def test_store_chunks(tmp_path):
    # Set cwd so data/code_chunk_bodies is created in tmp_path
    old_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute('''CREATE TABLE repos (id INTEGER PRIMARY KEY, path TEXT)''')
        conn.execute('''CREATE TABLE code_chunks (
            id INTEGER PRIMARY KEY, repo_id INTEGER, file_path TEXT, 
            symbol_name TEXT, symbol_type TEXT, start_line INTEGER, 
            end_line INTEGER, ast_hash TEXT, embedding_id TEXT)''')
            
        repo_id = 1
        
        c1 = CodeChunk("f.py", "foo", "function", 1, 2, "def foo(): pass", "doc", "hash1")
        c2 = CodeChunk("f.py", "bar", "function", 3, 4, "def bar(): pass", "doc", "hash2")
        
        store_chunks(conn, repo_id, [c1, c2])
        
        # Verify db state
        cursor = conn.execute("SELECT symbol_name FROM code_chunks WHERE repo_id = ?", (repo_id,))
        names = {row[0] for row in cursor.fetchall()}
        assert names == {"foo", "bar"}
        
        # Call again with different chunks
        c3 = CodeChunk("f.py", "baz", "function", 5, 6, "def baz(): pass", "doc", "hash3")
        store_chunks(conn, repo_id, [c3])
        
        cursor = conn.execute("SELECT symbol_name FROM code_chunks WHERE repo_id = ?", (repo_id,))
        names = {row[0] for row in cursor.fetchall()}
        assert names == {"baz"} # foo and bar should be deleted
        
    finally:
        os.chdir(old_cwd)
